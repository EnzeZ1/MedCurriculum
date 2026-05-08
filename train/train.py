#!/usr/bin/env python3

import os
import sys
import argparse
import json
import random
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser(description="Kvasir-VQA: QLoRA SFT + Rejection Sampling")
    p.add_argument("--gpus", type=str, default="0", help="Comma-separated physical GPU IDs, e.g. 1,2,6,7")
    p.add_argument("--stage", type=str, default="all", choices=["sft", "reject", "all"])
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--data_dir", type=str, default="./data/kvasir_vqa")
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--sft_checkpoint", type=str, default=None, help="Path to SFT checkpoint (for reject stage)")

    p.add_argument("--sft_epochs", type=int, default=3)
    p.add_argument("--sft_lr", type=float, default=2e-4)
    p.add_argument("--sft_batch_size", type=int, default=2)
    p.add_argument("--sft_grad_accum", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)

    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--num_samples", type=int, default=8, help="N for rejection sampling")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--reject_epochs", type=int, default=2)
    p.add_argument("--reject_lr", type=float, default=1e-4)
    p.add_argument("--rouge_weight", type=float, default=0.5, help="Lambda for ROUGE-L in reward")
    p.add_argument("--reject_rounds", type=int, default=1, help="Number of iterative rejection sampling rounds")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_pixels", type=int, default=256 * 28 * 28, help="Max image pixels for Qwen2-VL")
    p.add_argument("--min_pixels", type=int, default=4 * 28 * 28, help="Min image pixels for Qwen2-VL")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_samples", action="store_true", help="Save rejection sampling data to disk")
    return p.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

CATEGORIES = ["Normal", "Polyps", "Esophagitis", "Ulcerative Colitis", "Instrument"]
CLS_QUESTION = "What category is shown in this endoscopy image? Choose from: Normal, Polyps, Esophagitis, Ulcerative Colitis, Instrument."


def download_and_prepare_dataset(data_dir: str):
    split_file = Path(data_dir) / "splits.json"
    if split_file.exists():
        print(f"[DATA] Loading cached splits from {split_file}")
        with open(split_file) as f:
            splits = json.load(f)
        ds = load_dataset("SimulaMet-HOST/Kvasir-VQA", split="raw", trust_remote_code=True)
        return ds, splits

    print("[DATA] Downloading Kvasir-VQA dataset...")
    ds = load_dataset("SimulaMet-HOST/Kvasir-VQA", split="raw", trust_remote_code=True)

    print(f"[DATA] Columns: {ds.column_names}")
    print(f"[DATA] Total rows: {len(ds)}")
    print(f"[DATA] Example: {ds[0]}")

    cols = ds.column_names
    imgid_col = next((c for c in cols if c in ["imgid", "img_id", "image_id", "image_name"]), None)
    source_col = next((c for c in cols if c in ["source", "category", "label"]), None)

    if imgid_col is None:
        print("[DATA] No image ID column found, using row index groups")
        imgid_col = "__imgid"
        if "image" in cols:
            imgids = []
            for i, row in enumerate(ds):
                img = row.get("image")
                if hasattr(img, "filename") and img.filename:
                    imgids.append(img.filename)
                else:
                    imgids.append(str(i))
            ds = ds.add_column(imgid_col, imgids)
        else:
            ds = ds.add_column(imgid_col, [str(i) for i in range(len(ds))])

    if source_col is None:
        print("[DATA] WARNING: No source/category column found. Using 'unknown'.")
        source_col = "__source"
        ds = ds.add_column(source_col, ["unknown"] * len(ds))

    img2cat = {}
    img2indices = defaultdict(list)
    for i, row in enumerate(ds):
        imgid = str(row[imgid_col])
        img2indices[imgid].append(i)
        if imgid not in img2cat:
            img2cat[imgid] = row[source_col]

    unique_imgs = list(img2cat.keys())
    cats = [img2cat[img] for img in unique_imgs]
    print(f"[DATA] Unique images: {len(unique_imgs)}")

    train_imgs, temp_imgs, train_cats, temp_cats = train_test_split(
        unique_imgs, cats, test_size=0.2, stratify=cats, random_state=args.seed
    )
    val_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=0.5, stratify=temp_cats, random_state=args.seed
    )

    splits = {
        "train": [idx for img in train_imgs for idx in img2indices[img]],
        "val":   [idx for img in val_imgs   for idx in img2indices[img]],
        "test":  [idx for img in test_imgs  for idx in img2indices[img]],
    }

    print(f"[DATA] Split sizes — train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    with open(split_file, "w") as f:
        json.dump(splits, f)

    return ds, splits


class KvasirSFTDataset(Dataset):
    def __init__(self, hf_dataset, indices, processor, max_length=512):
        self.ds = hf_dataset
        self.indices = indices
        self.processor = processor
        self.max_length = max_length

        cols = hf_dataset.column_names
        self.q_col = next((c for c in cols if c in ["question", "q", "text"]), "question")
        self.a_col = next((c for c in cols if c in ["answer", "a", "response"]), "answer")
        self.img_col = next((c for c in cols if c in ["image", "img"]), "image")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.ds[self.indices[idx]]
        image = row[self.img_col]
        question = row[self.q_col]
        answer = row[self.a_col]

        if not isinstance(image, (type(None),)):
            image = image.convert("RGB") if hasattr(image, "convert") else image

        full_messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": str(answer)},
            ]},
        ]

        prompt_messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]},
        ]

        full_text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

        full_inputs = self.processor(
            text=[full_text], images=[image], return_tensors="pt",
            padding=False, truncation=True, max_length=self.max_length,
        )

        prompt_inputs = self.processor(
            text=[prompt_text], images=[image], return_tensors="pt",
            padding=False, truncation=True, max_length=self.max_length,
        )

        input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)
        pixel_values = full_inputs.get("pixel_values")
        image_grid_thw = full_inputs.get("image_grid_thw")

        labels = input_ids.clone()
        prompt_len = prompt_inputs["input_ids"].shape[-1]
        labels[:prompt_len] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values.squeeze(0) if pixel_values.dim() > 3 else pixel_values
        if image_grid_thw is not None:
            result["image_grid_thw"] = image_grid_thw.squeeze(0) if image_grid_thw.dim() > 2 else image_grid_thw

        return result


class SFTCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(b["input_ids"].shape[0] for b in batch)

        input_ids = []
        attention_mask = []
        labels = []
        pixel_values = []
        image_grid_thw = []

        for b in batch:
            seq_len = b["input_ids"].shape[0]
            pad_len = max_len - seq_len

            input_ids.append(torch.cat([b["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
            labels.append(torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))

            if "pixel_values" in b and b["pixel_values"] is not None:
                pixel_values.append(b["pixel_values"])
            if "image_grid_thw" in b and b["image_grid_thw"] is not None:
                image_grid_thw.append(b["image_grid_thw"])

        result = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }
        if pixel_values:
            result["pixel_values"] = torch.cat(pixel_values, dim=0)
        if image_grid_thw:
            result["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)

        return result


class RejectionSFTDataset(Dataset):
    def __init__(self, hf_dataset, entries, processor, max_length=512):
        self.ds = hf_dataset
        self.entries = entries
        self.processor = processor
        self.max_length = max_length
        self.img_col = next((c for c in hf_dataset.column_names if c in ["image", "img"]), "image")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        row = self.ds[entry["index"]]
        image = row[self.img_col]
        question = entry["question"]
        answer = entry["best_response"]

        if hasattr(image, "convert"):
            image = image.convert("RGB")

        full_messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": str(answer)},
            ]},
        ]
        prompt_messages = full_messages[:1]

        full_text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

        full_inputs = self.processor(
            text=[full_text], images=[image], return_tensors="pt",
            padding=False, truncation=True, max_length=self.max_length,
        )
        prompt_inputs = self.processor(
            text=[prompt_text], images=[image], return_tensors="pt",
            padding=False, truncation=True, max_length=self.max_length,
        )

        input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_inputs["input_ids"].shape[-1]] = -100

        result = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        pv = full_inputs.get("pixel_values")
        igt = full_inputs.get("image_grid_thw")
        if pv is not None:
            result["pixel_values"] = pv.squeeze(0) if pv.dim() > 3 else pv
        if igt is not None:
            result["image_grid_thw"] = igt.squeeze(0) if igt.dim() > 2 else igt
        return result


def load_model_and_processor(model_name, lora_r, lora_alpha, lora_dropout, max_pixels, min_pixels):
    print(f"[MODEL] Loading {model_name} with NF4 quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )

    processor = AutoProcessor.from_pretrained(
        model_name,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


def attach_lora(model, lora_r, lora_alpha, lora_dropout):
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def compute_reward(prediction: str, ground_truth: str, rouge_weight: float = 0.5) -> float:
    pred = prediction.strip().lower()
    gt = ground_truth.strip().lower()

    exact = 1.0 if pred == gt else 0.0
    rouge_l = _rouge.score(gt, pred)["rougeL"].fmeasure

    return exact + rouge_weight * rouge_l


def run_sft(model, processor, ds, splits, args):
    print("\n" + "=" * 70)
    print("STAGE 1: Supervised Fine-Tuning (QLoRA)")
    print("=" * 70)

    model = attach_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    train_dataset = KvasirSFTDataset(ds, splits["train"], processor, args.max_length)
    val_dataset = KvasirSFTDataset(ds, splits["val"], processor, args.max_length)
    collator = SFTCollator(processor.tokenizer.pad_token_id)

    sft_output = os.path.join(args.output_dir, "sft")

    training_args = TrainingArguments(
        output_dir=sft_output,
        num_train_epochs=args.sft_epochs,
        per_device_train_batch_size=args.sft_batch_size,
        per_device_eval_batch_size=args.sft_batch_size,
        gradient_accumulation_steps=args.sft_grad_accum,
        learning_rate=args.sft_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        bf16=False,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print(f"[SFT] Training on {len(train_dataset)} samples, validating on {len(val_dataset)}...")
    trainer.train()

    adapter_path = os.path.join(sft_output, "adapter_final")
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    print(f"[SFT] Adapter saved to {adapter_path}")

    log_history = trainer.state.log_history
    with open(os.path.join(sft_output, "train_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)

    return model, adapter_path


@torch.no_grad()
def sample_responses(model, processor, ds, indices, num_samples, temperature, top_p, max_new_tokens=128):
    model.eval()

    cols = ds.column_names
    q_col = next((c for c in cols if c in ["question", "q", "text"]), "question")
    a_col = next((c for c in cols if c in ["answer", "a", "response"]), "answer")
    img_col = next((c for c in cols if c in ["image", "img"]), "image")

    all_samples = []

    for i, idx in enumerate(tqdm(indices, desc="Sampling responses")):
        row = ds[idx]
        image = row[img_col]
        question = row[q_col]
        ground_truth = str(row[a_col])

        if hasattr(image, "convert"):
            image = image.convert("RGB")

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        responses = []
        for _ in range(num_samples):
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )

            new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
            response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            responses.append(response)

        all_samples.append({
            "index": idx,
            "question": question,
            "ground_truth": ground_truth,
            "responses": responses,
        })

    return all_samples


def rejection_filter(samples, rouge_weight=0.5, min_reward=0.0):
    filtered = []
    total = 0
    kept = 0

    for item in samples:
        gt = item["ground_truth"]
        best_response = None
        best_reward = -1.0

        for resp in item["responses"]:
            reward = compute_reward(resp, gt, rouge_weight)
            if reward > best_reward:
                best_reward = reward
                best_response = resp

        total += 1
        if best_reward > min_reward:
            filtered.append({
                "index": item["index"],
                "question": item["question"],
                "best_response": best_response,
                "reward": best_reward,
            })
            kept += 1

    print(f"[REJECT] Kept {kept}/{total} samples (dropped {total - kept} where all responses scored <= {min_reward})")
    return filtered


def run_rejection_sampling(model, processor, ds, splits, args, adapter_path=None):
    print("\n" + "=" * 70)
    print("STAGE 2: Rejection Sampling")
    print("=" * 70)

    if adapter_path and not isinstance(model, PeftModel):
        print(f"[REJECT] Loading SFT adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)

    for round_num in range(1, args.reject_rounds + 1):
        print(f"\n--- Rejection Sampling Round {round_num}/{args.reject_rounds} ---")

        print("[REJECT] Sampling responses from training set...")
        train_indices = splits["train"]

        if len(train_indices) > 10000:
            print(f"[REJECT] Subsampling to 10000 from {len(train_indices)} for efficiency")
            train_indices = random.sample(train_indices, 10000)

        samples = sample_responses(
            model, processor, ds, train_indices,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        if args.save_samples:
            samples_path = os.path.join(args.output_dir, f"rejection_samples_round{round_num}.json")
            with open(samples_path, "w") as f:
                json.dump(samples, f, indent=2)
            print(f"[REJECT] Samples saved to {samples_path}")

        filtered = rejection_filter(samples, rouge_weight=args.rouge_weight)

        if len(filtered) < 100:
            print(f"[REJECT] Only {len(filtered)} samples passed filtering. Skipping retraining.")
            break

        rewards = [e["reward"] for e in filtered]
        print(f"[REJECT] Reward stats — mean: {np.mean(rewards):.3f}, median: {np.median(rewards):.3f}, "
              f"min: {np.min(rewards):.3f}, max: {np.max(rewards):.3f}")

        print(f"[REJECT] Retraining on {len(filtered)} filtered samples...")


        if isinstance(model, PeftModel):
            for name, param in model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True
        else:
            model = attach_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)

        reject_dataset = RejectionSFTDataset(ds, filtered, processor, args.max_length)
        val_dataset = KvasirSFTDataset(ds, splits["val"], processor, args.max_length)
        collator = SFTCollator(processor.tokenizer.pad_token_id)

        reject_output = os.path.join(args.output_dir, f"reject_round{round_num}")

        training_args = TrainingArguments(
            output_dir=reject_output,
            num_train_epochs=args.reject_epochs,
            per_device_train_batch_size=args.sft_batch_size,
            per_device_eval_batch_size=args.sft_batch_size,
            gradient_accumulation_steps=args.sft_grad_accum,
            learning_rate=args.reject_lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            fp16=True,
            bf16=False,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=args.num_workers,
            remove_unused_columns=False,
            report_to="none",
            seed=args.seed,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=reject_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
        )

        trainer.train()

        adapter_path = os.path.join(reject_output, "adapter_final")
        model.save_pretrained(adapter_path)
        processor.save_pretrained(adapter_path)
        print(f"[REJECT] Round {round_num} adapter saved to {adapter_path}")

        with open(os.path.join(reject_output, "train_log.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)

    return model, adapter_path


def main():
    print("=" * 70)
    print("Kvasir-VQA Training Pipeline")
    print(f"GPUs: {args.gpus} | Stage: {args.stage}")
    print(f"Model: {args.model_name}")
    print("=" * 70)

    ds, splits = download_and_prepare_dataset(args.data_dir)

    model, processor = load_model_and_processor(
        args.model_name, args.lora_r, args.lora_alpha, args.lora_dropout,
        args.max_pixels, args.min_pixels,
    )

    adapter_path = args.sft_checkpoint

    if args.stage in ("sft", "all"):
        model, adapter_path = run_sft(model, processor, ds, splits, args)

    if args.stage in ("reject", "all"):
        if adapter_path is None:
            adapter_path = os.path.join(args.output_dir, "sft", "adapter_final")
            if not os.path.exists(adapter_path):
                print("[ERROR] No SFT checkpoint found. Run --stage sft first or provide --sft_checkpoint.")
                sys.exit(1)
        model, adapter_path = run_rejection_sampling(model, processor, ds, splits, args, adapter_path)

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Final adapter: {adapter_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
