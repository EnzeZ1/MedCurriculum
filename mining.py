#!/usr/bin/env python3

import os, sys, argparse, json, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--data_dir", type=str, default="./data/kvasir_vqa")
    p.add_argument("--output_dir", type=str, default="./checkpoints_curriculum")
    p.add_argument("--sft_checkpoint", type=str, required=True)
    p.add_argument("--phase2_epochs", type=int, default=1)
    p.add_argument("--phase2_lr", type=float, default=1e-4)
    p.add_argument("--hard_weight", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--min_pixels", type=int, default=4 * 28 * 28)
    p.add_argument("--mining_samples", type=int, default=5000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import PeftModel
from datasets import load_dataset

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def load_data(data_dir):
    ds = load_dataset("SimulaMet-HOST/Kvasir-VQA", split="raw", trust_remote_code=True)
    with open(Path(data_dir) / "splits.json") as f:
        splits = json.load(f)
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


class SFTCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(b["input_ids"].shape[0] for b in batch)
        input_ids, attention_mask, labels = [], [], []
        pixel_values, image_grid_thw = [], []
        for b in batch:
            pad_len = max_len - b["input_ids"].shape[0]
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


@torch.no_grad()
def find_hard_examples(model, processor, ds, indices, max_samples=20000):
    model.eval()
    cols = ds.column_names
    q_col = next((c for c in cols if c in ["question", "q", "text"]), "question")
    a_col = next((c for c in cols if c in ["answer", "a", "response"]), "answer")
    img_col = next((c for c in cols if c in ["image", "img"]), "image")

    eval_indices = random.sample(indices, min(max_samples, len(indices)))
    difficulties = {}

    for idx in tqdm(eval_indices, desc="Mining hard examples"):
        row = ds[idx]
        image = row[img_col]
        question = row[q_col]
        ground_truth = str(row[a_col]).strip().lower()

        if hasattr(image, "convert"):
            image = image.convert("RGB")

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        prediction = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

        difficulties[idx] = 0.0 if prediction == ground_truth else 1.0

    for idx in indices:
        if idx not in difficulties:
            difficulties[idx] = 0.5

    n_hard = sum(1 for v in difficulties.values() if v > 0.5)
    n_easy = sum(1 for v in difficulties.values() if v < 0.5)
    n_unk = sum(1 for v in difficulties.values() if v == 0.5)
    print(f"[MINING] Hard: {n_hard}, Easy: {n_easy}, Unknown: {n_unk}")
    if n_hard + n_easy > 0:
        print(f"[MINING] Error rate: {n_hard/(n_hard+n_easy)*100:.1f}%")

    return difficulties


def main():
    print("=" * 70)
    print("Curriculum SFT with Hard Example Mining")
    print(f"SFT checkpoint: {args.sft_checkpoint}")
    print(f"Hard weight: {args.hard_weight}x | Phase 2 LR: {args.phase2_lr}")
    print("=" * 70)

    ds, splits = load_data(args.data_dir)


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False,
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.float16, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base_model, args.sft_checkpoint, is_trainable=True)

    processor = AutoProcessor.from_pretrained(
        args.model_name, max_pixels=args.max_pixels, min_pixels=args.min_pixels,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token


    scores_path = os.path.join(args.output_dir, "difficulty_scores.json")
    if os.path.exists(scores_path):
        print(f"\n[MINING] Loading cached scores from {scores_path}")
        raw = json.load(open(scores_path))
        difficulties = {int(k): v for k, v in raw.items()}

        for idx in splits["train"]:
            if idx not in difficulties:
                difficulties[idx] = 0.5
    else:
        print("\n[MINING] Finding hard examples...")
        difficulties = find_hard_examples(model, processor, ds, splits["train"], args.mining_samples)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(scores_path, "w") as f:
            json.dump({str(k): v for k, v in difficulties.items()}, f)


    print("\n" + "=" * 70)
    print("PHASE 2: Hard Example Oversampling")
    print("=" * 70)

    train_indices = splits["train"]
    weights = [args.hard_weight if difficulties.get(idx, 0.5) > 0.5 else 1.0 for idx in train_indices]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_indices), replacement=True)

    hard_count = sum(1 for w in weights if w > 1.0)
    print(f"[PHASE 2] {hard_count} hard ({args.hard_weight}x), {len(weights)-hard_count} easy (1x)")


    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[PHASE 2] Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")

    train_dataset = KvasirSFTDataset(ds, train_indices, processor, args.max_length)
    val_dataset = KvasirSFTDataset(ds, splits["val"], processor, args.max_length)
    collator = SFTCollator(processor.tokenizer.pad_token_id)

    phase2_output = os.path.join(args.output_dir, "phase2_hard")

    training_args = TrainingArguments(
        output_dir=phase2_output,
        num_train_epochs=args.phase2_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.phase2_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True, bf16=False,
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

    class CurriculumTrainer(Trainer):
        def get_train_dataloader(self):
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=True,
            )

    trainer = CurriculumTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=collator,
    )

    print(f"\n[PHASE 2] Training...")
    trainer.train()

    adapter_path = os.path.join(phase2_output, "adapter_final")
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)

    with open(os.path.join(phase2_output, "train_log.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Done! Adapter saved to {adapter_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
