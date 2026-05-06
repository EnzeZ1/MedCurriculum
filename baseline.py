#!/usr/bin/env python3

import os
import sys
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Kvasir-VQA Baselines")
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--baseline", type=str, default="all", choices=["resnet", "zeroshot", "fewshot", "all"])
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--data_dir", type=str, default="./data/kvasir_vqa")
    p.add_argument("--output_dir", type=str, default="./checkpoints/baselines")

    p.add_argument("--resnet_epochs", type=int, default=20)
    p.add_argument("--resnet_lr", type=float, default=1e-4)
    p.add_argument("--resnet_batch_size", type=int, default=32)
    p.add_argument("--img_size", type=int, default=224)

    p.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--min_pixels", type=int, default=4 * 28 * 28)
    p.add_argument("--max_new_tokens", type=int, default=128)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CATEGORIES = ["Normal", "Polyps", "Esophagitis", "Ulcerative Colitis", "Instrument"]
CAT2IDX = {c: i for i, c in enumerate(CATEGORIES)}

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def load_and_split(data_dir):
    split_file = Path(data_dir) / "splits.json"
    ds = load_dataset("SimulaMet-HOST/Kvasir-VQA", split="raw", trust_remote_code=True)

    if split_file.exists():
        with open(split_file) as f:
            splits = json.load(f)
        return ds, splits

    cols = ds.column_names
    imgid_col = next((c for c in cols if c in ["imgid", "img_id", "image_id", "image_name"]), None)
    source_col = next((c for c in cols if c in ["source", "category", "label"]), None)

    if imgid_col is None:
        imgid_col = "__imgid"
        imgids = []
        for i, row in enumerate(ds):
            img = row.get("image")
            if hasattr(img, "filename") and img.filename:
                imgids.append(img.filename)
            else:
                imgids.append(str(i))
        ds = ds.add_column(imgid_col, imgids)

    if source_col is None:
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

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    with open(split_file, "w") as f:
        json.dump(splits, f)

    return ds, splits


def get_classification_rows(ds, indices):
    cols = ds.column_names
    source_col = next((c for c in cols if c in ["source", "category", "label"]), None)
    imgid_col = next((c for c in cols if c in ["imgid", "img_id", "image_id", "image_name", "__imgid"]), None)

    seen = set()
    rows = []
    for idx in indices:
        row = ds[idx]
        key = str(row.get(imgid_col, idx)) if imgid_col else str(idx)
        if key not in seen:
            seen.add(key)
            label = row[source_col] if source_col else "unknown"
            rows.append({"index": idx, "label": label})
    return rows


class ResNetDataset(Dataset):
    def __init__(self, hf_ds, cls_rows, transform, img_col="image"):
        self.hf_ds = hf_ds
        self.cls_rows = cls_rows
        self.transform = transform
        self.img_col = img_col

    def __len__(self):
        return len(self.cls_rows)

    def __getitem__(self, idx):
        entry = self.cls_rows[idx]
        row = self.hf_ds[entry["index"]]
        image = row[self.img_col].convert("RGB")
        image = self.transform(image)
        label = CAT2IDX.get(entry["label"], 0)
        return image, label


def run_resnet_baseline(ds, splits, args):
    print("\n" + "=" * 70)
    print("BASELINE 1: ResNet-50 Fine-tuning (5-class classification)")
    print("=" * 70)

    img_col = next((c for c in ds.column_names if c in ["image", "img"]), "image")

    train_rows = get_classification_rows(ds, splits["train"])
    val_rows = get_classification_rows(ds, splits["val"])
    test_rows = get_classification_rows(ds, splits["test"])

    print(f"[RESNET] Classification images — train: {len(train_rows)}, val: {len(val_rows)}, test: {len(test_rows)}")

    train_transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ResNetDataset(ds, train_rows, train_transform, img_col)
    val_ds = ResNetDataset(ds, val_rows, eval_transform, img_col)
    test_ds = ResNetDataset(ds, test_rows, eval_transform, img_col)

    train_loader = DataLoader(train_ds, batch_size=args.resnet_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.resnet_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.resnet_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(CATEGORIES))

    gpu_ids = list(range(torch.cuda.device_count()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"[RESNET] Using DataParallel on {len(gpu_ids)} GPUs")

    optimizer = AdamW(model.parameters(), lr=args.resnet_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.resnet_epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    out_dir = Path(args.output_dir) / "resnet"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.resnet_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.resnet_epochs} [train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.resnet_epochs} [val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += images.size(0)

        val_loss = total_loss / total
        val_acc = correct / total

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(raw_model.state_dict(), out_dir / "best_model.pt")

    print("\n[RESNET] Evaluating on test set...")
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test"):
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n[RESNET] Test Accuracy: {test_acc:.4f}")
    print(f"[RESNET] Test F1 (macro): {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CATEGORIES))

    results = {
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "confusion_matrix": cm.tolist(),
        "history": history,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs_range = range(1, args.resnet_epochs + 1)

    axes[0].plot(epochs_range, history["train_loss"], label="Train")
    axes[0].plot(epochs_range, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("ResNet-50 Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history["train_acc"], label="Train")
    axes[1].plot(epochs_range, history["val_acc"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("ResNet-50 Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "resnet_training_curves.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"ResNet-50 Confusion Matrix (Acc={test_acc:.3f}, F1={test_f1:.3f})")
    plt.tight_layout()
    plt.savefig(out_dir / "resnet_confusion_matrix.png", dpi=150)
    plt.close()

    print(f"[RESNET] Plots saved to {out_dir}")
    return results


@torch.no_grad()
def run_zeroshot_baseline(ds, splits, args):
    print("\n" + "=" * 70)
    print("BASELINE 2: Zero-shot Qwen2.5-VL-7B (Classification + VQA)")
    print("=" * 70)

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )
    model.eval()

    cols = ds.column_names
    q_col = next((c for c in cols if c in ["question", "q", "text"]), "question")
    a_col = next((c for c in cols if c in ["answer", "a", "response"]), "answer")
    img_col = next((c for c in cols if c in ["image", "img"]), "image")
    source_col = next((c for c in cols if c in ["source", "category", "label"]), None)
    qtype_col = next((c for c in cols if c in ["q_type", "question_type", "type"]), None)

    out_dir = Path(args.output_dir) / "zeroshot"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_indices = splits["test"]
    print(f"[ZEROSHOT] Evaluating on {len(test_indices)} test samples...")

    predictions = []
    for idx in tqdm(test_indices, desc="Zero-shot inference"):
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

        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        entry = {
            "question": question,
            "ground_truth": ground_truth,
            "prediction": response,
        }
        if source_col:
            entry["source"] = row[source_col]
        if qtype_col:
            entry["q_type"] = row[qtype_col]

        predictions.append(entry)

    with open(out_dir / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    exact_matches = [p["prediction"].strip().lower() == p["ground_truth"].strip().lower() for p in predictions]
    overall_acc = np.mean(exact_matches)
    print(f"\n[ZEROSHOT] Overall Exact-Match Accuracy: {overall_acc:.4f}")


    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(p["ground_truth"], p["prediction"])["rougeL"].fmeasure for p in predictions]
    overall_rouge = np.mean(rouge_scores)
    print(f"[ZEROSHOT] Overall ROUGE-L: {overall_rouge:.4f}")

    cls_preds = [p for p in predictions if p["ground_truth"].strip() in CATEGORIES]
    if cls_preds:
        cls_true = [CAT2IDX.get(p["ground_truth"].strip(), -1) for p in cls_preds]
        cls_pred_labels = []
        for p in cls_preds:
            pred = p["prediction"].strip()
            matched = -1
            for cat in CATEGORIES:
                if cat.lower() in pred.lower():
                    matched = CAT2IDX[cat]
                    break
            cls_pred_labels.append(matched)

        valid = [(t, p) for t, p in zip(cls_true, cls_pred_labels) if t >= 0 and p >= 0]
        if valid:
            t_valid, p_valid = zip(*valid)
            cls_acc = accuracy_score(t_valid, p_valid)
            cls_f1 = f1_score(t_valid, p_valid, average="macro", zero_division=0)
            print(f"[ZEROSHOT] Classification Accuracy: {cls_acc:.4f} ({len(valid)} samples)")
            print(f"[ZEROSHOT] Classification F1 (macro): {cls_f1:.4f}")

    if qtype_col:
        qtype_results = defaultdict(list)
        for p in predictions:
            qt = p.get("q_type", "unknown")
            em = p["prediction"].strip().lower() == p["ground_truth"].strip().lower()
            qtype_results[qt].append(em)

        print("\n[ZEROSHOT] Per Question-Type Accuracy:")
        qtype_accs = {}
        for qt, matches in sorted(qtype_results.items()):
            acc = np.mean(matches)
            qtype_accs[qt] = {"accuracy": acc, "count": len(matches)}
            print(f"  {qt:30s}: {acc:.4f} ({len(matches)} samples)")

        fig, ax = plt.subplots(figsize=(10, 5))
        types = list(qtype_accs.keys())
        accs = [qtype_accs[t]["accuracy"] for t in types]
        counts = [qtype_accs[t]["count"] for t in types]

        bars = ax.bar(range(len(types)), accs, color="steelblue")
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels(types, rotation=30, ha="right")
        ax.set_ylabel("Exact-Match Accuracy")
        ax.set_title(f"Zero-shot Qwen2.5-VL — Per Question-Type Accuracy (Overall: {overall_acc:.3f})")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"n={count}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.savefig(out_dir / "zeroshot_per_type_accuracy.png", dpi=150)
        plt.close()

    results = {
        "overall_exact_match": overall_acc,
        "overall_rouge_l": overall_rouge,
        "total_samples": len(predictions),
    }
    if qtype_col:
        results["per_question_type"] = {qt: float(np.mean(m)) for qt, m in qtype_results.items()}

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[ZEROSHOT] Results saved to {out_dir}")
    return results


FEW_SHOT_EXAMPLES = """Here are examples of how to answer medical image questions. Reply ONLY with the answer, exactly like the examples:

Q: Does this image contain any finding?
A: yes

Q: Does this image contain any finding?
A: no

Q: Are there any abnormalities in the image? Check all that are present.
A: polyp

Q: Are there any abnormalities in the image? Check all that are present.
A: ulcerative colitis

Q: How many polyps are in the image?
A: 2

Q: How many instrumnets are in the image?
A: 0

Q: What type of procedure is the image taken from?
A: colonoscopy

Q: Is there text?
A: yes

Q: Is there a green/black box artefact?
A: no

Q: What is the size of the polyp?
A: 11-20mm

Q: What type of polyp is present?
A: paris iia

Q: What color is the abnormality? If more than one separate with ;
A: red; pink; brown

Q: Where in the image is the abnormality?
A: center; center-left; lower-center

Q: Have all polyps been removed?
A: not relevant

Q: Is this finding easy to detect?
A: yes

Q: Are there any instruments in the image? Check all that are present.
A: biopsy forceps

Q: Are there any anatomical landmarks in the image? Check all that are present.
A: z-line

Now answer the following question about the provided medical image. Reply with ONLY the answer, nothing else."""


@torch.no_grad()
def run_fewshot_baseline(ds, splits, args):
    print("\n" + "=" * 70)
    print("BASELINE 3: Few-shot Qwen2.5-VL-7B (Text-only examples)")
    print("=" * 70)

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )
    model.eval()

    cols = ds.column_names
    q_col = next((c for c in cols if c in ["question", "q", "text"]), "question")
    a_col = next((c for c in cols if c in ["answer", "a", "response"]), "answer")
    img_col = next((c for c in cols if c in ["image", "img"]), "image")
    source_col = next((c for c in cols if c in ["source", "category", "label"]), None)
    qtype_col = next((c for c in cols if c in ["q_type", "question_type", "type"]), None)

    out_dir = Path(args.output_dir) / "fewshot"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_indices = splits["test"]
    print(f"[FEWSHOT] Evaluating on {len(test_indices)} test samples...")

    predictions = []
    for idx in tqdm(test_indices, desc="Few-shot inference"):
        row = ds[idx]
        image = row[img_col]
        question = row[q_col]
        ground_truth = str(row[a_col])

        if hasattr(image, "convert"):
            image = image.convert("RGB")

        messages = [
            {"role": "system", "content": FEW_SHOT_EXAMPLES},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


        response = response.split("\n")[0].strip().rstrip(".")
        for prefix in ["A:", "Answer:", "The answer is:", "The answer is "]:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        entry = {
            "question": question,
            "ground_truth": ground_truth,
            "prediction": response,
        }
        if source_col:
            entry["source"] = row[source_col]
        if qtype_col:
            entry["q_type"] = row[qtype_col]

        predictions.append(entry)

    with open(out_dir / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


    from rouge_score import rouge_scorer as rs
    rouge = rs.RougeScorer(["rougeL"], use_stemmer=True)

    exact_matches = [p["prediction"].strip().lower() == p["ground_truth"].strip().lower() for p in predictions]
    overall_acc = np.mean(exact_matches)
    rouge_scores = [rouge.score(p["ground_truth"], p["prediction"])["rougeL"].fmeasure for p in predictions]
    overall_rouge = np.mean(rouge_scores)

    contains = sum(1 for p in predictions if p["ground_truth"].strip().lower() in p["prediction"].strip().lower())
    contains_rate = contains / len(predictions)

    print(f"\n[FEWSHOT] Overall Exact-Match Accuracy: {overall_acc:.4f}")
    print(f"[FEWSHOT] Overall ROUGE-L: {overall_rouge:.4f}")
    print(f"[FEWSHOT] Contains GT: {contains_rate:.4f} ({contains}/{len(predictions)})")

    results = {
        "overall_exact_match": overall_acc,
        "overall_rouge_l": overall_rouge,
        "contains_gt": contains_rate,
        "total_samples": len(predictions),
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[FEWSHOT] Results saved to {out_dir}")
    return results


def main():
    print("=" * 70)
    print("Kvasir-VQA Baselines")
    print(f"GPUs: {args.gpus} | Baseline: {args.baseline}")
    print("=" * 70)

    ds, splits = load_and_split(args.data_dir)

    if args.baseline in ("resnet", "all"):
        run_resnet_baseline(ds, splits, args)

    if args.baseline in ("zeroshot", "all"):
        run_zeroshot_baseline(ds, splits, args)

    if args.baseline in ("fewshot", "all"):
        run_fewshot_baseline(ds, splits, args)

    print("\n[DONE] All baselines complete.")


if __name__ == "__main__":
    main()
