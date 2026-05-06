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
    p = argparse.ArgumentParser(description="Evaluate Qwen2-VL on Kvasir-VQA")
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter checkpoint")
    p.add_argument("--compare", type=str, default=None, help="Optional second adapter to compare (e.g., reject vs sft)")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--data_dir", type=str, default="./data/kvasir_vqa")
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--min_pixels", type=int, default=4 * 28 * 28)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--num_examples", type=int, default=10, help="Number of example inferences to show/save")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from rouge_score import rouge_scorer
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CATEGORIES = ["Normal", "Polyps", "Esophagitis", "Ulcerative Colitis", "Instrument"]
CAT2IDX = {c: i for i, c in enumerate(CATEGORIES)}

random.seed(args.seed)
np.random.seed(args.seed)


def load_data(data_dir):
    split_file = Path(data_dir) / "splits.json"
    if not split_file.exists():
        print("[ERROR] splits.json not found. Run train.py or baseline.py first to generate splits.")
        sys.exit(1)

    ds = load_dataset("SimulaMet-HOST/Kvasir-VQA", split="raw", trust_remote_code=True)
    with open(split_file) as f:
        splits = json.load(f)
    return ds, splits

def load_model(model_name, adapter_path, max_pixels, min_pixels):
    print(f"[MODEL] Loading base model: {model_name}")
    print(f"[MODEL] Loading adapter: {adapter_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )

    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_name,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
    )

    return model, processor


@torch.no_grad()
def run_inference(model, processor, ds, indices, max_new_tokens=128):
    cols = ds.column_names
    q_col = next((c for c in cols if c in ["question", "q", "text"]), "question")
    a_col = next((c for c in cols if c in ["answer", "a", "response"]), "answer")
    img_col = next((c for c in cols if c in ["image", "img"]), "image")
    source_col = next((c for c in cols if c in ["source", "category", "label"]), None)
    qtype_col = next((c for c in cols if c in ["q_type", "question_type", "type"]), None)

    predictions = []

    for idx in tqdm(indices, desc="Inference"):
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

        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        entry = {
            "index": idx,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": response,
        }
        if source_col:
            entry["source"] = row[source_col]
        if qtype_col:
            entry["q_type"] = row[qtype_col]

        predictions.append(entry)

    return predictions


def compute_metrics(predictions):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = {}

    exact_matches = [
        p["prediction"].strip().lower() == p["ground_truth"].strip().lower()
        for p in predictions
    ]
    results["overall_exact_match"] = float(np.mean(exact_matches))
    results["total_samples"] = len(predictions)

    cls_preds = [p for p in predictions if p["ground_truth"].strip() in CATEGORIES]
    if cls_preds:
        true_labels = []
        pred_labels = []
        for p in cls_preds:
            gt = p["ground_truth"].strip()
            pred = p["prediction"].strip()
            true_idx = CAT2IDX.get(gt, -1)

            pred_idx = -1
            for cat in CATEGORIES:
                if cat.lower() in pred.lower():
                    pred_idx = CAT2IDX[cat]
                    break
            if true_idx >= 0:
                true_labels.append(true_idx)
                pred_labels.append(pred_idx if pred_idx >= 0 else len(CATEGORIES))

        valid_mask = [p >= 0 and p < len(CATEGORIES) for p in pred_labels]
        t_valid = [t for t, v in zip(true_labels, valid_mask) if v]
        p_valid = [p for p, v in zip(pred_labels, valid_mask) if v]

        if t_valid:
            results["cls_accuracy"] = float(accuracy_score(t_valid, p_valid))
            results["cls_f1_macro"] = float(f1_score(t_valid, p_valid, average="macro", zero_division=0))
            results["cls_confusion_matrix"] = confusion_matrix(
                t_valid, p_valid, labels=list(range(len(CATEGORIES)))
            ).tolist()
            results["cls_report"] = classification_report(
                t_valid, p_valid, target_names=CATEGORIES, zero_division=0
            )
            results["cls_total"] = len(cls_preds)
            results["cls_valid"] = len(t_valid)

    vqa_preds = [p for p in predictions if p["ground_truth"].strip() not in CATEGORIES]
    if vqa_preds:
        vqa_em = [
            p["prediction"].strip().lower() == p["ground_truth"].strip().lower()
            for p in vqa_preds
        ]
        vqa_rouge = [
            scorer.score(p["ground_truth"], p["prediction"])["rougeL"].fmeasure
            for p in vqa_preds
        ]
        results["vqa_exact_match"] = float(np.mean(vqa_em))
        results["vqa_rouge_l"] = float(np.mean(vqa_rouge))
        results["vqa_total"] = len(vqa_preds)

    qtype_metrics = defaultdict(lambda: {"em": [], "rouge": []})
    for p in predictions:
        qt = p.get("q_type", "unknown")
        em = p["prediction"].strip().lower() == p["ground_truth"].strip().lower()
        rl = scorer.score(p["ground_truth"], p["prediction"])["rougeL"].fmeasure
        qtype_metrics[qt]["em"].append(em)
        qtype_metrics[qt]["rouge"].append(rl)

    results["per_question_type"] = {}
    for qt, m in qtype_metrics.items():
        results["per_question_type"][qt] = {
            "exact_match": float(np.mean(m["em"])),
            "rouge_l": float(np.mean(m["rouge"])),
            "count": len(m["em"]),
        }

    return results


def plot_confusion_matrix(cm, out_path, title="Confusion Matrix", acc=None, f1=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    subtitle = ""
    if acc is not None:
        subtitle += f"Acc={acc:.3f}"
    if f1 is not None:
        subtitle += f", F1={f1:.3f}"
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_per_type_accuracy(qtype_data, out_path, title="Per Question-Type Accuracy"):
    types = sorted(qtype_data.keys())
    accs = [qtype_data[t]["exact_match"] for t in types]
    counts = [qtype_data[t]["count"] for t in types]
    rouges = [qtype_data[t]["rouge_l"] for t in types]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(types))
    width = 0.35

    bars1 = ax.bar(x - width / 2, accs, width, label="Exact Match", color="steelblue")
    bars2 = ax.bar(x + width / 2, rouges, width, label="ROUGE-L", color="coral")

    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bar, count in zip(bars1, counts):
        ax.text(bar.get_x() + bar.get_width(), bar.get_height() + 0.02,
                f"n={count}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_comparison(results_list, labels, out_path):
    metrics = ["overall_exact_match"]
    optional = ["cls_accuracy", "cls_f1_macro", "vqa_exact_match", "vqa_rouge_l"]
    for m in optional:
        if any(m in r for r in results_list):
            metrics.append(m)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.8 / len(results_list)

    for i, (res, label) in enumerate(zip(results_list, labels)):
        vals = [res.get(m, 0) for m in metrics]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=label)

    ax.set_xticks(x)
    metric_names = [m.replace("_", " ").title() for m in metrics]
    ax.set_xticklabels(metric_names, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_example_inferences(predictions, out_path, num_examples=10):
    correct = [p for p in predictions if p["prediction"].strip().lower() == p["ground_truth"].strip().lower()]
    incorrect = [p for p in predictions if p["prediction"].strip().lower() != p["ground_truth"].strip().lower()]

    n_correct = min(num_examples // 2, len(correct))
    n_incorrect = min(num_examples - n_correct, len(incorrect))

    samples = random.sample(correct, n_correct) + random.sample(incorrect, n_incorrect)
    random.shuffle(samples)

    lines = []
    lines.append("=" * 80)
    lines.append("EXAMPLE INFERENCES")
    lines.append("=" * 80)

    for i, p in enumerate(samples, 1):
        is_correct = p["prediction"].strip().lower() == p["ground_truth"].strip().lower()
        status = "CORRECT" if is_correct else "WRONG"

        lines.append(f"\n--- Example {i} [{status}] ---")
        if "source" in p:
            lines.append(f"  Source:       {p['source']}")
        if "q_type" in p:
            lines.append(f"  Q-Type:       {p['q_type']}")
        lines.append(f"  Question:     {p['question']}")
        lines.append(f"  Ground Truth: {p['ground_truth']}")
        lines.append(f"  Prediction:   {p['prediction']}")

    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)

    print(text)


def evaluate_adapter(adapter_path, model_name, ds, splits, args, label="model"):
    out_dir = Path(args.output_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(model_name, adapter_path, args.max_pixels, args.min_pixels)
    predictions = run_inference(model, processor, ds, splits["test"], args.max_new_tokens)

    with open(out_dir / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    metrics = compute_metrics(predictions)

    print(f"\n{'=' * 60}")
    print(f"Results for: {label} ({adapter_path})")
    print(f"{'=' * 60}")
    print(f"  Overall Exact Match:  {metrics['overall_exact_match']:.4f}")
    if "cls_accuracy" in metrics:
        print(f"  Classification Acc:   {metrics['cls_accuracy']:.4f}")
        print(f"  Classification F1:    {metrics['cls_f1_macro']:.4f}")
    if "vqa_exact_match" in metrics:
        print(f"  VQA Exact Match:      {metrics['vqa_exact_match']:.4f}")
        print(f"  VQA ROUGE-L:          {metrics['vqa_rouge_l']:.4f}")

    if "cls_report" in metrics:
        print(f"\nClassification Report:\n{metrics['cls_report']}")

    if "per_question_type" in metrics:
        print("\nPer Question-Type:")
        for qt, m in sorted(metrics["per_question_type"].items()):
            print(f"  {qt:30s}: EM={m['exact_match']:.4f}, ROUGE-L={m['rouge_l']:.4f} (n={m['count']})")

    metrics_save = {k: v for k, v in metrics.items() if k != "cls_report"}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_save, f, indent=2)
    if "cls_report" in metrics:
        with open(out_dir / "classification_report.txt", "w") as f:
            f.write(metrics["cls_report"])

    if "cls_confusion_matrix" in metrics:
        cm = np.array(metrics["cls_confusion_matrix"])
        plot_confusion_matrix(
            cm, out_dir / "confusion_matrix.png",
            title=f"{label} — Classification Confusion Matrix",
            acc=metrics.get("cls_accuracy"),
            f1=metrics.get("cls_f1_macro"),
        )

    if "per_question_type" in metrics:
        plot_per_type_accuracy(
            metrics["per_question_type"],
            out_dir / "per_type_accuracy.png",
            title=f"{label} — Per Question-Type Metrics",
        )

    save_example_inferences(predictions, out_dir / "example_inferences.txt", args.num_examples)

    del model
    torch.cuda.empty_cache()

    return metrics


def main():
    print("=" * 70)
    print("Kvasir-VQA — Model Evaluation")
    print(f"GPUs: {args.gpus}")
    print(f"Adapter: {args.adapter_path}")
    if args.compare:
        print(f"Compare: {args.compare}")
    print("=" * 70)

    ds, splits = load_data(args.data_dir)

    adapter_name = Path(args.adapter_path).parent.name
    metrics_primary = evaluate_adapter(
        args.adapter_path, args.model_name, ds, splits, args,
        label=adapter_name,
    )

    all_results = [metrics_primary]
    all_labels = [adapter_name]

    if args.compare:
        compare_name = Path(args.compare).parent.name
        metrics_compare = evaluate_adapter(
            args.compare, args.model_name, ds, splits, args,
            label=compare_name,
        )
        all_results.append(metrics_compare)
        all_labels.append(compare_name)

    baseline_resnet = Path("./checkpoints/baselines/resnet/results.json")
    baseline_zeroshot = Path("./checkpoints/baselines/zeroshot/results.json")

    if baseline_resnet.exists():
        with open(baseline_resnet) as f:
            resnet_data = json.load(f)
        resnet_metrics = {
            "overall_exact_match": resnet_data.get("test_accuracy", 0),
            "cls_accuracy": resnet_data.get("test_accuracy", 0),
            "cls_f1_macro": resnet_data.get("test_f1_macro", 0),
        }
        all_results.append(resnet_metrics)
        all_labels.append("ResNet-50")

    if baseline_zeroshot.exists():
        with open(baseline_zeroshot) as f:
            zs_data = json.load(f)
        zs_metrics = {"overall_exact_match": zs_data.get("overall_exact_match", 0)}
        all_results.append(zs_metrics)
        all_labels.append("Zero-shot")

    if len(all_results) > 1:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison(all_results, all_labels, out_dir / "model_comparison.png")
        print(f"\n[PLOT] Comparison chart saved to {out_dir / 'model_comparison.png'}")

    print("\n[DONE] Evaluation complete.")


if __name__ == "__main__":
    main()
