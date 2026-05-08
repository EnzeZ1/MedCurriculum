# MedCurriculum

Medical visual question answering (VQA) on endoscopic images with curriculum learning. This project fine-tunes **Qwen2.5-VL-7B** on the [Kvasir-VQA](https://datasets.simula.no/kvasir-vqa/) dataset for the CS 567 final project.

## Results

| Method | EM (%) | ROUGE-L (%) | Notes |
|--------|--------|-------------|-------|
| ResNet-50 (cls only) | 99.0 (5-class) | -- | Cannot do VQA |
| Zero-shot Qwen2.5-VL | 0.0 | 1.3 | No alignment/weak knowledge |
| Few-shot (18 examples) | 36.1 | 41.8 | No training needed |
| QLoRA SFT (ours) | 87.4 | 92.3 | 1 epoch, 9 hours |
| Rejection Sampling (ours) | 86.6 | 92.2 | 4 samples/question |
| **Curriculum (ours)** | **87.4** | **92.6** | Hard example mining |

## Files

| Path | Description |
|------|-------------|
| `train/train.py` | QLoRA supervised fine-tuning and rejection sampling |
| `train/mining.py` | Curriculum learning with hard example mining |
| `train/baseline.py` | ResNet-50, zero-shot, and few-shot baselines |
| `eval/eval.py` | Test-set evaluation and metric computation |
| `results/curriculum_result.json` | Final curriculum evaluation result |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

The scripts use the Hugging Face dataset `SimulaMet-HOST/Kvasir-VQA`. The `--data_dir` folder is used to cache the image-level train/validation/test split.

## Usage and Examples

Run the baselines:

```bash
python train/baseline.py \
  --gpus 0 \
  --baseline all \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --data_dir ./data/kvasir_vqa \
  --output_dir ./checkpoints/baselines
```

Run QLoRA supervised fine-tuning:

```bash
python train/train.py \
  --gpus 0 \
  --stage sft \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --data_dir ./data/kvasir_vqa \
  --output_dir ./checkpoints \
  --sft_epochs 1
```

Run rejection sampling from the SFT checkpoint:

```bash
python train/train.py \
  --gpus 0 \
  --stage reject \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --data_dir ./data/kvasir_vqa \
  --output_dir ./checkpoints \
  --sft_checkpoint ./checkpoints/sft/adapter_final \
  --num_samples 4 \
  --reject_epochs 2
```

Run curriculum learning from the SFT checkpoint:

```bash
python train/mining.py \
  --gpus 0 \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --data_dir ./data/kvasir_vqa \
  --sft_checkpoint ./checkpoints/sft/adapter_final \
  --output_dir ./checkpoints_curriculum \
  --mining_samples 20000 \
  --hard_weight 5.0 \
  --phase2_lr 1e-4
```

Evaluate the final curriculum adapter:

```bash
python eval/eval.py \
  --gpus 0 \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter_path ./checkpoints_curriculum/phase2_hard/adapter_final \
  --data_dir ./data/kvasir_vqa \
  --output_dir ./results
```

## Hardware

Experiments used 4-bit NF4 quantization. The main QLoRA runs were tested on an NVIDIA A40 GPU with about 40GB memory.
