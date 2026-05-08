# MedCurriculum

Medical VQA on endoscopic images with curriculum learning. Fine-tunes Qwen2.5-VL-7B on the [Kvasir-VQA](https://datasets.simula.no/kvasir-vqa/) dataset.

**CS 567 Final Project**

## Results

## Results

## Results

| Method | Task | EM / Accuracy | ROUGE-L | Notes |
|--------|------|---------------|---------|-------|
| ResNet-50 | 5-class classification | 99.0% | -- | Cannot do VQA |
| Zero-shot Qwen2.5-VL | VQA | 0.0% | 1.3% | 27.6% contains GT |
| Few-shot Qwen2.5-VL | VQA | 36.1% | 41.8% | 18 examples |
| QLoRA SFT | VQA | 87.4% EM | 92.3% | 1 epoch, 9 hours |
| Rejection Sampling | VQA | 86.6% | 92.2% | 4 samples/question |
| **Curriculum** | VQA | **87.4%** | **92.6%** | Hard example mining |

## Files

| File | Description |
|------|-------------|
| `train.py` | QLoRA SFT + rejection sampling training |
| `baseline.py` | ResNet-50, zero-shot, and few-shot baselines |
| `mining.py` | Curriculum learning with hard example mining |
| `eval.py` | Test set evaluation and metric computation |

## Usage

```bash
# Install
pip install torch transformers accelerate peft bitsandbytes datasets rouge-score scikit-learn matplotlib

# 1. SFT
python train.py --gpus 0 --stage sft --model_name Qwen/Qwen2.5-VL-7B-Instruct --data_dir ./data/kvasir_vqa --output_dir ./checkpoints

# 2. Curriculum (requires SFT checkpoint)
python mining.py --gpus 0 --model_name Qwen/Qwen2.5-VL-7B-Instruct --data_dir ./data/kvasir_vqa --output_dir ./checkpoints_curriculum --sft_checkpoint ./checkpoints/sft/adapter_final --mining_samples 20000 --hard_weight 5.0 --phase2_lr 1e-4

# 3. Eval
python eval.py --gpus 0 --adapter_path ./checkpoints_curriculum/phase2_hard/adapter_final --model_name Qwen/Qwen2.5-VL-7B-Instruct --data_dir ./data/kvasir_vqa --output_dir ./results
```

## Hardware

Tested on NVIDIA A40 (~40GB). Uses 4-bit NF4 quantization.
