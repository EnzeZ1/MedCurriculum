#!/bin/bash

cd /nobackup/enzez/reasoning-chains/cs567
source /nobackup/enzez/reasoning-chains/.venv/bin/activate
export HF_HOME="/nobackup/enzez/hf_cache"
export TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
DATA_DIR="./data/kvasir_vqa"
SFT_CKPT="./checkpoints/sft/adapter_final"

echo "========== Mining Hyperparameter Sweep =========="
echo "SFT checkpoint: $SFT_CKPT"
echo "GPUs: 0, 1, 6, 7"
echo ""

SCORES="./sweep_mining/difficulty_scores.json"
if [ -f "$SCORES" ]; then
    echo "[SKIP] Difficulty scores already exist at $SCORES"
else
    echo "[MINING] Running hard example mining on GPU 0..."
    mkdir -p ./sweep_mining
    python mining.py --gpus 0 \
        --model_name $MODEL --data_dir $DATA_DIR \
        --output_dir ./sweep_mining \
        --sft_checkpoint $SFT_CKPT \
        --phase2_epochs 0 \
        --mining_samples 20000 \
        --max_pixels 100352
    echo "[MINING] Done."
fi

if [ ! -f "$SCORES" ]; then
    echo "[MINING] Running full mining + 1 exp on GPU 0 to get scores..."
    python mining.py --gpus 0 \
        --model_name $MODEL --data_dir $DATA_DIR \
        --output_dir ./sweep_mining \
        --sft_checkpoint $SFT_CKPT \
        --phase2_epochs 1 --phase2_lr 1e-4 \
        --hard_weight 3.0 --batch_size 2 --grad_accum 4 \
        --mining_samples 20000 --max_pixels 100352
fi

echo ""
echo "[INFO] Difficulty scores ready. Starting sweep..."
echo ""

run_exp() {
    local GPU=$1
    local NAME=$2
    local LR=$3
    local EPOCHS=$4
    local HARD_W=$5
    local GA=$6
    local PIXELS=$7

    local OUT="./sweep_mining/$NAME"
    mkdir -p "$OUT"

    cp "$SCORES" "$OUT/difficulty_scores.json"

    echo "[GPU$GPU] $NAME: lr=$LR, epochs=$EPOCHS, hard=$HARD_W, ga=$GA, px=$PIXELS"

    python mining.py --gpus $GPU \
        --model_name $MODEL --data_dir $DATA_DIR \
        --output_dir "$OUT" \
        --sft_checkpoint $SFT_CKPT \
        --phase2_epochs $EPOCHS --phase2_lr $LR \
        --hard_weight $HARD_W \
        --batch_size 2 --grad_accum $GA \
        --mining_samples 1 \
        --max_pixels $PIXELS

    echo "[GPU$GPU] $NAME done."
}

run_gpu0() {
    run_exp        0   exp01_lr5e5_hw3         5e-5   1   3.0   4    100352
    run_exp        0   exp02_lr3e4_hw3         3e-4   1   3.0   4    100352
    run_exp        0   exp03_lr1e4_hw3_e2      1e-4   2   3.0   4    100352
}

run_gpu1() {
    run_exp        1   exp04_lr1e4_hw2         1e-4   1   2.0   4    100352
    run_exp        1   exp05_lr1e4_hw5         1e-4   1   5.0   4    100352
    run_exp        1   exp06_lr1e4_hw10        1e-4   1   10.0  4    100352
}

run_gpu6() {
    run_exp        6   exp07_lr1e4_hw3_ga8     1e-4   1   3.0   8    100352
    run_exp        6   exp08_lr2e4_hw5_e2      2e-4   2   5.0   4    100352
    run_exp        6   exp09_lr5e5_hw5_e2      5e-5   2   5.0   4    100352
}

run_gpu7() {
    run_exp        7   exp10_lr1e4_hw3_hires   1e-4   1   3.0   4    200704
    run_exp        7   exp11_lr2e4_hw3_ga8     2e-4   1   3.0   8    100352
    run_exp        7   exp12_lr1e4_hw5_ga8_e2  1e-4   2   5.0   8    100352
}

echo "Sweep config:"
echo "  exp01: lr=5e-5,  hw=3,  e=1  (low LR)"
echo "  exp02: lr=3e-4,  hw=3,  e=1  (high LR)"
echo "  exp03: lr=1e-4,  hw=3,  e=2  (2 epochs)"
echo "  exp04: lr=1e-4,  hw=2,  e=1  (low weight)"
echo "  exp05: lr=1e-4,  hw=5,  e=1  (high weight)"
echo "  exp06: lr=1e-4,  hw=10, e=1  (very high weight)"
echo "  exp07: lr=1e-4,  hw=3,  ga=8 (big batch)"
echo "  exp08: lr=2e-4,  hw=5,  e=2  (high LR+weight+epochs)"
echo "  exp09: lr=5e-5,  hw=5,  e=2  (low LR+high weight+2ep)"
echo "  exp10: lr=1e-4,  hw=3,  hires (200704px)"
echo "  exp11: lr=2e-4,  hw=3,  ga=8 (high LR+big batch)"
echo "  exp12: lr=1e-4,  hw=5,  ga=8, e=2 (kitchen sink)"
echo ""
echo "Baseline: lr=1e-4, hw=3, e=1, ga=4, px=100352 → 87.4% EM"
echo "=========================================="

run_gpu0 &
PID0=$!
run_gpu1 &
PID1=$!
run_gpu6 &
PID6=$!
run_gpu7 &
PID7=$!

wait $PID0; echo ">>> GPU 0 finished."
wait $PID1; echo ">>> GPU 1 finished."
wait $PID6; echo ">>> GPU 6 finished."
wait $PID7; echo ">>> GPU 7 finished."

echo ""
echo "========== Results Comparison =========="
python -c "
import json
from pathlib import Path

results = []
for exp_dir in sorted(Path('./sweep_mining').iterdir()):
    if not exp_dir.is_dir() or exp_dir.name == 'phase2_hard':
        continue
    log = exp_dir / 'phase2_hard' / 'train_log.json'
    if not log.exists():
        continue
    logs = json.load(open(log))
    eval_entries = [e for e in logs if 'eval_loss' in e]
    train_entries = [e for e in logs if 'loss' in e and 'eval_loss' not in e]
    if eval_entries:
        best_eval = min(e['eval_loss'] for e in eval_entries)
        final_train = train_entries[-1]['loss'] if train_entries else 0
        results.append((exp_dir.name, best_eval, final_train))

results.sort(key=lambda x: x[1])
print(f'{\"Experiment\":<35} {\"Eval Loss\":>10} {\"Train Loss\":>10}')
print('-' * 60)
for name, ev, tr in results:
    marker = ' <-- BEST' if name == results[0][0] else ''
    print(f'{name:<35} {ev:>10.5f} {tr:>10.5f}{marker}')

if results:
    best = results[0][0]
    print(f'\nBest: {best}')
    print(f'Adapter: ./sweep_mining/{best}/phase2_hard/adapter_final')
    print(f'\nRun eval:')
    print(f'HF_HOME=\"/nobackup/enzez/hf_cache\" python eval.py --gpus 0 --adapter_path ./sweep_mining/{best}/phase2_hard/adapter_final --model_name Qwen/Qwen2.5-VL-7B-Instruct --data_dir ./data/kvasir_vqa --output_dir ./results/sweep_best --max_pixels 100352')
"

echo ""
echo "========== Sweep Done =========="
