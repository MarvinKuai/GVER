#!/bin/bash
# Full training pipeline: pack -> train quantizer -> train retriever -> eval
# Assumes CLIP-SF features for train/val/test have all been extracted.
set -e
PRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PRO_ROOT"

LOG_DIR="$PRO_ROOT/logs/full"
mkdir -p $LOG_DIR

CONFIG="$PRO_ROOT/configs/pro_oven_infoseek.yaml"

echo "[full] step 1: pack all splits (cand + train + val + test)"
python pack_for_pro.py \
    --clipsf_root "$PRO_ROOT/outputs/clipsf_feats" \
    --out_root "$PRO_ROOT/outputs/data_packed" \
    --datasets OVEN INFOSEEK \
    --splits train val test \
    2>&1 | tee $LOG_DIR/pack.log

echo "[full] step 2: train quantizer (20 epochs)"
CUDA_VISIBLE_DEVICES=0 python -m src.pro.train_quantizer \
    --config $CONFIG --log_every 100 \
    2>&1 | tee $LOG_DIR/train_quantizer.log

echo "[full] step 3: train retriever (30 epochs)"
CUDA_VISIBLE_DEVICES=0 python -m src.pro.train_retriever \
    --config $CONFIG --log_every 100 \
    2>&1 | tee $LOG_DIR/train_retriever.log

echo "[full] step 4: evaluate on OVEN + INFOSEEK test (SHARED cand pool)"
CUDA_VISIBLE_DEVICES=0 python evaluate_oven_infoseek.py \
    --config $CONFIG --log_every 500 \
    2>&1 | tee $LOG_DIR/eval.log

echo "[full] all done."
