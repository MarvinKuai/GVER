#!/bin/bash
# Extract CLIP-SF features for OVEN/INFOSEEK train + val queries
# Single-GPU sequential: parallelism hurts due to 174GB tar random-read contention.
# Sequential run yields ~260 query/s (per GENIUS test-set baseline);
# total 330k queries -> ~21 minutes.
set -e
PRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PRO_ROOT"

LOG_DIR="$PRO_ROOT/logs/extract"
mkdir -p $LOG_DIR

OUT_ROOT="$PRO_ROOT/outputs/clipsf_feats"

echo "[extract] sequential extraction of 4 (dataset, split) combinations on GPU 0"
echo "[extract] pairs: OVEN/train, OVEN/val, INFOSEEK/train, INFOSEEK/val"

CUDA_VISIBLE_DEVICES=0 python extract_clipsf_feats.py \
    --stage query \
    --datasets OVEN OVEN INFOSEEK INFOSEEK \
    --splits   train val  train    val \
    --out_root $OUT_ROOT --batch_size 128 --num_workers 6 \
    2>&1 | tee $LOG_DIR/all.log

echo "[extract] all done."
