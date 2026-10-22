#!/bin/bash
# Per-dataset independent training + independent-pool evaluation for MGR_PRO.
#
# Usage:
#   DS=OVEN     GPU=0 bash scripts/orchestrate_perpool.sh
#   DS=INFOSEEK GPU=1 bash scripts/orchestrate_perpool.sh
#
# Steps (all in-process, will resume/skip if intermediate artifacts already exist):
#   1) pack independent cand pool + queries into outputs/data_perpool/$DS/
#      (uses --cand_prefix_filter '5:' for OVEN, '6:' for INFOSEEK)
#   2) train quantizer (20 ep) -> outputs/ckpt_perpool/$DS/quantizer.pt
#   3) train retriever (30 ep) -> outputs/ckpt_perpool/$DS/retriever.pt
#   4) evaluate on the independent pool ($DS-only) with paper protocol
#      (--no_tie_break, beam=50, output=100)  -> outputs/eval_perpool/$DS/results.json
#
# Final metrics file for the user:
#   outputs/eval_perpool/$DS/results.json          (paper-protocol, R@1/5/10/50/nDCG@10)
#
set -eu

# Repo root is the parent of this scripts/ directory.
PRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PRO_ROOT"

DS="${DS:?must set DS=OVEN or DS=INFOSEEK}"
GPU="${GPU:-0}"
LOG_EVERY="${LOG_EVERY:-500}"

case "$DS" in
    OVEN)     PREFIX='5:' ; CONFIG=configs/pro_oven_only.yaml     ;;
    INFOSEEK) PREFIX='6:' ; CONFIG=configs/pro_infoseek_only.yaml ;;
    *) echo "DS must be OVEN or INFOSEEK, got '$DS'"; exit 1 ;;
esac

LOG_DIR=logs/perpool/$DS
DATA_DIR=outputs/data_perpool/$DS
CKPT_DIR=outputs/ckpt_perpool/$DS
EVAL_DIR=outputs/eval_perpool/$DS
mkdir -p "$LOG_DIR" "$DATA_DIR" "$CKPT_DIR" "$EVAL_DIR"

echo "[perpool:$DS] === step 1/4: pack ($DS-only cand pool, prefix='$PREFIX') ==="
if [ ! -f "$DATA_DIR/cand/cand_feats.pt" ] || [ ! -f "$DATA_DIR/test/qrels.txt" ]; then
    python pack_for_pro.py \
        --clipsf_root outputs/clipsf_feats \
        --out_root "$DATA_DIR" \
        --datasets "$DS" \
        --splits train val test \
        --cand_prefix_filter "$PREFIX" \
        2>&1 | tee "$LOG_DIR/pack.log"
else
    echo "[perpool:$DS] pack: outputs already exist, skip"
fi

echo "[perpool:$DS] === step 2/4: train quantizer (20 ep, GPU $GPU) ==="
if [ ! -f "$CKPT_DIR/quantizer.pt" ]; then
    CUDA_VISIBLE_DEVICES="$GPU" python -m src.pro.train_quantizer \
        --config "$CONFIG" --log_every 100 \
        2>&1 | tee "$LOG_DIR/train_quantizer.log"
else
    echo "[perpool:$DS] quantizer ckpt already exists, skip"
fi

echo "[perpool:$DS] === step 3/4: train retriever (30 ep, GPU $GPU) ==="
if [ ! -f "$CKPT_DIR/retriever.pt" ]; then
    CUDA_VISIBLE_DEVICES="$GPU" python -m src.pro.train_retriever \
        --config "$CONFIG" --log_every 100 \
        2>&1 | tee "$LOG_DIR/train_retriever.log"
else
    echo "[perpool:$DS] retriever ckpt already exists, skip"
fi

echo "[perpool:$DS] === step 4/4: evaluate on $DS-only pool (paper protocol) ==="
CUDA_VISIBLE_DEVICES="$GPU" python evaluate_oven_infoseek.py \
    --config "$CONFIG" \
    --log_every "$LOG_EVERY" \
    --no_tie_break \
    2>&1 | tee "$LOG_DIR/eval.log"

echo "[perpool:$DS] ALL DONE."
echo "[perpool:$DS] metrics file: $EVAL_DIR/results.json"
python -c "import json;d=json.load(open('$EVAL_DIR/results.json'));print(json.dumps(d['metrics'], indent=2))"
