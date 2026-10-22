#!/usr/bin/env bash
# One-click driver for Qwen3-VL-Embedding-8B baseline on M-BEIR task 8.
#
# 8-GPU data-parallel encoding pipeline:
#   Stage 1: (8 workers) encode SHARED cand pool  (~816K items, ~2-3h wall)
#   Stage 2: merge 8 shards -> cand_feats.pt
#   Stage 3: (single card) encode OVEN test + Infoseek test queries (~15min)
#   Stage 4: evaluate against SHARED gallery (Recall@k, nDCG@10)
#
# Usage:
#   nohup bash run_qwen3vl_baseline.sh > logs/qwen3vl.log 2>&1 &
#   tail -f logs/qwen3vl.log

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p logs outputs

# --- Config -----------------------------------------------------------------
WORLD="${WORLD:-8}"                          # number of GPUs to use for cand
BATCH="${BATCH:-4}"                          # per-GPU batch size (8B is big)
DTYPE="${DTYPE:-bf16}"
ATTN="${ATTN:-flash_attention_2}"
MAX_LEN="${MAX_LEN:-1024}"
MAX_PIXELS="${MAX_PIXELS:-$((1024 * 512))}"  # ~0.5MP per image

# Prefer /dev/shm (tmpfs, GB/s) copy of the 8B checkpoint to avoid ceph-fuse
# cold-read bottleneck (~60 MB/s -> ~5 min per rank; tmpfs makes it <10s).
MODEL_PATH="${MODEL_PATH:-/dev/shm/qwen3vl-emb-8b}"
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[WARN] MODEL_PATH=$MODEL_PATH not ready; falling back to ceph copy."
    MODEL_PATH="xxx/Qwen3-VL-Embedding-8B"
fi
echo "    MODEL_PATH=$MODEL_PATH"

# --- Tar-based random access (reuse CLIP_baseline's index) ------------------
export MBEIR_TAR_PATH="${MBEIR_TAR_PATH:-xxx/mbeir_images.tar}"
export MBEIR_TAR_INDEX="${MBEIR_TAR_INDEX:-xxx/mbeir_images_oven_index.pkl}"

STAGE="${STAGE:-all}"   # all | smoke | cand | merge | query | eval

echo "[$(date)] === Qwen3-VL-Embedding-8B baseline ==="
echo "    WORLD=$WORLD  BATCH=$BATCH  DTYPE=$DTYPE  ATTN=$ATTN"
echo "    MAX_LEN=$MAX_LEN  MAX_PIXELS=$MAX_PIXELS"
echo "    STAGE=$STAGE"

# --- Stage: smoke test (single card, 20 items) -------------------------------
if [[ "$STAGE" == "smoke" ]]; then
    echo "[$(date)] === SMOKE TEST (20 cand + 20 queries on single GPU) ==="
    CUDA_VISIBLE_DEVICES=0 python3 -u extract_qwen3vl_feats.py \
        --model_path "$MODEL_PATH" \
        --stage cand --pools SHARED --rank 0 --world 1 --limit 20 \
        --batch_size "$BATCH" --dtype "$DTYPE" --attn "$ATTN" \
        --max_length "$MAX_LEN" --max_pixels "$MAX_PIXELS" \
        --out_root "$HERE/outputs_smoke" --overwrite
    CUDA_VISIBLE_DEVICES=0 python3 -u extract_qwen3vl_feats.py \
        --model_path "$MODEL_PATH" \
        --stage query --datasets OVEN INFOSEEK --splits test test \
        --rank 0 --world 1 --limit 20 \
        --batch_size "$BATCH" --dtype "$DTYPE" --attn "$ATTN" \
        --max_length "$MAX_LEN" --max_pixels "$MAX_PIXELS" \
        --out_root "$HERE/outputs_smoke" --overwrite
    echo "[$(date)] smoke test done -> $HERE/outputs_smoke"
    exit 0
fi

# --- Stage 1: encode SHARED cand pool, 8-GPU data parallel ------------------
if [[ "$STAGE" == "all" || "$STAGE" == "cand" ]]; then
    echo "[$(date)] === Stage 1: encode SHARED cand pool with $WORLD GPUs ==="
    pids=()
    for R in $(seq 0 $((WORLD - 1))); do
        LOG="logs/cand_r${R}.log"
        CUDA_VISIBLE_DEVICES=$R python3 -u extract_qwen3vl_feats.py \
            --model_path "$MODEL_PATH" \
            --stage cand --pools SHARED --rank $R --world $WORLD \
            --batch_size "$BATCH" --dtype "$DTYPE" --attn "$ATTN" \
            --max_length "$MAX_LEN" --max_pixels "$MAX_PIXELS" \
            > "$LOG" 2>&1 &
        pids+=($!)
        echo "    launched rank=$R pid=${pids[-1]} log=$LOG"
    done
    # Wait for all shards
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "[FATAL] worker pid=$pid failed. Check per-rank logs." >&2
            exit 1
        fi
    done
    echo "[$(date)] all $WORLD cand shards done."
fi

# --- Stage 2: merge shards --------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "cand" || "$STAGE" == "merge" ]]; then
    echo "[$(date)] === Stage 2: merge $WORLD cand shards ==="
    python3 -u extract_qwen3vl_feats.py --stage merge --pools SHARED --world $WORLD
fi

# --- Stage 3: encode queries (single card is enough) ------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "query" ]]; then
    echo "[$(date)] === Stage 3: encode OVEN + Infoseek test queries (rank 0) ==="
    CUDA_VISIBLE_DEVICES=0 python3 -u extract_qwen3vl_feats.py \
        --model_path "$MODEL_PATH" \
        --stage query --datasets OVEN INFOSEEK --splits test test \
        --rank 0 --world 1 \
        --batch_size "$BATCH" --dtype "$DTYPE" --attn "$ATTN" \
        --max_length "$MAX_LEN" --max_pixels "$MAX_PIXELS" \
        2>&1 | tee logs/query.log
fi

# --- Stage 4: evaluate (per-dataset independent-pool protocol) -------------
if [[ "$STAGE" == "all" || "$STAGE" == "eval" ]]; then
    echo "[$(date)] === Stage 4: evaluate against per-dataset gallery (OVEN / INFOSEEK) ==="
    python3 -u eval_perpool.py \
        --split test \
        --device cuda \
        --out_root "$HERE/outputs" \
        2>&1 | tee logs/eval.log
fi

echo "[$(date)] === All done. Metrics at outputs/eval_metrics.json ==="
