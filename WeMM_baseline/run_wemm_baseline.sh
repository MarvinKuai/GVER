#!/usr/bin/env bash
# One-click driver for WeMM-Embedding-9B baseline on M-BEIR task 8
# (OVEN + Infoseek), evaluated on the SHARED cand pool (816,917 items).
#
# Pipeline:
#   Stage 1: 8-GPU data-parallel encoding of the SHARED cand pool
#   Stage 2: merge 8 shards -> cand_feats.pt
#   Stage 3: single-GPU encoding of OVEN & Infoseek test queries
#   Stage 4: cosine-sim retrieval and Recall@k / nDCG@10 metrics
#
# Usage:
#   nohup bash run_wemm_baseline.sh > logs/run.log 2>&1 &
#   tail -f logs/run.log

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p logs outputs

# --- Interpreter -------------------------------------------------------------
# WeMM requires transformers==5.2.0 + qwen_vl_utils; the llama_factory env has
# both, so we invoke its interpreter explicitly.
PY="${PY:-xxx}"                                 # path to your python interpreter (needs transformers==5.2.0)
echo "    PY=$PY"
"$PY" -c "import transformers; assert transformers.__version__.startswith('5.'), \
'WeMM needs transformers>=5.0, got '+transformers.__version__" \
    || { echo "[FATAL] wrong python env: transformers version mismatch"; exit 1; }

# --- Config -----------------------------------------------------------------
WORLD="${WORLD:-4}"
BATCH="${BATCH:-1}"            # 9B model + high-res images -> keep batch=1
DTYPE="${DTYPE:-bf16}"
ATTN="${ATTN:-flash_attention_2}"
MODEL_PATH="${MODEL_PATH:-xxx/WeMM-Embedding-9B}"

# --- Tar-based random access for MBEIR images (fast) -------------------------
export MBEIR_TAR_PATH="${MBEIR_TAR_PATH:-xxx/mbeir_images.tar}"
export MBEIR_TAR_INDEX="${MBEIR_TAR_INDEX:-xxx/mbeir_images_oven_index.pkl}"

STAGE="${STAGE:-all}"          # all | smoke | cand | merge | query | eval

echo "[$(date)] === WeMM-Embedding-9B baseline ==="
echo "    MODEL_PATH=$MODEL_PATH"
echo "    WORLD=$WORLD  BATCH=$BATCH  DTYPE=$DTYPE  ATTN=$ATTN"
echo "    STAGE=$STAGE"

# --- Smoke test --------------------------------------------------------------
if [[ "$STAGE" == "smoke" ]]; then
    echo "[$(date)] === SMOKE TEST (20 cand + 20 queries on single GPU) ==="
    CUDA_VISIBLE_DEVICES=0 "$PY" -u extract_wemm_feats.py \
        --model_path "$MODEL_PATH" \
        --stage cand --pools SHARED --rank 0 --world 1 --limit 20 \
        --batch_size "$BATCH" --dtype "$DTYPE" --attn "$ATTN" \
        --out_root "$HERE/outputs_smoke" --overwrite
    CUDA_VISIBLE_DEVICES=0 "$PY" -u extract_wemm_feats.py \
        --model_path "$MODEL_PATH" \
        --stage query --datasets OVEN INFOSEEK --splits test test \
        --rank 0 --world 1 --limit 20 \
        --batch_size "$BATCH" --dtype "$DTYPE" --attn "$ATTN" \
        --out_root "$HERE/outputs_smoke" --overwrite
    echo "[$(date)] smoke test done -> $HERE/outputs_smoke"
    exit 0
fi

# --- Stage 1: SHARED cand pool (8-GPU data-parallel) ------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "cand" ]]; then
    echo "[$(date)] === Stage 1: encode SHARED cand pool with $WORLD GPUs ==="
    pids=()
    for R in $(seq 0 $((WORLD - 1))); do
        LOG="logs/cand_r${R}.log"
        CUDA_VISIBLE_DEVICES=$R "$PY" -u extract_wemm_feats.py \
            --model_path "$MODEL_PATH" \
            --stage cand --pools SHARED --rank $R --world $WORLD \
            --batch_size "$BATCH" --dtype "$DTYPE" --attn "$ATTN" \
            > "$LOG" 2>&1 &
        pids+=($!)
        echo "    launched rank=$R pid=${pids[-1]} log=$LOG"
    done
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
    "$PY" -u extract_wemm_feats.py --stage merge --pools SHARED --world $WORLD
fi

# --- Stage 3: encode queries (single card) ----------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "query" ]]; then
    echo "[$(date)] === Stage 3: encode OVEN + Infoseek test queries with $WORLD GPUs ==="
    pids=()
    for R in $(seq 0 $((WORLD - 1))); do
        LOG="logs/query_r${R}.log"
        CUDA_VISIBLE_DEVICES=$R "$PY" -u extract_wemm_feats.py \
            --model_path "$MODEL_PATH" \
            --stage query --datasets OVEN INFOSEEK --splits test test \
            --rank $R --world $WORLD \
            --batch_size "$BATCH" --dtype "$DTYPE" --attn "$ATTN" \
            > "$LOG" 2>&1 &
        pids+=($!)
        echo "    launched query rank=$R pid=${pids[-1]} log=$LOG"
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "[FATAL] query worker pid=$pid failed. Check per-rank logs." >&2
            exit 1
        fi
    done
    echo "[$(date)] === Stage 3b: merge query shards ==="
    "$PY" -u extract_wemm_feats.py --stage merge_query \
        --datasets OVEN INFOSEEK --splits test test --world $WORLD \
        2>&1 | tee logs/query_merge.log
fi

# --- Stage 4: evaluate ------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "eval" ]]; then
    echo "[$(date)] === Stage 4: evaluate against SHARED gallery ==="
    "$PY" -u eval_wemm.py \
        --datasets OVEN INFOSEEK --splits test test \
        --pool SHARED --device cuda \
        --out_root "$HERE/outputs" \
        2>&1 | tee logs/eval.log
fi

echo "[$(date)] === All done. Metrics at outputs/eval_metrics.json ==="
