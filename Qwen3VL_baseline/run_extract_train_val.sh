#!/usr/bin/env bash
# ------------------------------------------------------------------
# Extract Qwen3-VL-Embedding features for the *train* and *val* splits
# of M-BEIR task-8 (OVEN + Infoseek), sharded across 4 local GPUs.
#
# Outputs (colocated with the existing test-split features):
#   outputs/queries/<DATASET>/<SPLIT>/query_feats.pt
#   outputs/queries/<DATASET>/<SPLIT>/qids.json
#   outputs/queries/<DATASET>/<SPLIT>/qid_to_positives.json
#   outputs/queries/<DATASET>/<SPLIT>/meta.json
#
# Scale reminder:
#   OVEN     train 154,460  |  val 14,749
#   INFOSEEK train 143,131  |  val 17,688
# On 4x H20 at ~1.4 q/s/GPU, train ~= 15h wall, val ~= 1.6h wall.
# ------------------------------------------------------------------
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

# ---- Python env: must have transformers>=4.55 (qwen3_vl model class) ----
PY_BIN="${PY_BIN:-xxx}"                   # path to your python interpreter (needs transformers>=4.55)
if [[ ! -x "$PY_BIN" ]]; then
    echo "[FATAL] python not found at $PY_BIN" >&2
    exit 1
fi
echo "[env] python = $PY_BIN"
"$PY_BIN" -c "import transformers, torch; print(f'[env] transformers={transformers.__version__} torch={torch.__version__}')"

# ---- config -----------------------------------------------------
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export TOKENIZERS_PARALLELISM=false
export MBEIR_TAR_PATH="${MBEIR_TAR_PATH:-xxx/mbeir_images.tar}"
export MBEIR_TAR_INDEX="${MBEIR_TAR_INDEX:-xxx/mbeir_images_oven_index.pkl}"

# Prefer tmpfs copy of the 8B checkpoint; fall back to ceph.
MODEL_SRC="xxx/Qwen3-VL-Embedding-8B"
MODEL_TMPFS="/dev/shm/qwen3vl-emb-8b"
if [[ -f "$MODEL_TMPFS/config.json" ]]; then
    MODEL_PATH="$MODEL_TMPFS"
    echo "[model] using tmpfs copy: $MODEL_PATH"
else
    if [[ "${STAGE_MODEL_TMPFS:-1}" == "1" ]]; then
        echo "[model] staging weights to tmpfs (one-time, ~17GB)..."
        mkdir -p "$MODEL_TMPFS"
        cp -a "$MODEL_SRC/." "$MODEL_TMPFS/"
        MODEL_PATH="$MODEL_TMPFS"
        echo "[model] tmpfs staged: $MODEL_PATH"
    else
        MODEL_PATH="$MODEL_SRC"
        echo "[model] using ceph path (slower cold-load): $MODEL_PATH"
    fi
fi

WORLD=${WORLD:-4}                       # 4 GPUs on this box
# Visible CUDA ids per rank.  Space-separated string in $DEVICES_STR (env-overridable),
# then split into an array.  Written this way to stay safe under `set -u`.
DEVICES_STR="${DEVICES_STR:-0 1 2 3}"
read -r -a DEVICES <<< "$DEVICES_STR"
if [[ ${#DEVICES[@]} -lt $WORLD ]]; then
    echo "[FATAL] DEVICES=(${DEVICES[*]}) has fewer entries than WORLD=$WORLD" >&2
    exit 1
fi
BATCH=${BATCH:-4}
DTYPE=${DTYPE:-bf16}
ATTN=${ATTN:-flash_attention_2}
MAX_LEN=${MAX_LEN:-1024}
MAX_PIXELS=${MAX_PIXELS:-$((1024 * 512))}
LOG_DIR=${LOG_DIR:-"$HERE/logs/train_val_feats"}
mkdir -p "$LOG_DIR"

# splits to extract: (dataset,split) pairs.  val is much smaller so we run it
# first as an end-to-end smoke test before committing 15h to train.
declare -a JOBS=(
    "OVEN val"
    "INFOSEEK val"
    "OVEN train"
    "INFOSEEK train"
)

extract_one() {
    local ds="$1"
    local sp="$2"
    local out_dir="$HERE/outputs/queries/${ds}/${sp}"
    if [[ -f "$out_dir/query_feats.pt" && "${OVERWRITE:-0}" != "1" ]]; then
        echo "[$(date +%H:%M:%S)] [skip] $out_dir/query_feats.pt already exists."
        return
    fi
    echo ""
    echo "============================================================"
    echo "[$(date +%H:%M:%S)] Extract  ds=${ds}  split=${sp}  WORLD=${WORLD}"
    echo "============================================================"

    # Stage 1: per-rank shards
    local pids=()
    for R in $(seq 0 $((WORLD-1))); do
        local DEV=${DEVICES[$R]}
        local LOG_R="$LOG_DIR/${ds}_${sp}_rank${R}.log"
        (
            CUDA_VISIBLE_DEVICES=$DEV \
            "$PY_BIN" -u extract_qwen3vl_feats.py \
                --model_path "$MODEL_PATH" \
                --stage query \
                --datasets "$ds" \
                --splits "$sp" \
                --rank "$R" \
                --world "$WORLD" \
                --batch_size "$BATCH" \
                --dtype "$DTYPE" \
                --attn "$ATTN" \
                --max_length "$MAX_LEN" \
                --max_pixels "$MAX_PIXELS" \
                --overwrite \
                > "$LOG_R" 2>&1
        ) &
        pids+=($!)
        echo "  launched rank=$R  device=$DEV  pid=${pids[-1]}  log=$LOG_R"
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "[FATAL] worker pid=$pid failed. Check per-rank logs." >&2
            exit 1
        fi
    done
    echo "[$(date +%H:%M:%S)] shards finished for ${ds}/${sp}"

    # Stage 2: merge -> query_feats.pt
    "$PY_BIN" -u extract_qwen3vl_feats.py \
        --stage merge_query \
        --datasets "$ds" \
        --splits "$sp" \
        --world "$WORLD" \
        > "$LOG_DIR/${ds}_${sp}_merge.log" 2>&1
    echo "[$(date +%H:%M:%S)] merged ${ds}/${sp} -> $out_dir/query_feats.pt"
}

for job in "${JOBS[@]}"; do
    read -r ds sp <<< "$job"
    extract_one "$ds" "$sp"
done

echo ""
echo "[$(date +%H:%M:%S)] ALL DONE. Summary:"
for job in "${JOBS[@]}"; do
    read -r ds sp <<< "$job"
    d="$HERE/outputs/queries/${ds}/${sp}"
    if [[ -f "$d/query_feats.pt" ]]; then
        du -h "$d/query_feats.pt"
    else
        echo "  [MISSING] $d/query_feats.pt"
    fi
done
