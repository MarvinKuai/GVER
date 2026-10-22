#!/bin/bash
# M2M-SID: self-contained feature preparation.
#
# Rebuilds the CLIP-SF (UniIR ViT-L/14) features that the quantizer / retriever
# need as input, directly inside the M2M_SID project tree.  Reuses the already
# validated extraction code in ../PRO_baseline (extract_clipsf_feats.py,
# pack_for_pro.py, mbeir_data.py, clip_sf_model.py, mbeir_tar_reader.py).
#
# Steps:
#   0) stage CLIP-SF ckpt + CLIP ViT-L/14 backbone to /dev/shm/genius_ckpt/
#   1) extract SHARED cand pool features in parallel on all 4 GPUs
#   2) merge cand shards -> cand_pool/SHARED/{cand_feats,cand_img,cand_txt}.pt
#   3) extract queries for OVEN + INFOSEEK x {train, val, test} (single GPU)
#   4) pack per-dataset data_perpool (OVEN prefix='5:', INFOSEEK prefix='6:')
#
# Idempotent: existing cand shards / query bundles / pack outputs are skipped.
#
# Usage (default):
#   bash scripts/prepare_feats.sh
#
# Env overrides:
#   PY=... WORLD=4 BS=128 NW=6
set -eu

M2M_SID_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$M2M_SID_ROOT/.." && pwd)"
cd "$M2M_SID_ROOT"

PY="${PY:-xxx}"                        # path to your python interpreter
WORLD="${WORLD:-4}"
BS="${BS:-128}"
NW="${NW:-6}"

PRO_DIR="$REPO_ROOT/PRO_baseline"
OUT_FEAT_ROOT="$M2M_SID_ROOT/outputs/features"
CLIPSF_ROOT=$OUT_FEAT_ROOT/clipsf_feats
PERPOOL_ROOT=$OUT_FEAT_ROOT/data_perpool
LOG_DIR="$M2M_SID_ROOT/logs/prepare"
mkdir -p "$CLIPSF_ROOT" "$PERPOOL_ROOT" "$LOG_DIR"

# Checkpoints (fill in the paths on your machine).
CLIP_SF_SRC="${CLIP_SF_SRC:-xxx/clip_sf_large.pth}"
VITL14_SRC="${VITL14_SRC:-xxx/ViT-L-14.pt}"

# PRO_baseline scripts import clip_sf_model + mbeir_data from CWD.
export PYTHONPATH="$PRO_DIR:${PYTHONPATH:-}"

# ----------------------------------------------------------------------------
# Step 0: stage CLIP-SF + CLIP backbone into /dev/shm (extract script's default)
# ----------------------------------------------------------------------------
GENIUS_CKPT=/dev/shm/genius_ckpt
CLIP_DL_ROOT=$GENIUS_CKPT/clip_dl_root
mkdir -p "$GENIUS_CKPT" "$CLIP_DL_ROOT"
if [ ! -f "$GENIUS_CKPT/clip_sf_large.pth" ]; then
    echo "[prepare] staging clip_sf_large.pth -> $GENIUS_CKPT (4.8 GB, ~30s)"
    cp "$CLIP_SF_SRC" "$GENIUS_CKPT/clip_sf_large.pth"
fi
if [ ! -f "$GENIUS_CKPT/ViT-L-14.pt" ]; then
    echo "[prepare] staging ViT-L-14.pt -> $GENIUS_CKPT (0.9 GB)"
    cp "$VITL14_SRC" "$GENIUS_CKPT/ViT-L-14.pt"
fi
if [ ! -f "$CLIP_DL_ROOT/ViT-L-14.pt" ]; then
    cp "$GENIUS_CKPT/ViT-L-14.pt" "$CLIP_DL_ROOT/ViT-L-14.pt"
fi

cd "$PRO_DIR"   # extract_clipsf_feats.py relies on CWD imports (clip_sf_model, mbeir_data)

# ----------------------------------------------------------------------------
# Step 1: cand pool extraction (SHARED, striped over $WORLD GPUs in parallel)
# ----------------------------------------------------------------------------
CAND_DIR=$CLIPSF_ROOT/cand_pool/SHARED
if [ -f "$CAND_DIR/cand_feats.pt" ] && [ -f "$CAND_DIR/cand_img.pt" ] && [ -f "$CAND_DIR/cand_txt.pt" ]; then
    echo "[prepare] cand pool already merged, skip step 1 & 2"
else
    echo "[prepare] === step 1: extract SHARED cand pool on $WORLD GPUs ==="
    pids=()
    for r in $(seq 0 $((WORLD-1))); do
        shard="$CAND_DIR/shards/shard_$(printf %02d $r)_of_$(printf %02d $WORLD).pt"
        if [ -f "$shard" ]; then
            echo "  [cand:r$r] shard exists, skip"
            continue
        fi
        CUDA_VISIBLE_DEVICES=$r "$PY" -u extract_clipsf_feats.py \
            --stage cand --pool SHARED --rank $r --world $WORLD \
            --out_root "$CLIPSF_ROOT" \
            --batch_size "$BS" --num_workers "$NW" \
            --save_raw \
            > "$LOG_DIR/cand_r${r}.log" 2>&1 &
        pids+=($!)
        echo "  [cand:r$r] launched pid=$!"
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || { echo "[prepare] cand shard pid=$pid FAILED"; exit 1; }
    done
    echo "[prepare] === step 2: merge cand shards ==="
    "$PY" -u extract_clipsf_feats.py \
        --stage cand --pool SHARED --merge --world "$WORLD" \
        --out_root "$CLIPSF_ROOT" \
        > "$LOG_DIR/cand_merge.log" 2>&1
fi

# ----------------------------------------------------------------------------
# Step 3: query extraction (single GPU is fine; splits are small)
# ----------------------------------------------------------------------------
echo "[prepare] === step 3: extract queries (OVEN + INFOSEEK x train/val/test) ==="
Q_ROOT=$CLIPSF_ROOT/queries
NEED_Q=0
for ds in OVEN INFOSEEK; do
    for sp in train val test; do
        if [ ! -f "$Q_ROOT/$ds/$sp/query_feats.pt" ]; then NEED_Q=1; fi
    done
done
if [ "$NEED_Q" = "1" ]; then
    CUDA_VISIBLE_DEVICES=0 "$PY" -u extract_clipsf_feats.py \
        --stage query \
        --datasets OVEN OVEN OVEN INFOSEEK INFOSEEK INFOSEEK \
        --splits   train val test train val test \
        --out_root "$CLIPSF_ROOT" \
        --batch_size "$BS" --num_workers "$NW" \
        --save_raw \
        > "$LOG_DIR/query.log" 2>&1
else
    echo "[prepare] all query bundles exist, skip step 3"
fi

# ----------------------------------------------------------------------------
# Step 4: pack per-dataset data_perpool
# ----------------------------------------------------------------------------
for ds in OVEN INFOSEEK; do
    case "$ds" in
        OVEN)     PREFIX='5:' ;;
        INFOSEEK) PREFIX='6:' ;;
    esac
    OUT_DS=$PERPOOL_ROOT/$ds
    if [ -f "$OUT_DS/cand/cand_feats.pt" ] && [ -f "$OUT_DS/test/qrels.txt" ]; then
        echo "[prepare] data_perpool/$ds already packed, skip"
        continue
    fi
    echo "[prepare] === step 4: pack data_perpool/$ds (prefix='$PREFIX') ==="
    "$PY" -u pack_for_pro.py \
        --clipsf_root "$CLIPSF_ROOT" \
        --out_root "$OUT_DS" \
        --datasets "$ds" \
        --splits train val test \
        --cand_prefix_filter "$PREFIX" \
        > "$LOG_DIR/pack_${ds}.log" 2>&1
done

echo "[prepare] ALL DONE. features at: $OUT_FEAT_ROOT"
du -sh "$CLIPSF_ROOT" "$PERPOOL_ROOT"/*
