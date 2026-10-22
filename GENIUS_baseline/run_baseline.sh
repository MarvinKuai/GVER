#!/usr/bin/env bash
# One-click driver for GENIUS baseline on M-BEIR task 8 (OVEN + Infoseek, test).
#
# This script reproduces the GENIUS_R generative-retrieval track (T5 + RQ
# codes).  Stages 1-2 encode CLIP-SF features for the SHARED cand pool and
# the test queries; these features are consumed by GENIUS_R's Combiner + RQ
# quantizer in Stage 4.
#
# Stage layout:
#   Stage 1a) 8-way SHARED cand pool encoding with CLIP_SF   ~ 70 min
#   Stage 1b) merge shards -> outputs/clipsf/cand_pool/SHARED/{feats,img,txt}
#   Stage 2 ) queries for OVEN/test + Infoseek/test          ~ 30 min
#   Stage 3 ) (optional) standalone CLIP_SF retrieval eval   -- omitted
#   Stage 4 ) generate GENIUS codes for SHARED pool + queries + rerank
#   Stage 5 ) evaluate GENIUS_R track
#
# Usage:
#   nohup bash run_baseline.sh > logs/baseline.log 2>&1 &
#   tail -f logs/baseline.log
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

mkdir -p logs outputs/clipsf outputs/genius_r

# --- Data & model paths ----------------------------------------------------
export MBEIR_TAR_PATH="${MBEIR_TAR_PATH:-xxx/mbeir_images.tar}"
export MBEIR_TAR_INDEX="${MBEIR_TAR_INDEX:-xxx/mbeir_images_oven_index.pkl}"
export CLIPSF_CKPT="${CLIPSF_CKPT:-/dev/shm/genius_ckpt/clip_sf_large.pth}"
export CLIP_DOWNLOAD_ROOT="${CLIP_DOWNLOAD_ROOT:-/dev/shm/genius_ckpt/clip_dl_root}"
export VITL14_SRC="${VITL14_SRC:-/dev/shm/genius_ckpt/ViT-L-14.pt}"

WORLD="${WORLD:-8}"
BATCH="${BATCH:-128}"
# Per-rank DataLoader worker count.  With WORLD=8 we already have 8 main
# processes each running a torch inference loop; each worker adds another
# Python process doing PIL preprocessing.  Keep this tight (WORLD * NUM_WORKERS
# should stay well below physical core count) to avoid load-average blowup.
NUM_WORKERS="${NUM_WORKERS:-2}"
QUERY_NUM_WORKERS="${QUERY_NUM_WORKERS:-6}"

CLIPSF_OUT="outputs/clipsf"
CLIPSF_RAW_OUT="outputs/clipsf_raw"
GENIUS_OUT="outputs/genius_r"

log() { echo "[$(date +'%F %T')] $*"; }

# =========================================================================
# Stage 1a: 8-way SHARED cand pool encoding (CLIP-SF)
# =========================================================================
if [[ -f "$CLIPSF_OUT/cand_pool/SHARED/cand_feats.pt" ]]; then
    log "=== Stage 1a: SHARED cand feats already exist, skipping ==="
else
    log "=== Stage 1a: encode SHARED cand pool with ${WORLD}-way DP (CLIP_SF) ==="
    pids=()
    for r in $(seq 0 $((WORLD - 1))); do
        CUDA_VISIBLE_DEVICES=$r python -u extract_clipsf_feats.py \
            --stage cand --pool SHARED \
            --rank "$r" --world "$WORLD" \
            --batch_size "$BATCH" --num_workers "$NUM_WORKERS" \
            --out_root "$CLIPSF_OUT" \
            > "logs/cand_r${r}.log" 2>&1 &
        pids+=($!)
        log "  launched rank=$r on GPU=$r (pid=$!)"
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || { log "!! rank pid=$pid failed"; exit 1; }
    done
    log "=== Stage 1a: all shards done ==="

    log "=== Stage 1b: merge shards -> flat cand_{feats,img,txt}.pt ==="
    python -u extract_clipsf_feats.py \
        --stage cand --pool SHARED --merge --world "$WORLD" \
        --out_root "$CLIPSF_OUT"
fi

# =========================================================================
# Stage 2: encode OVEN/test and Infoseek/test queries (single GPU is enough)
# =========================================================================
if [[ -f "$CLIPSF_OUT/queries/OVEN/test/query_feats.pt"     \
   && -f "$CLIPSF_OUT/queries/INFOSEEK/test/query_feats.pt" ]]; then
    log "=== Stage 2: query features already exist, skipping ==="
else
    log "=== Stage 2: encode OVEN + Infoseek test queries (CLIP_SF) ==="
    CUDA_VISIBLE_DEVICES=0 python -u extract_clipsf_feats.py \
        --stage query --datasets OVEN INFOSEEK --splits test test \
        --batch_size "$BATCH" --num_workers "$QUERY_NUM_WORKERS" \
        --out_root "$CLIPSF_OUT"
fi

# =========================================================================
# Stage 3: (optional) evaluate CLIP_SF retrieval track
#   The CLIP-SF features extracted in Stage 1 are also reused as inputs to
#   GENIUS_R below.  Standalone CLIP-SF retrieval evaluation is omitted
#   from this repo; see the UniIR / CLIP-SF reference implementation if
#   you want to reproduce that track.
# =========================================================================

# =========================================================================
# Stage 4a: 8-way SHARED cand pool encoding (CLIP-SF *raw* per-branch feats)
#           needed as input to GENIUS's Combiner + RQ quantizer.
# =========================================================================
STAGES="${STAGES:-clipsf genius_r}"
run_genius_r=1
if [[ ! " $STAGES " =~ " genius_r " ]]; then
    run_genius_r=0
fi

if [[ "$run_genius_r" == "1" ]]; then
    if [[ -f "$CLIPSF_RAW_OUT/cand_pool/SHARED/cand_img_raw.pt" \
       && -f "$CLIPSF_RAW_OUT/cand_pool/SHARED/cand_txt_raw.pt" ]]; then
        log "=== Stage 4a: raw cand feats already exist, skipping ==="
    else
        log "=== Stage 4a: encode SHARED cand pool RAW feats with ${WORLD}-way DP ==="
        pids=()
        for r in $(seq 0 $((WORLD - 1))); do
            CUDA_VISIBLE_DEVICES=$r python -u extract_clipsf_feats.py \
                --stage cand --pool SHARED \
                --rank "$r" --world "$WORLD" \
                --batch_size "$BATCH" --num_workers "$NUM_WORKERS" \
                --out_root "$CLIPSF_RAW_OUT" \
                --save_raw \
                > "logs/cand_raw_r${r}.log" 2>&1 &
            pids+=($!)
            log "  launched raw rank=$r on GPU=$r (pid=$!)"
        done
        for pid in "${pids[@]}"; do
            wait "$pid" || { log "!! raw rank pid=$pid failed"; exit 1; }
        done
        log "=== Stage 4a: all raw shards done ==="

        log "=== Stage 4b: merge raw shards -> cand_{img,txt}_raw.pt ==="
        python -u extract_clipsf_feats.py \
            --stage cand --pool SHARED --merge --world "$WORLD" \
            --out_root "$CLIPSF_RAW_OUT"
    fi

    # =====================================================================
    # Stage 4c: raw query feats (single GPU)
    # =====================================================================
    if [[ -f "$CLIPSF_RAW_OUT/queries/OVEN/test/query_img_raw.pt" \
       && -f "$CLIPSF_RAW_OUT/queries/INFOSEEK/test/query_img_raw.pt" ]]; then
        log "=== Stage 4c: raw query feats already exist, skipping ==="
    else
        log "=== Stage 4c: encode OVEN + Infoseek test queries RAW feats ==="
        CUDA_VISIBLE_DEVICES=0 python -u extract_clipsf_feats.py \
            --stage query --datasets OVEN INFOSEEK --splits test test \
            --batch_size "$BATCH" --num_workers "$QUERY_NUM_WORKERS" \
            --out_root "$CLIPSF_RAW_OUT" \
            --save_raw
    fi

    # =====================================================================
    # Stage 4d: run GENIUS_R pipeline (quantize + trie + beam search)
    # =====================================================================
    log "=== Stage 4d: GENIUS_R pipeline (quantize cand + beam search queries) ==="
    CUDA_VISIBLE_DEVICES=0 python -u run_genius_r.py \
        --clipsf_root "$CLIPSF_OUT" \
        --clipsf_raw_root "$CLIPSF_RAW_OUT" \
        --out_root "$GENIUS_OUT" \
        --pool SHARED \
        --datasets OVEN INFOSEEK --splits test test \
    --num_beams "${NUM_BEAMS:-50}" \
        --beam_batch "${BEAM_BATCH:-32}" \
        --quant_batch "${QUANT_BATCH:-8192}"

    # =====================================================================
    # Stage 5: evaluate GENIUS_R
    # =====================================================================
    log "=== Stage 5: evaluate GENIUS_R against SHARED gallery ==="
    CUDA_VISIBLE_DEVICES=0 python -u eval_genius_r.py \
        --clipsf_root "$CLIPSF_OUT" \
        --geniusr_root "$GENIUS_OUT" \
        --pool SHARED \
        --datasets OVEN INFOSEEK --splits test test \
        --topk_final "${TOPK_FINAL:-50}" \
        --max_cand_per_q "${MAX_CAND_PER_Q:-4096}"

    log "=== GENIUS_R metrics -> $GENIUS_OUT/eval_metrics.json ==="
fi

log "=== ALL DONE. ==="
log "    CLIP_SF   : $CLIPSF_OUT/eval_metrics.json"
log "    GENIUS_R  : $GENIUS_OUT/eval_metrics.json"
