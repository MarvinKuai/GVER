#!/bin/bash
# M2M-SID: NPC (shallow pull) + HardNeg Contrastive (deep push, INFOSEEK only).
#   quantizer picks up `hard_neg_contrastive_*` keys automatically.
# Usage:
#   GPU=0 CONFIG=configs/oven.yaml     TAG=OVEN     bash scripts/run.sh
#   GPU=1 CONFIG=configs/infoseek.yaml TAG=INFOSEEK bash scripts/run.sh
set -eu
# Repo root is the parent of this scripts/ directory.
M2M_SID_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$M2M_SID_ROOT"

PY="${PY:-xxx}"                        # path to your python interpreter
GPU="${GPU:?GPU required}"
CONFIG="${CONFIG:?CONFIG required}"
TAG="${TAG:?TAG required}"
LOG_DIR="logs/${TAG}"
mkdir -p "$LOG_DIR" "outputs/${TAG}"

export PYTHONPATH="$M2M_SID_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "[m2m_sid:${TAG}] stage=train_quantizer gpu=${GPU} config=${CONFIG}"
"$PY" -u -m src.m2m_sid.train_quantizer --config "$CONFIG" --log_every 100 \
    > "$LOG_DIR/train_quantizer.log" 2>&1
echo "[m2m_sid:${TAG}] stage=train_retriever"
"$PY" -u -m src.m2m_sid.train_retriever --config "$CONFIG" --log_every 100 \
    > "$LOG_DIR/train_retriever.log" 2>&1
echo "[m2m_sid:${TAG}] stage=evaluate_fast"
"$PY" -u -m src.m2m_sid.evaluate_m2m_fast --config "$CONFIG" --log_every 2000 \
    > "$LOG_DIR/eval.log" 2>&1
echo "[m2m_sid:${TAG}] done. results:"
"$PY" -c "
import json
d = json.load(open('outputs/${TAG}/results.json'))
for pool in ('ALL','OVEN','INFOSEEK'):
    m = d['metrics'].get(pool, {})
    if m.get('n_queries',0)>0:
        print(f'  {pool}: R@1={m[\"R@1\"]:.4f}  R@5={m[\"R@5\"]:.4f}  R@10={m[\"R@10\"]:.4f}  R@50={m[\"R@50\"]:.4f}  nDCG@10={m[\"nDCG@10\"]:.4f}')
"
