# -*- coding: utf-8 -*-
"""Evaluate pure GENIUS (no CLIP-SF rerank) on M-BEIR task 8 (OVEN + Infoseek).

Difference vs eval_genius_r.py
------------------------------
GENIUS-R: T5 beam search -> expand SID buckets -> **cosine rerank** by
          CLIP-SF features -> top-K.
GENIUS  : T5 beam search -> expand SID buckets -> keep **beam order**
          (first-seen row wins ties inside a bucket) -> top-K.

This script reuses the ``query_codes_topN.pt`` and ``code_to_rows.pkl``
already produced by ``run_genius_r.py``, so no re-running of T5 beam
search is required.

Consumes:
  <clipsf_root>/cand_pool/SHARED/cand_ids.json               # id order
  <clipsf_root>/queries/<DS>/<SPLIT>/{qids.json, qid_to_positives.json}
  <geniusr_root>/cand_pool/SHARED/code_to_rows.pkl
  <geniusr_root>/queries/<DS>/<SPLIT>/query_codes_topN.pt

Writes:
  <out_root>/eval_metrics.json     (default: outputs/genius/eval_metrics.json)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from typing import Dict, List, Sequence, Tuple

import torch


HERE = os.path.dirname(os.path.abspath(__file__))
KS = (1, 5, 10, 50)
NDCG_K = 10


def _load_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_positives(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def evaluate_pure_genius(
    qids: Sequence[str],
    qid_to_pos: Dict[str, List[str]],
    cand_ids: Sequence[str],
    query_codes_topN: torch.Tensor,   # int32 [Nq, num_beams, L]
    code_to_rows: Dict[Tuple[int, ...], List[int]],
    topk_final: int = 50,
    max_cand_per_q: int = 4096,
    tag: str = "",
) -> Dict[str, float]:
    """Expand beam-search SIDs into candidate rows in beam order (no rerank),
    then compute retrieval metrics."""
    Nq = len(qids)
    N_gallery = len(cand_ids)

    cid_to_row = {c: i for i, c in enumerate(cand_ids)}

    hits = {k: 0.0 for k in KS}
    ndcg_sum = 0.0
    n_valid = 0
    n_zero_cands = 0
    total_cands = 0

    t0 = time.time()
    for i in range(Nq):
        qid = qids[i]
        positives = qid_to_pos.get(qid, [])
        if not positives:
            continue
        pos_rows = [cid_to_row[c] for c in positives if c in cid_to_row]
        if not pos_rows:
            continue

        # Expand beams -> candidate rows, preserving first-seen order.
        # NOTE: unlike GENIUS-R, we do NOT rerank; the beam order (== T5 log-prob
        # order) IS the final ranking. Rows inside the same SID bucket keep
        # their pool-order (i.e. insertion order in code_to_rows).
        cand_rows: List[int] = []
        seen = set()
        beams = query_codes_topN[i].tolist()
        for beam in beams:
            key = tuple(beam)
            if -1 in key:
                continue
            rows = code_to_rows.get(key)
            if rows is None:
                continue
            for r in rows:
                if r not in seen:
                    seen.add(r)
                    cand_rows.append(r)
                    if len(cand_rows) >= max_cand_per_q:
                        break
            if len(cand_rows) >= max_cand_per_q:
                break

        if not cand_rows:
            n_zero_cands += 1
            n_valid += 1
            continue
        total_cands += len(cand_rows)

        # No rerank: keep beam order and truncate to top-K.
        ranked_rows = cand_rows[:topk_final]

        pos_set = set(pos_rows)
        n_valid += 1
        for k in KS:
            if any(r in pos_set for r in ranked_rows[:k]):
                hits[k] += 1.0
        dcg = 0.0
        for pos, r in enumerate(ranked_rows[:NDCG_K]):
            if r in pos_set:
                dcg += 1.0 / math.log2(pos + 2)
        ideal_hits = min(len(pos_set), NDCG_K)
        idcg = sum(1.0 / math.log2(p + 2) for p in range(ideal_hits))
        ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0

        if (i + 1) % 2000 == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt
            print(f"  [eval:{tag}] {i+1}/{Nq} ({rate:.1f}/s, {dt:.1f}s)",
                  flush=True)

    metrics = {f"R@{k}": (hits[k] / n_valid) if n_valid else 0.0 for k in KS}
    metrics[f"nDCG@{NDCG_K}"] = (ndcg_sum / n_valid) if n_valid else 0.0
    metrics["n_query"] = Nq
    metrics["n_query_valid"] = n_valid
    metrics["n_gallery"] = int(N_gallery)
    metrics["n_zero_cand_queries"] = n_zero_cands
    metrics["avg_shortlist_size"] = (
        total_cands / max(n_valid - n_zero_cands, 1)
    )
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clipsf_root", type=str,
                   default=os.path.join(HERE, "outputs", "clipsf"),
                   help="Root that holds cand_ids.json / qids.json / positives")
    p.add_argument("--geniusr_root", type=str,
                   default=os.path.join(HERE, "outputs", "genius_r"),
                   help="Root that holds code_to_rows.pkl and "
                        "query_codes_topN.pt (produced by run_genius_r.py)")
    p.add_argument("--out_root", type=str,
                   default=os.path.join(HERE, "outputs", "genius"),
                   help="Where to write pure-GENIUS metrics.")
    p.add_argument("--pool", type=str, default="SHARED")
    p.add_argument("--datasets", nargs="+", default=["OVEN", "INFOSEEK"])
    p.add_argument("--splits", nargs="+", default=["test", "test"])
    p.add_argument("--topk_final", type=int, default=50)
    p.add_argument("--max_cand_per_q", type=int, default=4096)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    # Cand ids + code index (no cand_feats needed since we skip rerank).
    cand_root = os.path.join(args.clipsf_root, "cand_pool", args.pool)
    cand_ids = _load_ids(os.path.join(cand_root, "cand_ids.json"))

    gr_cand_root = os.path.join(args.geniusr_root, "cand_pool", args.pool)
    with open(os.path.join(gr_cand_root, "code_to_rows.pkl"), "rb") as f:
        code_to_rows = pickle.load(f)
    print(f"[cand] pool={args.pool} n_gallery={len(cand_ids)} "
          f"unique_codes={len(code_to_rows)}")

    all_metrics: Dict[str, Dict] = {}
    for ds, sp in zip(args.datasets, args.splits):
        q_dir = os.path.join(args.clipsf_root, "queries", ds, sp)
        qids = _load_ids(os.path.join(q_dir, "qids.json"))
        qid_to_pos = _load_positives(os.path.join(q_dir, "qid_to_positives.json"))

        gr_q_dir = os.path.join(args.geniusr_root, "queries", ds, sp)
        topN = torch.load(os.path.join(gr_q_dir, "query_codes_topN.pt"),
                          map_location="cpu", weights_only=True)
        assert topN.shape[0] == len(qids), \
            f"{ds}/{sp} topN vs qids mismatch: {topN.shape[0]} vs {len(qids)}"

        print(f"[eval {ds}/{sp}] n_query={len(qids)} "
              f"num_beams={topN.shape[1]} code_len={topN.shape[2]}")

        m = evaluate_pure_genius(
            qids=qids, qid_to_pos=qid_to_pos, cand_ids=cand_ids,
            query_codes_topN=topN, code_to_rows=code_to_rows,
            topk_final=args.topk_final,
            max_cand_per_q=args.max_cand_per_q,
            tag=f"{ds}/{sp}",
        )
        m["pool"] = args.pool
        key = f"{ds}/{sp}@{args.pool}"
        all_metrics[key] = m
        print(f"  -> {key}: R@1={m['R@1']:.4f} R@5={m['R@5']:.4f} "
              f"R@10={m['R@10']:.4f} R@50={m['R@50']:.4f} "
              f"nDCG@{NDCG_K}={m[f'nDCG@{NDCG_K}']:.4f} "
              f"(shortlist_avg={m['avg_shortlist_size']:.0f}, "
              f"zero_cand={m['n_zero_cand_queries']})")

    out_json = os.path.join(args.out_root, "eval_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[done] metrics -> {out_json}")


if __name__ == "__main__":
    main()
