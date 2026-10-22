# -*- coding: utf-8 -*-
"""Evaluate GENIUS_R on M-BEIR task 8 (OVEN + Infoseek) against the SHARED
candidate pool.

Consumes:
  <clipsf_root>/cand_pool/SHARED/{cand_feats.pt, cand_ids.json}         # rerank feats (already normalised)
  <clipsf_root>/queries/<DS>/<SPLIT>/{query_feats.pt, qids.json, qid_to_positives.json}
  <geniusr_root>/cand_pool/SHARED/{cand_codes.pt, code_to_rows.pkl}
  <geniusr_root>/queries/<DS>/<SPLIT>/query_codes_topN.pt

Pipeline per query:
  1. For each of the ``num_beams`` code-tuples returned by T5 beam search,
     look up ``code_to_rows`` to expand into candidate document rows (the
     beam order is preserved as the initial "generative" ranking).
  2. Rerank the union of candidates by cosine similarity between the query's
     ``rerank`` feature (= L2Norm(v_img + v_txt) from CLIP-SF) and each
     candidate's rerank feature.
  3. Truncate to top-K per query and compute R@1/5/10/50 + nDCG@10.

Metrics are written to ``<out_root>/eval_metrics.json`` (default
``<geniusr_root>/eval_metrics.json``).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
KS = (1, 5, 10, 50)
NDCG_K = 10
MAX_POS_PER_Q = 128


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def _load_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_positives(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Rerank + metrics
# ----------------------------------------------------------------------------
@torch.no_grad()
def rerank_and_eval(
    query_feats: torch.Tensor,        # fp16 [Nq, D], L2-normalised
    qids: Sequence[str],
    qid_to_pos: Dict[str, List[str]],
    cand_feats: torch.Tensor,         # fp16 [Nc, D], L2-normalised
    cand_ids: Sequence[str],
    query_codes_topN: torch.Tensor,   # int32 [Nq, num_beams, L]
    code_to_rows: Dict[Tuple[int, ...], List[int]],
    device: torch.device,
    topk_final: int = 50,
    max_cand_per_q: int = 4096,
    tag: str = "",
) -> Dict[str, float]:
    """Rerank T5 beam outputs and compute retrieval metrics."""
    Nq = len(qids)
    L = query_codes_topN.shape[-1]
    K = query_codes_topN.shape[1]
    D = cand_feats.shape[-1]
    N_gallery = cand_feats.shape[0]

    # Fast id->row lookup for positives (only rows that live in this pool).
    cid_to_row = {c: i for i, c in enumerate(cand_ids)}

    # Pre-move cand feats to GPU (816K x 768 fp16 = 1.2 GB; fits comfortably).
    cand_feats_gpu = cand_feats.to(device=device, dtype=torch.float16,
                                    non_blocking=True)

    # Metrics accumulators
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
        # Only positives that live in the pool are eligible for recall.
        pos_rows = [cid_to_row[c] for c in positives if c in cid_to_row]
        if not pos_rows:
            continue

        # Step 1: expand beams -> candidate rows (preserve first-seen order)
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

        # Step 2: cosine rerank on the shortlist
        q = query_feats[i].to(device=device, dtype=torch.float16).unsqueeze(0)
        cand_sub = cand_feats_gpu.index_select(
            0, torch.tensor(cand_rows, device=device, dtype=torch.long))
        sims = (q @ cand_sub.t()).squeeze(0).float()   # [len(cand_rows)]
        # Rank shortlist by similarity, keep top-K_final
        order = torch.argsort(sims, descending=True)
        order = order[:topk_final].cpu().tolist()
        ranked_rows = [cand_rows[o] for o in order]

        # Step 3: metric bookkeeping
        pos_set = set(pos_rows)
        n_valid += 1
        for k in KS:
            if any(r in pos_set for r in ranked_rows[:k]):
                hits[k] += 1.0
        # nDCG@10 (binary relevance)
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
            print(f"  [rerank:{tag}] {i+1}/{Nq} ({rate:.1f}/s, {dt:.1f}s)",
                  flush=True)

    metrics = {
        f"R@{k}": (hits[k] / n_valid) if n_valid else 0.0 for k in KS
    }
    metrics[f"nDCG@{NDCG_K}"] = (ndcg_sum / n_valid) if n_valid else 0.0
    metrics["n_query"] = Nq
    metrics["n_query_valid"] = n_valid
    metrics["n_gallery"] = int(N_gallery)
    metrics["n_zero_cand_queries"] = n_zero_cands
    metrics["avg_shortlist_size"] = (total_cands / max(n_valid - n_zero_cands, 1))
    return metrics


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clipsf_root", type=str,
                   default=os.path.join(HERE, "outputs", "clipsf"),
                   help="Root that holds cand_feats.pt / query_feats.pt")
    p.add_argument("--geniusr_root", type=str,
                   default=os.path.join(HERE, "outputs", "genius_r"),
                   help="Root that holds cand_codes.pt / code_to_rows.pkl / "
                        "query_codes_topN.pt")
    p.add_argument("--pool", type=str, default="SHARED")
    p.add_argument("--datasets", nargs="+", default=["OVEN", "INFOSEEK"])
    p.add_argument("--splits", nargs="+", default=["test", "test"])
    p.add_argument("--out_json", type=str,
                   default=None,
                   help="Where to write final metrics (default: "
                        "<geniusr_root>/eval_metrics.json)")
    p.add_argument("--topk_final", type=int, default=50)
    p.add_argument("--max_cand_per_q", type=int, default=4096)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Cand pool feats + ids + code index
    cand_root = os.path.join(args.clipsf_root, "cand_pool", args.pool)
    cand_feats = torch.load(os.path.join(cand_root, "cand_feats.pt"),
                            map_location="cpu", weights_only=True)
    cand_ids = _load_ids(os.path.join(cand_root, "cand_ids.json"))
    assert cand_feats.shape[0] == len(cand_ids), "cand_feats vs cand_ids mismatch"

    gr_cand_root = os.path.join(args.geniusr_root, "cand_pool", args.pool)
    code_to_rows_path = os.path.join(gr_cand_root, "code_to_rows.pkl")
    with open(code_to_rows_path, "rb") as f:
        code_to_rows = pickle.load(f)
    print(f"[cand] pool={args.pool} n_gallery={len(cand_ids)} "
          f"dim={cand_feats.shape[-1]} unique_codes={len(code_to_rows)}")

    all_metrics: Dict[str, Dict] = {}
    for ds, sp in zip(args.datasets, args.splits):
        q_dir = os.path.join(args.clipsf_root, "queries", ds, sp)
        query_feats = torch.load(os.path.join(q_dir, "query_feats.pt"),
                                 map_location="cpu", weights_only=True)
        qids = _load_ids(os.path.join(q_dir, "qids.json"))
        qid_to_pos = _load_positives(os.path.join(q_dir, "qid_to_positives.json"))

        gr_q_dir = os.path.join(args.geniusr_root, "queries", ds, sp)
        topN = torch.load(os.path.join(gr_q_dir, "query_codes_topN.pt"),
                          map_location="cpu", weights_only=True)
        assert topN.shape[0] == len(qids), \
            f"{ds}/{sp} topN vs qids mismatch: {topN.shape[0]} vs {len(qids)}"

        print(f"[eval {ds}/{sp}] n_query={len(qids)}, "
              f"num_beams={topN.shape[1]}, code_len={topN.shape[2]}")

        m = rerank_and_eval(
            query_feats=query_feats, qids=qids, qid_to_pos=qid_to_pos,
            cand_feats=cand_feats, cand_ids=cand_ids,
            query_codes_topN=topN, code_to_rows=code_to_rows,
            device=device, topk_final=args.topk_final,
            max_cand_per_q=args.max_cand_per_q, tag=f"{ds}/{sp}",
        )
        m["pool"] = args.pool
        key = f"{ds}/{sp}@{args.pool}"
        all_metrics[key] = m
        print(f"  -> {key}: R@1={m['R@1']:.4f} R@5={m['R@5']:.4f} "
              f"R@10={m['R@10']:.4f} R@50={m['R@50']:.4f} "
              f"nDCG@{NDCG_K}={m[f'nDCG@{NDCG_K}']:.4f} "
              f"(shortlist_avg={m['avg_shortlist_size']:.0f}, "
              f"zero_cand={m['n_zero_cand_queries']})")

    out_json = args.out_json or os.path.join(args.geniusr_root, "eval_metrics.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[done] metrics -> {out_json}")


if __name__ == "__main__":
    main()
