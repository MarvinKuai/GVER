# -*- coding: utf-8 -*-
"""Evaluate the CLIP-SF baseline on M-BEIR task 8 (OVEN + Infoseek).

Given the fused (image+text) features produced by ``extract_clip_feats.py``:

  outputs/cand_pool/SHARED/{cand_feats.pt, cand_ids.json}
  outputs/queries/<DATASET>/<SPLIT>/{query_feats.pt, qids.json, qid_to_positives.json}

we compute the standard M-BEIR retrieval metrics:
  * Recall@1, Recall@5, Recall@10, Recall@50
  * nDCG@10

Multi-positive handling: qrels list *all* relevant did per qid.  Recall@k is
'hit if any positive is in top-k'.  nDCG uses binary gains.

Similarity is cosine (dot-product of L2-normalised fused vectors).  We chunk
the query batch to bound peak memory when the gallery is large (~800K rows).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import torch


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_ROOT = os.path.join(HERE, "outputs")


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def _dcg_at_k(hits_topk: torch.Tensor, k: int) -> torch.Tensor:
    """DCG@k with binary gains.  ``hits_topk``: [N, k] bool/float."""
    hits = hits_topk[:, :k].float()
    denom = torch.log2(torch.arange(2, k + 2, device=hits.device, dtype=hits.dtype))  # log2(rank+1), rank=1..k
    return (hits / denom).sum(dim=-1)                                                 # [N]


def _ideal_dcg_at_k(num_pos: torch.Tensor, k: int) -> torch.Tensor:
    """IDCG@k when there are ``num_pos`` relevants (all with gain=1).

    IDCG@k = sum_{i=1..min(num_pos, k)} 1 / log2(i+1)
    """
    device = num_pos.device
    denom = 1.0 / torch.log2(torch.arange(2, k + 2, device=device, dtype=torch.float32))  # length k
    cum = torch.cumsum(denom, dim=0)                                                       # length k
    n_clip = torch.clamp(num_pos.long(), max=k)                                            # [N]
    # Avoid index -1: for num_pos==0 return IDCG=0
    idx = (n_clip - 1).clamp(min=0)
    idcg = cum[idx]
    idcg = torch.where(n_clip > 0, idcg, torch.zeros_like(idcg))
    return idcg


# ---------------------------------------------------------------------------
# Per-(dataset, split) evaluator
# ---------------------------------------------------------------------------
def evaluate_one(
    dataset: str,
    split: str,
    pool_name: str,
    out_root: str,
    device: torch.device,
    query_batch: int,
    ks: Tuple[int, ...] = (1, 5, 10, 50),
    ndcg_k: int = 10,
) -> Dict[str, float]:
    q_dir = os.path.join(out_root, "queries", dataset, split)
    c_dir = os.path.join(out_root, "cand_pool", pool_name)

    q_feats = torch.load(os.path.join(q_dir, "query_feats.pt"), map_location="cpu")  # [Nq, D] fp16
    c_feats = torch.load(os.path.join(c_dir, "cand_feats.pt"),  map_location="cpu")  # [Nc, D] fp16
    with open(os.path.join(q_dir, "qids.json"), "r", encoding="utf-8") as f:
        qids: List[str] = [str(x) for x in json.load(f)]
    with open(os.path.join(c_dir, "cand_ids.json"), "r", encoding="utf-8") as f:
        dids: List[str] = [str(x) for x in json.load(f)]
    with open(os.path.join(q_dir, "qid_to_positives.json"), "r", encoding="utf-8") as f:
        qid_to_pos_raw: Dict[str, List[str]] = json.load(f)

    did_to_idx = {d: i for i, d in enumerate(dids)}

    # Build a padded ragged tensor of positive indices per query.
    n_query = len(qids)
    top_k = max(max(ks), ndcg_k)

    pos_lists: List[List[int]] = []
    n_missing_all = 0
    n_missing_some = 0
    num_positives = torch.zeros(n_query, dtype=torch.long)
    for i, qid in enumerate(qids):
        raw = qid_to_pos_raw.get(qid, []) or []
        mapped = [did_to_idx[d] for d in raw if d in did_to_idx]
        if raw and not mapped:
            n_missing_all += 1
        elif len(mapped) < len(raw):
            n_missing_some += 1
        pos_lists.append(mapped)
        num_positives[i] = len(mapped)

    if n_missing_all or n_missing_some:
        print(f"[warn][{dataset}/{split}] positives not in gallery: "
              f"{n_missing_all} queries with 0 mappable pos, "
              f"{n_missing_some} queries with partial loss.")

    max_pos = max((len(x) for x in pos_lists), default=0)
    if max_pos == 0:
        raise RuntimeError(f"[{dataset}/{split}] no positives could be mapped into gallery {pool_name!r}")

    # Ragged -> padded  (-1 as pad sentinel; broadcast comparison ignores pad).
    pos_pad = torch.full((n_query, max_pos), -1, dtype=torch.long)
    for i, lst in enumerate(pos_lists):
        if lst:
            pos_pad[i, : len(lst)] = torch.tensor(lst, dtype=torch.long)

    print(f"[eval] {dataset}/{split} vs pool='{pool_name}': "
          f"n_query={n_query}, n_gallery={len(dids)}, dim={q_feats.shape[-1]}, "
          f"max_positives_per_q={max_pos}")

    # Move gallery to device once (fp32 for stable dot product).
    c_feats_dev = c_feats.to(device=device, dtype=torch.float32)
    n_query_valid = 0

    hits_topk_all: List[torch.Tensor] = []                        # [b, top_k] bool
    for start in range(0, n_query, query_batch):
        end = min(start + query_batch, n_query)
        qf = q_feats[start:end].to(device=device, dtype=torch.float32)  # [b, D]
        sim = qf @ c_feats_dev.T                                        # [b, Nc]
        _, topk_idx = sim.topk(top_k, dim=-1, largest=True, sorted=True)  # [b, top_k]

        pos_b = pos_pad[start:end].to(device)                            # [b, max_pos]
        # For each row, check if topk_idx contains any of the row's positives.
        # topk_idx: [b, top_k, 1], pos_b: [b, 1, max_pos]
        eq = (topk_idx.unsqueeze(-1) == pos_b.unsqueeze(1))              # [b, top_k, max_pos]
        # Pad positions have value -1, which never matches a valid idx (>=0),
        # so they are correctly excluded from `.any(-1)`.
        hit_topk = eq.any(dim=-1)                                        # [b, top_k]
        hits_topk_all.append(hit_topk.cpu())
        n_query_valid = end
    hits_topk_all_t = torch.cat(hits_topk_all, dim=0)                    # [Nq, top_k]

    # ---- Recall@k --------------------------------------------------------
    metrics: Dict[str, float] = {}
    valid_mask = num_positives > 0                                       # queries with at least one mappable positive
    n_valid = int(valid_mask.sum().item())
    for k in ks:
        hit_k = hits_topk_all_t[:, :k].any(dim=-1).float()               # [Nq]
        r = (hit_k * valid_mask.float()).sum().item() / max(n_valid, 1)
        metrics[f"R@{k}"] = r

    # ---- nDCG@k ----------------------------------------------------------
    dcg = _dcg_at_k(hits_topk_all_t.float(), ndcg_k)                     # [Nq]
    idcg = _ideal_dcg_at_k(num_positives, ndcg_k)                        # [Nq]
    ndcg = torch.where(idcg > 0, dcg / idcg.clamp(min=1e-9), torch.zeros_like(dcg))
    metrics[f"nDCG@{ndcg_k}"] = (ndcg * valid_mask.float()).sum().item() / max(n_valid, 1)

    metrics["n_query"] = n_query
    metrics["n_query_valid"] = n_valid
    metrics["n_gallery"] = len(dids)
    metrics["pool"] = pool_name
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT)
    p.add_argument("--datasets", nargs="+", default=["OVEN", "INFOSEEK"])
    p.add_argument("--splits", nargs="+", default=["test", "test"])
    p.add_argument("--pool", type=str, default="SHARED",
                   help="Which cand pool to evaluate against (SHARED | OVEN | INFOSEEK).")
    p.add_argument("--query_batch", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.splits) != len(args.datasets):
        raise ValueError("--splits length must match --datasets length.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    all_metrics: Dict[str, Dict[str, float]] = {}
    for ds, sp in zip(args.datasets, args.splits):
        m = evaluate_one(
            dataset=ds, split=sp, pool_name=args.pool,
            out_root=args.out_root, device=device, query_batch=args.query_batch,
        )
        key = f"{ds}/{sp}@{args.pool}"
        all_metrics[key] = m
        pretty = "  ".join(
            f"{k}={m[k]:.4f}" for k in ("R@1", "R@5", "R@10", "R@50", "nDCG@10") if k in m
        )
        print(f"  -> {key}: {pretty}   "
              f"(gallery={m['n_gallery']}, queries={m['n_query_valid']}/{m['n_query']})")

    out_path = os.path.join(args.out_root, "eval_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[done] metrics -> {out_path}")


if __name__ == "__main__":
    main()
