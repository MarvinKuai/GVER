# -*- coding: utf-8 -*-
"""Per-dataset independent-pool evaluation.

Given the SHARED candidate pool feats extracted by ``extract_wemm_feats.py``:

  outputs/cand_pool/SHARED/{cand_feats.pt, cand_ids.json}   (816917 rows)

the pool is a concatenation of OVEN's task-8 pool (rows [0, 335135)) and
INFOSEEK's task-8 pool (rows [335135, 816917)).  This script evaluates each
dataset's queries against ONLY its own pool by slicing the SHARED assets
in-memory; no re-extraction is needed.

Positives handling: for each per-pool run, we keep the query count fixed
(denominator = n_query) but restrict the ground-truth doc-ids to those
present in the current pool.  Queries whose positives all live in the
OTHER pool contribute 0 to recall.  This matches ``GENIUS_baseline/eval_perpool.py``.

Metrics: R@{1,5,10,50}, nDCG@10 - all in the same format as ``eval_wemm.py``.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import torch


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_ROOT = os.path.join(HERE, "outputs")

# The SHARED pool is exactly OVEN|INFOSEEK concatenated, in that order.
OVEN_RANGE = (0, 335135)
INFOSEEK_RANGE = (335135, 816917)


def _dcg_at_k(hits_topk: torch.Tensor, k: int) -> torch.Tensor:
    hits = hits_topk[:, :k].float()
    denom = torch.log2(torch.arange(2, k + 2, device=hits.device, dtype=hits.dtype))
    return (hits / denom).sum(dim=-1)


def _ideal_dcg_at_k(num_pos: torch.Tensor, k: int) -> torch.Tensor:
    device = num_pos.device
    denom = 1.0 / torch.log2(torch.arange(2, k + 2, device=device, dtype=torch.float32))
    cum = torch.cumsum(denom, dim=0)
    n_clip = torch.clamp(num_pos.long(), max=k)
    idx = (n_clip - 1).clamp(min=0)
    idcg = cum[idx]
    idcg = torch.where(n_clip > 0, idcg, torch.zeros_like(idcg))
    return idcg


def evaluate_slice(
    dataset: str,
    split: str,
    pool_name: str,
    slice_range: Tuple[int, int],
    q_feats: torch.Tensor,
    c_feats_full: torch.Tensor,
    dids_full: Sequence[str],
    qids: Sequence[str],
    qid_to_pos_raw: Dict[str, List[str]],
    device: torch.device,
    query_batch: int,
    ks: Tuple[int, ...] = (1, 5, 10, 50),
    ndcg_k: int = 10,
) -> Dict[str, float]:
    s, e = slice_range
    dids = list(dids_full[s:e])
    c_feats = c_feats_full[s:e]
    did_to_idx = {d: i for i, d in enumerate(dids)}

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

    max_pos = max((len(x) for x in pos_lists), default=0)
    print(f"[eval] {dataset}/{split} vs pool='{pool_name}': "
          f"n_query={n_query}, n_gallery={len(dids)}, "
          f"dim={q_feats.shape[-1]}, max_pos_per_q={max_pos}, "
          f"queries_with_zero_mappable_pos={n_missing_all}, "
          f"queries_with_partial_pos_loss={n_missing_some}")

    if max_pos == 0:
        # Extremely unlikely, but be defensive.
        metrics = {f"R@{k}": 0.0 for k in ks}
        metrics[f"nDCG@{ndcg_k}"] = 0.0
        metrics.update(dict(n_query=n_query, n_query_valid=0,
                            n_gallery=len(dids), pool=pool_name))
        return metrics

    pos_pad = torch.full((n_query, max_pos), -1, dtype=torch.long)
    for i, lst in enumerate(pos_lists):
        if lst:
            pos_pad[i, : len(lst)] = torch.tensor(lst, dtype=torch.long)

    # Gallery to GPU once (fp32 for stable dot product).
    c_feats_dev = c_feats.to(device=device, dtype=torch.float32)

    hits_topk_all: List[torch.Tensor] = []
    for start in range(0, n_query, query_batch):
        end = min(start + query_batch, n_query)
        qf = q_feats[start:end].to(device=device, dtype=torch.float32)
        sim = qf @ c_feats_dev.T
        _, topk_idx = sim.topk(top_k, dim=-1, largest=True, sorted=True)
        pos_b = pos_pad[start:end].to(device)
        eq = (topk_idx.unsqueeze(-1) == pos_b.unsqueeze(1))
        hit_topk = eq.any(dim=-1)
        hits_topk_all.append(hit_topk.cpu())
    hits_topk_all_t = torch.cat(hits_topk_all, dim=0)

    metrics: Dict[str, float] = {}
    # IMPORTANT: denominator = n_query (fixed), to be consistent with the
    # SHARED-pool eval and with GENIUS_baseline/eval_perpool.py.  Queries whose
    # positives all live in the other pool count as 0.
    for k in ks:
        hit_k = hits_topk_all_t[:, :k].any(dim=-1).float()
        metrics[f"R@{k}"] = hit_k.sum().item() / max(n_query, 1)

    dcg = _dcg_at_k(hits_topk_all_t.float(), ndcg_k)
    idcg = _ideal_dcg_at_k(num_positives, ndcg_k)
    ndcg = torch.where(idcg > 0, dcg / idcg.clamp(min=1e-9), torch.zeros_like(dcg))
    metrics[f"nDCG@{ndcg_k}"] = ndcg.sum().item() / max(n_query, 1)

    metrics["n_query"] = n_query
    metrics["n_query_with_positive_in_pool"] = int((num_positives > 0).sum().item())
    metrics["n_gallery"] = len(dids)
    metrics["pool"] = pool_name

    # Free GPU
    del c_feats_dev
    torch.cuda.empty_cache()
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--query_batch", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_json", type=str,
                   default=os.path.join(DEFAULT_OUT_ROOT, "eval_metrics_perpool.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load SHARED gallery once.
    c_dir = os.path.join(args.out_root, "cand_pool", "SHARED")
    print(f"[load] SHARED gallery from {c_dir}")
    c_feats_full = torch.load(os.path.join(c_dir, "cand_feats.pt"),
                              map_location="cpu")
    with open(os.path.join(c_dir, "cand_ids.json"), "r", encoding="utf-8") as f:
        dids_full: List[str] = [str(x) for x in json.load(f)]
    print(f"[load]   feats={tuple(c_feats_full.shape)}  dids={len(dids_full)}")
    assert len(dids_full) == c_feats_full.shape[0] == 816917, \
        f"Unexpected SHARED gallery size: dids={len(dids_full)}, " \
        f"feats={c_feats_full.shape}"

    all_metrics: Dict[str, Dict[str, float]] = {}

    for ds, slice_range in [("OVEN", OVEN_RANGE), ("INFOSEEK", INFOSEEK_RANGE)]:
        q_dir = os.path.join(args.out_root, "queries", ds, args.split)
        q_feats = torch.load(os.path.join(q_dir, "query_feats.pt"),
                             map_location="cpu")
        with open(os.path.join(q_dir, "qids.json"), "r", encoding="utf-8") as f:
            qids = [str(x) for x in json.load(f)]
        with open(os.path.join(q_dir, "qid_to_positives.json"), "r",
                  encoding="utf-8") as f:
            qid_to_pos_raw = json.load(f)
        print(f"[load] {ds}/{args.split}: qfeats={tuple(q_feats.shape)}  "
              f"qids={len(qids)}")

        # (A) Evaluate against SHARED (full) pool.
        m_shared = evaluate_slice(
            dataset=ds, split=args.split, pool_name="SHARED",
            slice_range=(0, len(dids_full)),
            q_feats=q_feats, c_feats_full=c_feats_full,
            dids_full=dids_full, qids=qids, qid_to_pos_raw=qid_to_pos_raw,
            device=device, query_batch=args.query_batch,
        )
        key = f"{ds}/{args.split}@SHARED"
        all_metrics[key] = m_shared
        pretty = "  ".join(f"{k}={m_shared[k]:.4f}"
                           for k in ("R@1", "R@5", "R@10", "R@50", "nDCG@10"))
        print(f"  -> {key}: {pretty}")

        # (B) Evaluate against own pool.
        m_own = evaluate_slice(
            dataset=ds, split=args.split, pool_name=ds,
            slice_range=slice_range,
            q_feats=q_feats, c_feats_full=c_feats_full,
            dids_full=dids_full, qids=qids, qid_to_pos_raw=qid_to_pos_raw,
            device=device, query_batch=args.query_batch,
        )
        key = f"{ds}/{args.split}@{ds}"
        all_metrics[key] = m_own
        pretty = "  ".join(f"{k}={m_own[k]:.4f}"
                           for k in ("R@1", "R@5", "R@10", "R@50", "nDCG@10"))
        print(f"  -> {key}: {pretty}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[done] metrics -> {args.out_json}")


if __name__ == "__main__":
    main()
