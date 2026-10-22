# -*- coding: utf-8 -*-
"""Per-dataset independent-pool evaluation for CLIP-SF / GENIUS / GENIUS-R.

The SHARED cand_pool used by the default eval scripts is the concatenation
of OVEN's task-8 pool (rows [0, 335135)) and INFOSEEK's task-8 pool
(rows [335135, 816917)).  This script evaluates each dataset's queries
against ONLY its own pool, by slicing the existing SHARED assets in-memory.

Reused assets (nothing is re-computed):
  outputs/clipsf/cand_pool/SHARED/{cand_feats.pt, cand_ids.json}
  outputs/clipsf/queries/<DS>/test/{query_feats.pt, qids.json, qid_to_positives.json}
  outputs/genius_r/cand_pool/SHARED/{cand_codes.pt}         (recomputed code_to_rows on the fly)
  outputs/genius_r/queries/<DS>/test/query_codes_topN.pt    (unchanged - trie is over SHARED)

Metrics: R@1 / R@5 / R@10 / R@50 / nDCG@10, per (dataset, method).

Notes on GENIUS/GENIUS-R with sliced pools:
- The T5 beam search generated ``query_codes_topN.pt`` under the SHARED trie.
  We reuse those SIDs directly.  When looking up a SID against a sliced pool,
  the bucket may be empty (all rows of that SID live in the OTHER dataset's
  pool).  Such beams are simply skipped, matching the same expand-then-rerank
  logic as ``eval_genius_r.py``.
- GENIUS-R still reranks the surviving shortlist with CLIP-SF cosine on the
  sliced ``cand_feats``.
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
KS: Tuple[int, ...] = (1, 5, 10, 50)
NDCG_K = 10

# ---------------------------------------------------------------------------
# Pool slicing
# ---------------------------------------------------------------------------
def compute_slice(cand_ids: Sequence[str], ds: str) -> Tuple[int, int]:
    """Return [start, end) index range of dataset ``ds`` inside the SHARED pool.

    OVEN dids start with ``5:``; INFOSEEK dids start with ``6:``.  The SHARED
    pool is constructed as OVEN | INFOSEEK contiguously (see
    ``build_shared_cand_bundle`` in mbeir_data.py), so a single slice is enough.
    """
    prefix = {"OVEN": "5:", "INFOSEEK": "6:"}[ds]
    start = end = -1
    for i, d in enumerate(cand_ids):
        if d.startswith(prefix):
            if start < 0:
                start = i
            end = i + 1
    assert start >= 0 and end > start, f"empty slice for {ds}"
    # Sanity: the slice must be contiguous.
    for d in cand_ids[start:end]:
        assert d.startswith(prefix), \
            f"non-contiguous {ds} slice at index; found {d!r}"
    return start, end


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _dcg_at_k(hits_topk: torch.Tensor, k: int) -> torch.Tensor:
    hits = hits_topk[:, :k].float()
    denom = torch.log2(torch.arange(2, k + 2, device=hits.device,
                                     dtype=hits.dtype))
    return (hits / denom).sum(dim=-1)


def _ideal_dcg_at_k(num_pos: torch.Tensor, k: int) -> torch.Tensor:
    device = num_pos.device
    denom = 1.0 / torch.log2(torch.arange(2, k + 2, device=device,
                                           dtype=torch.float32))
    cum = torch.cumsum(denom, dim=0)
    n_clip = torch.clamp(num_pos.long(), max=k)
    idx = (n_clip - 1).clamp(min=0)
    idcg = cum[idx]
    idcg = torch.where(n_clip > 0, idcg, torch.zeros_like(idcg))
    return idcg


# ---------------------------------------------------------------------------
# CLIP-SF: dense cosine on sliced pool
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_clipsf_on_pool(
    ds: str,
    cand_ids_full: Sequence[str],
    cand_feats_full: torch.Tensor,   # fp16 [Nc_full, D]
    slice_range: Tuple[int, int],
    q_feats: torch.Tensor,           # fp16 [Nq, D]
    qids: Sequence[str],
    qid_to_pos: Dict[str, List[str]],
    device: torch.device,
    query_batch: int = 256,
) -> Dict[str, float]:
    s, e = slice_range
    cand_ids = cand_ids_full[s:e]
    cand_feats = cand_feats_full[s:e]
    did_to_idx = {d: i for i, d in enumerate(cand_ids)}
    n_query = len(qids)

    pos_lists: List[List[int]] = []
    num_positives = torch.zeros(n_query, dtype=torch.long)
    n_missing_all = 0
    n_missing_some = 0
    for i, qid in enumerate(qids):
        raw = qid_to_pos.get(qid, []) or []
        mapped = [did_to_idx[d] for d in raw if d in did_to_idx]
        if raw and not mapped:
            n_missing_all += 1
        elif len(mapped) < len(raw):
            n_missing_some += 1
        pos_lists.append(mapped)
        num_positives[i] = len(mapped)

    top_k = max(max(KS), NDCG_K)
    max_pos = max((len(x) for x in pos_lists), default=0)
    pos_pad = torch.full((n_query, max(max_pos, 1)), -1, dtype=torch.long)
    for i, lst in enumerate(pos_lists):
        if lst:
            pos_pad[i, : len(lst)] = torch.tensor(lst, dtype=torch.long)

    print(f"[clipsf/{ds}] gallery={len(cand_ids)} queries={n_query} "
          f"missing_all={n_missing_all} missing_some={n_missing_some}")

    cand_feats_dev = cand_feats.to(device=device, dtype=torch.float32)
    hits_all: List[torch.Tensor] = []
    for start in range(0, n_query, query_batch):
        end = min(start + query_batch, n_query)
        qf = q_feats[start:end].to(device=device, dtype=torch.float32)
        sim = qf @ cand_feats_dev.T
        _, topk_idx = sim.topk(top_k, dim=-1, largest=True, sorted=True)
        pos_b = pos_pad[start:end].to(device)
        eq = (topk_idx.unsqueeze(-1) == pos_b.unsqueeze(1))
        hits_all.append(eq.any(dim=-1).cpu())
    hits_t = torch.cat(hits_all, dim=0)

    metrics: Dict[str, float] = {}
    valid_mask = num_positives > 0
    n_valid = int(valid_mask.sum().item())
    for k in KS:
        hit_k = hits_t[:, :k].any(dim=-1).float()
        metrics[f"R@{k}"] = (hit_k * valid_mask.float()).sum().item() / max(n_valid, 1)
    dcg = _dcg_at_k(hits_t.float(), NDCG_K)
    idcg = _ideal_dcg_at_k(num_positives, NDCG_K)
    ndcg = torch.where(idcg > 0, dcg / idcg.clamp(min=1e-9), torch.zeros_like(dcg))
    metrics[f"nDCG@{NDCG_K}"] = (ndcg * valid_mask.float()).sum().item() / max(n_valid, 1)
    metrics["n_query"] = n_query
    metrics["n_query_valid"] = n_valid
    metrics["n_gallery"] = len(cand_ids)
    metrics["n_missing_all_pos"] = n_missing_all
    return metrics


# ---------------------------------------------------------------------------
# GENIUS / GENIUS-R: expand beam SIDs on sliced pool
# ---------------------------------------------------------------------------
def build_code_to_rows_local(codes_slice: torch.Tensor) -> Dict[Tuple[int, ...], List[int]]:
    idx: Dict[Tuple[int, ...], List[int]] = {}
    for i, c in enumerate(codes_slice.tolist()):
        idx.setdefault(tuple(c), []).append(i)
    return idx


@torch.no_grad()
def eval_genius_on_pool(
    ds: str,
    cand_ids_full: Sequence[str],
    slice_range: Tuple[int, int],
    codes_full: torch.Tensor,           # int32 [Nc_full, L]
    query_codes_topN: torch.Tensor,     # int32 [Nq, num_beams, L]
    qids: Sequence[str],
    qid_to_pos: Dict[str, List[str]],
    q_feats: torch.Tensor,              # fp16 [Nq, D] (only used if rerank=True)
    cand_feats_full: torch.Tensor,      # fp16 [Nc_full, D] (only used if rerank=True)
    device: torch.device,
    rerank: bool,
    topk_final: int = 50,
    max_cand_per_q: int = 4096,
) -> Dict[str, float]:
    s, e = slice_range
    cand_ids = cand_ids_full[s:e]
    codes_slice = codes_full[s:e]
    code_to_rows = build_code_to_rows_local(codes_slice)
    cid_to_row = {c: i for i, c in enumerate(cand_ids)}

    if rerank:
        cand_feats_sub = cand_feats_full[s:e].to(device=device,
                                                  dtype=torch.float16)

    Nq = len(qids)
    hits = {k: 0.0 for k in KS}
    ndcg_sum = 0.0
    n_valid = 0
    n_zero_cands = 0
    total_cands = 0

    tag = f"{'genius_r' if rerank else 'genius'}/{ds}"
    t0 = time.time()
    for i in range(Nq):
        qid = qids[i]
        positives = qid_to_pos.get(qid, []) or []
        if not positives:
            continue
        pos_rows = [cid_to_row[c] for c in positives if c in cid_to_row]
        if not pos_rows:
            # No positive lives in this pool -> not a valid retrieval task
            continue
        n_valid += 1

        cand_rows: List[int] = []
        seen = set()
        for beam in query_codes_topN[i].tolist():
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
            # Match eval_genius(_r).py behaviour: still counted as a valid query
            # (recall is trivially 0 because no docs were returned).
            n_zero_cands += 1
            continue
        total_cands += len(cand_rows)

        if rerank:
            q = q_feats[i].to(device=device, dtype=torch.float16).unsqueeze(0)
            cand_sub = cand_feats_sub.index_select(
                0, torch.tensor(cand_rows, device=device, dtype=torch.long))
            sims = (q @ cand_sub.t()).squeeze(0).float()
            order = torch.argsort(sims, descending=True)[:topk_final].cpu().tolist()
            ranked_rows = [cand_rows[o] for o in order]
        else:
            ranked_rows = cand_rows[:topk_final]

        pos_set = set(pos_rows)
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

        if (n_valid) % 2000 == 0:
            dt = time.time() - t0
            print(f"  [{tag}] {i+1}/{Nq} ({(i+1)/dt:.1f}/s, {dt:.1f}s)",
                  flush=True)

    metrics = {f"R@{k}": (hits[k] / n_valid) if n_valid else 0.0 for k in KS}
    metrics[f"nDCG@{NDCG_K}"] = (ndcg_sum / n_valid) if n_valid else 0.0
    metrics["n_query"] = Nq
    metrics["n_query_valid"] = n_valid
    metrics["n_gallery"] = len(cand_ids)
    metrics["n_zero_cand_queries"] = n_zero_cands
    metrics["avg_shortlist_size"] = (
        total_cands / max(n_valid - n_zero_cands, 1))
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clipsf_root", type=str,
                   default=os.path.join(HERE, "outputs", "clipsf"))
    p.add_argument("--geniusr_root", type=str,
                   default=os.path.join(HERE, "outputs", "genius_r"))
    p.add_argument("--out_json", type=str,
                   default=os.path.join(HERE, "outputs",
                                        "eval_metrics_final.json"))
    p.add_argument("--datasets", nargs="+", default=["OVEN", "INFOSEEK"])
    p.add_argument("--methods", nargs="+",
                   default=["clipsf", "genius", "genius_r"],
                   choices=["clipsf", "genius", "genius_r"])
    p.add_argument("--pool_mode", type=str, default="both",
                   choices=["shared", "perpool", "both"],
                   help="Which candidate pool(s) to evaluate against. "
                        "'shared' = full SHARED pool (816K). "
                        "'perpool' = each dataset's own slice. "
                        "'both' = do both back-to-back.")
    p.add_argument("--query_batch", type=int, default=256)
    p.add_argument("--topk_final", type=int, default=50)
    p.add_argument("--max_cand_per_q", type=int, default=4096)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _run_pool(
    pool_tag: str,
    slice_range: Tuple[int, int],
    topN_filename: str,
    args: argparse.Namespace,
    cand_ids_full: List[str],
    cand_feats_full,
    codes_full,
    device: torch.device,
    all_metrics: Dict[str, Dict[str, float]],
) -> None:
    """Run every (dataset, method) for a single pool configuration.

    ``slice_range`` = [s, e); when pool_tag == 'SHARED' this is [0, N_full).
    Only ``cand_ids_full[s:e]`` participates in retrieval; positives that fall
    outside the slice are simply invisible to recall.
    """
    for ds in args.datasets:
        q_dir = os.path.join(args.clipsf_root, "queries", ds, "test")
        with open(os.path.join(q_dir, "qids.json"), "r", encoding="utf-8") as f:
            qids = [str(x) for x in json.load(f)]
        with open(os.path.join(q_dir, "qid_to_positives.json"), "r",
                  encoding="utf-8") as f:
            qid_to_pos: Dict[str, List[str]] = json.load(f)

        q_feats = None
        if "clipsf" in args.methods or "genius_r" in args.methods:
            q_feats = torch.load(os.path.join(q_dir, "query_feats.pt"),
                                 map_location="cpu", weights_only=True)

        topN = None
        if "genius" in args.methods or "genius_r" in args.methods:
            topN_path = os.path.join(args.geniusr_root, "queries", ds,
                                     "test", topN_filename)
            print(f"[topN][{pool_tag}] loading {topN_path}")
            topN = torch.load(topN_path, map_location="cpu", weights_only=True)

        # Per-pool slice_range: for SHARED it's the entire pool; for perpool
        # it's the dataset-specific slice we precomputed.
        s, e = slice_range if pool_tag != "SHARED" else (0, len(cand_ids_full))

        for method in args.methods:
            key = f"{ds}@{pool_tag}::{method}"
            print(f"\n===== {key} =====")
            if method == "clipsf":
                m = eval_clipsf_on_pool(
                    ds=ds, cand_ids_full=cand_ids_full,
                    cand_feats_full=cand_feats_full,
                    slice_range=(s, e),
                    q_feats=q_feats, qids=qids, qid_to_pos=qid_to_pos,
                    device=device, query_batch=args.query_batch,
                )
            elif method == "genius":
                m = eval_genius_on_pool(
                    ds=ds, cand_ids_full=cand_ids_full,
                    slice_range=(s, e),
                    codes_full=codes_full,
                    query_codes_topN=topN,
                    qids=qids, qid_to_pos=qid_to_pos,
                    q_feats=q_feats,
                    cand_feats_full=cand_feats_full,
                    device=device, rerank=False,
                    topk_final=args.topk_final,
                    max_cand_per_q=args.max_cand_per_q,
                )
            elif method == "genius_r":
                m = eval_genius_on_pool(
                    ds=ds, cand_ids_full=cand_ids_full,
                    slice_range=(s, e),
                    codes_full=codes_full,
                    query_codes_topN=topN,
                    qids=qids, qid_to_pos=qid_to_pos,
                    q_feats=q_feats,
                    cand_feats_full=cand_feats_full,
                    device=device, rerank=True,
                    topk_final=args.topk_final,
                    max_cand_per_q=args.max_cand_per_q,
                )
            else:
                continue
            m["method"] = method
            m["dataset"] = ds
            m["pool"] = pool_tag
            all_metrics[key] = m
            print(f"  -> R@1={m['R@1']:.4f} R@5={m['R@5']:.4f} "
                  f"R@10={m['R@10']:.4f} R@50={m['R@50']:.4f} "
                  f"nDCG@10={m['nDCG@10']:.4f}   "
                  f"(gallery={m['n_gallery']}, valid={m['n_query_valid']}/{m['n_query']})")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Load SHARED assets ONCE -----------------------------------------
    shared_cand_dir = os.path.join(args.clipsf_root, "cand_pool", "SHARED")
    with open(os.path.join(shared_cand_dir, "cand_ids.json"), "r",
              encoding="utf-8") as f:
        cand_ids_full = [str(x) for x in json.load(f)]
    print(f"[shared] cand pool total = {len(cand_ids_full)}")

    need_feats = ("clipsf" in args.methods) or ("genius_r" in args.methods)
    need_codes = ("genius" in args.methods) or ("genius_r" in args.methods)

    if need_feats:
        cand_feats_full = torch.load(
            os.path.join(shared_cand_dir, "cand_feats.pt"),
            map_location="cpu", weights_only=True)
        print(f"[shared] cand_feats {tuple(cand_feats_full.shape)} "
              f"dtype={cand_feats_full.dtype}")
    else:
        cand_feats_full = None  # type: ignore[assignment]

    if need_codes:
        codes_full = torch.load(
            os.path.join(args.geniusr_root, "cand_pool", "SHARED", "cand_codes.pt"),
            map_location="cpu", weights_only=True)
        print(f"[shared] cand_codes {tuple(codes_full.shape)} "
              f"dtype={codes_full.dtype}")
    else:
        codes_full = None  # type: ignore[assignment]

    slices: Dict[str, Tuple[int, int]] = {
        ds: compute_slice(cand_ids_full, ds) for ds in args.datasets
    }
    for ds, (s, e) in slices.items():
        print(f"[slice] {ds}: [{s}, {e})  size={e-s}")

    all_metrics: Dict[str, Dict[str, float]] = {}

    if args.pool_mode in ("shared", "both"):
        print("\n########## POOL_MODE = SHARED (816K union) ##########")
        _run_pool(
            pool_tag="SHARED",
            slice_range=(0, len(cand_ids_full)),
            topN_filename="query_codes_topN.pt",
            args=args, cand_ids_full=cand_ids_full,
            cand_feats_full=cand_feats_full, codes_full=codes_full,
            device=device, all_metrics=all_metrics,
        )

    if args.pool_mode in ("perpool", "both"):
        print("\n########## POOL_MODE = PERPOOL (dataset-owned slice) ##########")
        for ds in args.datasets:
            _run_pool(
                pool_tag=f"{ds}_pool",
                slice_range=slices[ds],
                topN_filename="query_codes_topN_perpool.pt",
                args=argparse.Namespace(**{**vars(args), "datasets": [ds]}),
                cand_ids_full=cand_ids_full,
                cand_feats_full=cand_feats_full, codes_full=codes_full,
                device=device, all_metrics=all_metrics,
            )

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[done] metrics -> {args.out_json}")


if __name__ == "__main__":
    main()
