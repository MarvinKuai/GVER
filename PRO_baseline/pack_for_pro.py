"""Pack CLIP-SF features (extracted by extract_clipsf_feats.py) into the
format that MGR_PRO's data.py expects.

Layout produced (under --out_root, e.g. outputs/data_packed):

    cand/
        cand_feats.pt            dict{img, text, img_mask, text_mask, id_to_index}
                                 img/text: fp32 [Nc, D]; masks: fp32 [Nc,]
                                 id_to_index: dict[int -> int]  (uses hash_item_id)
        cand_modalities.pt       fp32/long [Nc,]   (all 2, multimodal)
    train/
        query_feats.pt           dict{img, text, img_mask, text_mask, id_to_index}
        pairs.jsonl              [{qid, pos_cand_list}]   (single positive per query)
    val/
        query_feats.pt
        pairs.jsonl              (multi positives kept)
    test/
        query_feats.pt
        pairs.jsonl              (multi positives)
        qrels.txt                TREC format:  qid 0 did 1

qid / did encoding follows m-beir "<dataset_id>:<local_id>" convention.
The hashing (hash_query_id / hash_item_id) in data.py is used implicitly:
we store the *hashed integer* as id_to_index key so that PairEmbeddingDataset
can look up features by hash directly.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


# hash_query_id / hash_item_id constants must match src/pro/data.py
DATASET_CANDIDATE_BOUND = 10_000_000
DATASET_QUERY_BOUND = 500_000


def hash_query_id(value: str) -> int:
    if isinstance(value, str) and ":" in value:
        ds, local = value.split(":", 1)
        return int(ds) * DATASET_QUERY_BOUND + int(local)
    return int(value)


def hash_item_id(value: str) -> int:
    if isinstance(value, str) and ":" in value:
        ds, local = value.split(":", 1)
        return int(ds) * DATASET_CANDIDATE_BOUND + int(local)
    return int(value)


def _load_pt(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def pack_cand(clipsf_root: str, out_root: str, limit: int = -1,
              cand_prefix_filter: str | None = None) -> Dict[str, int]:
    """Pack SHARED cand pool into MGR_PRO multimodal feature store.

    If cand_prefix_filter is provided (e.g. "5:" or "6:"), only keep candidates
    whose original id starts with that prefix — used for independent per-dataset
    candidate pools (OVEN-only / INFOSEEK-only).
    """
    cand_dir = os.path.join(clipsf_root, "cand_pool", "SHARED")
    print(f"[cand] loading cand_img / cand_txt / cand_ids from {cand_dir}")
    cand_img = _load_pt(os.path.join(cand_dir, "cand_img.pt")).float()   # [Nc, D] fp16 -> fp32
    cand_txt = _load_pt(os.path.join(cand_dir, "cand_txt.pt")).float()
    with open(os.path.join(cand_dir, "cand_ids.json"), "r") as f:
        cand_ids: List[str] = json.load(f)
    assert cand_img.shape[0] == cand_txt.shape[0] == len(cand_ids), \
        f"shape mismatch: img={cand_img.shape} txt={cand_txt.shape} ids={len(cand_ids)}"

    if cand_prefix_filter is not None:
        keep_idx = [i for i, did in enumerate(cand_ids) if did.startswith(cand_prefix_filter)]
        keep_idx_t = torch.tensor(keep_idx, dtype=torch.long)
        cand_img = cand_img.index_select(0, keep_idx_t).contiguous()
        cand_txt = cand_txt.index_select(0, keep_idx_t).contiguous()
        cand_ids = [cand_ids[i] for i in keep_idx]
        print(f"[cand] filter prefix='{cand_prefix_filter}': kept {len(cand_ids)} / {len(keep_idx)+0} candidates")

    Nc = cand_img.shape[0]
    D = cand_img.shape[1]

    if limit > 0 and limit < Nc:
        cand_img = cand_img[:limit].contiguous()
        cand_txt = cand_txt[:limit].contiguous()
        cand_ids = cand_ids[:limit]
        Nc = limit
        print(f"[cand] SMOKE: truncated to first {Nc} candidates")

    # OVEN/INFOSEEK cand pool: every item has BOTH image and text -> masks = 1
    img_mask = torch.ones(Nc, dtype=torch.float32)
    text_mask = torch.ones(Nc, dtype=torch.float32)

    id_to_index: Dict[int, int] = {}
    for idx, did in enumerate(cand_ids):
        id_to_index[hash_item_id(did)] = idx

    out_dir = Path(out_root) / "cand"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "img": cand_img,
        "text": cand_txt,
        "img_mask": img_mask,
        "text_mask": text_mask,
        "id_to_index": id_to_index,
    }
    torch.save(payload, out_dir / "cand_feats.pt")

    # All cand are multimodal -> modality index 2
    modalities = {int(hash_item_id(did)): 2 for did in cand_ids}
    torch.save(modalities, out_dir / "cand_modalities.pt")

    print(f"[cand] saved {Nc} candidates (D={D}) to {out_dir}")
    return {"n_cand": Nc, "dim": D}


def _load_split_queries(clipsf_root: str, dataset: str, split: str,
                        limit: int = -1) -> Tuple[torch.Tensor, torch.Tensor, List[str], Dict[str, List[str]]]:
    """Load per-dataset query features and qid_to_positives at a given split.

    Returns (img [n,D] fp32, txt [n,D] fp32, qids [n], qid_to_positives)
    """
    qdir = os.path.join(clipsf_root, "queries", dataset, split)
    q_img = _load_pt(os.path.join(qdir, "query_img.pt")).float()
    q_txt = _load_pt(os.path.join(qdir, "query_txt.pt")).float()
    with open(os.path.join(qdir, "qids.json"), "r") as f:
        qids: List[str] = json.load(f)
    with open(os.path.join(qdir, "qid_to_positives.json"), "r") as f:
        qid_to_pos: Dict[str, List[str]] = json.load(f)
    assert q_img.shape[0] == q_txt.shape[0] == len(qids)
    if limit > 0 and limit < len(qids):
        q_img = q_img[:limit].contiguous()
        q_txt = q_txt[:limit].contiguous()
        qids = qids[:limit]
        qid_to_pos = {q: qid_to_pos[q] for q in qids if q in qid_to_pos}
    return q_img, q_txt, qids, qid_to_pos


def pack_split_queries(clipsf_root: str, out_root: str, split: str,
                       datasets: List[str], limit_per_ds: int = -1,
                       write_qrels: bool = False,
                       cand_hash_lookup: Dict[int, int] | None = None) -> Dict[str, int]:
    """Pack the queries for a given split by concatenating OVEN + INFOSEEK.

    - Stores query_feats.pt (multimodal FeatureStore)
    - Stores pairs.jsonl (one row per qid, only queries with >=1 positive kept)
    - Optionally writes qrels.txt (TREC) with all positives
    """
    all_img = []
    all_txt = []
    all_qids = []
    all_pairs = []          # for pairs.jsonl (first positive)
    all_qrel_rows = []      # for qrels.txt (all positives)
    n_dropped_no_pos = 0
    n_dropped_bad_pos = 0
    for ds in datasets:
        q_img, q_txt, qids, qid_to_pos = _load_split_queries(clipsf_root, ds, split, limit_per_ds)
        keep_mask = torch.zeros(len(qids), dtype=torch.bool)
        for i, qid in enumerate(qids):
            positives = qid_to_pos.get(qid, [])
            if not positives:
                n_dropped_no_pos += 1
                continue
            # If we have a cand_hash_lookup (smoke mode with truncated cand),
            # keep only queries whose FIRST positive is in the truncated pool
            filtered = positives
            if cand_hash_lookup is not None:
                filtered = [p for p in positives if hash_item_id(p) in cand_hash_lookup]
                if not filtered:
                    n_dropped_bad_pos += 1
                    continue
            keep_mask[i] = True
            all_pairs.append({"qid": qid, "pos_cand_list": [filtered[0]]})
            for did in filtered:
                all_qrel_rows.append(f"{qid} 0 {did} 1")
        kept_idx = torch.nonzero(keep_mask).squeeze(-1)
        all_img.append(q_img.index_select(0, kept_idx))
        all_txt.append(q_txt.index_select(0, kept_idx))
        all_qids.extend([qids[i] for i in kept_idx.tolist()])
        print(f"[queries:{split}:{ds}] kept {int(keep_mask.sum())}/{len(qids)} queries")

    q_img_all = torch.cat(all_img, dim=0)
    q_txt_all = torch.cat(all_txt, dim=0)
    Nq = q_img_all.shape[0]
    D = q_img_all.shape[1]

    img_mask = torch.ones(Nq, dtype=torch.float32)
    text_mask = torch.ones(Nq, dtype=torch.float32)
    id_to_index: Dict[int, int] = {hash_query_id(qid): idx for idx, qid in enumerate(all_qids)}

    out_dir = Path(out_root) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "img": q_img_all, "text": q_txt_all,
        "img_mask": img_mask, "text_mask": text_mask,
        "id_to_index": id_to_index,
    }, out_dir / "query_feats.pt")

    with (out_dir / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for row in all_pairs:
            f.write(json.dumps(row) + "\n")

    if write_qrels:
        with (out_dir / "qrels.txt").open("w", encoding="utf-8") as f:
            for line in all_qrel_rows:
                f.write(line + "\n")

    print(f"[queries:{split}] saved {Nq} queries (dropped no_pos={n_dropped_no_pos}, "
          f"bad_pos={n_dropped_bad_pos}) -> {out_dir}")
    return {"n_query": Nq, "dim": D, "n_pairs": len(all_pairs)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clipsf_root", required=True,
                   help="Root of CLIP-SF features (contains cand_pool/, queries/)")
    p.add_argument("--out_root", required=True)
    p.add_argument("--datasets", nargs="+", default=["OVEN", "INFOSEEK"])
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--limit_cand", type=int, default=-1,
                   help="SMOKE: truncate cand pool to first N")
    p.add_argument("--limit_query_per_ds", type=int, default=-1,
                   help="SMOKE: truncate queries per dataset per split")
    p.add_argument("--cand_prefix_filter", type=str, default=None,
                   help="Only keep candidates whose id starts with this prefix "
                        "(e.g. '5:' for OVEN-only or '6:' for INFOSEEK-only). "
                        "Enables independent per-dataset candidate pools.")
    args = p.parse_args()

    t0 = time.time()
    stats: Dict[str, Dict[str, int]] = {}

    # 1) cand
    stats["cand"] = pack_cand(args.clipsf_root, args.out_root, limit=args.limit_cand,
                              cand_prefix_filter=args.cand_prefix_filter)

    # For smoke mode or per-pool filter, build a lookup so we can drop queries
    # whose positives are outside the truncated / filtered cand pool.
    cand_hash_lookup: Dict[int, int] | None = None
    if args.limit_cand > 0 or args.cand_prefix_filter is not None:
        cand_payload = _load_pt(os.path.join(args.out_root, "cand", "cand_feats.pt"))
        cand_hash_lookup = cand_payload["id_to_index"]

    # 2) each split
    for split in args.splits:
        write_qrels = (split == "test")
        stats[f"queries_{split}"] = pack_split_queries(
            args.clipsf_root, args.out_root, split, args.datasets,
            limit_per_ds=args.limit_query_per_ds,
            write_qrels=write_qrels,
            cand_hash_lookup=cand_hash_lookup,
        )

    # Summary file
    summary_path = Path(args.out_root) / "pack_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"stats": stats, "args": vars(args),
                   "elapsed_sec": round(time.time() - t0, 1)}, f, indent=2)
    print(f"[done] pack summary saved: {summary_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
