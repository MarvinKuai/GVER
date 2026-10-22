# -*- coding: utf-8 -*-
"""Independent-pool T5 constrained beam search for GENIUS_R.

Why this script exists
----------------------
The default ``run_genius_r.py`` builds a single trie over the SHARED cand
pool (OVEN + INFOSEEK concatenated) and beams 100 SIDs per query against
that trie.  When we later evaluate on *sliced* pools (OVEN queries against
OVEN-only pool, etc.), most of those 100 beams end up matching SIDs that
only live in the *other* dataset's sub-pool, so the effective shortlist
collapses (~80% of OVEN beams live only in INFOSEEK codes -- see
``logs/diag_beam_hit.txt``).

This script fixes that by building a **separate trie per dataset** (from the
codes of that dataset's sub-pool only) and re-running beam search for each
dataset's queries against its own trie.  All 100 beams then have a chance
to hit its own pool.

Reuses
------
- outputs/genius_r/cand_pool/SHARED/cand_codes.pt   (sliced per-dataset)
- outputs/clipsf/cand_pool/SHARED/cand_ids.json     (to compute slice bounds)
- outputs/clipsf_raw/queries/<DS>/test/query_{img,txt}_raw.pt
- GENIUS T5 + RQ checkpoints                        (frozen)

Outputs
-------
outputs/genius_r/queries/<DS>/test/query_codes_topN_perpool.pt
    int32 [Nq, num_beams, codebook_level]
outputs/genius_r/cand_pool/<DS>/{cand_codes.pt, cand_ids.json, code_to_rows.pkl}
    (per-dataset sliced views, saved once)

Downstream
----------
Use ``eval_perpool.py --topN_suffix _perpool`` to evaluate.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from typing import Dict, List, Tuple

# Reuse the heavy lifting from run_genius_r.py: T5/RQ loading, beam search,
# code -> rows indexing.  Import order matters because ``run_genius_r``
# monkey-patches ``torch.load`` at import time.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

import run_genius_r as R                                            # noqa: E402
from run_genius_r import (                                          # noqa: E402
    _build_genius_config,
    load_generator,
    beam_search_queries,
    build_code_to_rows,
)


# ---------------------------------------------------------------------------
# Slice helpers (mirrors eval_perpool.compute_slice)
# ---------------------------------------------------------------------------
def compute_slice(cand_ids: List[str], ds: str) -> Tuple[int, int]:
    prefix = {"OVEN": "5:", "INFOSEEK": "6:"}[ds]
    start = end = -1
    for i, d in enumerate(cand_ids):
        if d.startswith(prefix):
            if start < 0:
                start = i
            end = i + 1
    assert start >= 0 and end > start, f"empty slice for {ds}"
    return start, end


# ---------------------------------------------------------------------------
# Trie install: reset ``model.trie_index`` to a new trie built from ``codes``
# ---------------------------------------------------------------------------
def install_trie(model, codes: torch.Tensor) -> int:
    """Rebuild the C++ trie inside ``model`` from ``codes`` (int32 [N, L]).

    Returns the number of unique codes registered.
    """
    tokenizer = model.tokenizer
    unique_codes = np.unique(codes.numpy(), axis=0).tolist()
    cand_codes_str = [model.transform_row(row) for row in unique_codes]
    cand_codes_ids = np.array(
        tokenizer(cand_codes_str, add_special_tokens=False).input_ids
    ).tolist()
    import models.generative_retriever.trie_cpp as Trie_Cpp  # noqa: WPS433
    model.trie_index = Trie_Cpp.Trie(cand_codes_ids)
    return len(unique_codes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clipsf_root", type=str,
                   default=os.path.join(HERE, "outputs", "clipsf"))
    p.add_argument("--clipsf_raw_root", type=str,
                   default=os.path.join(HERE, "outputs", "clipsf_raw"))
    p.add_argument("--out_root", type=str,
                   default=os.path.join(HERE, "outputs", "genius_r"))
    p.add_argument("--datasets", nargs="+", default=["OVEN", "INFOSEEK"])
    p.add_argument("--splits", nargs="+", default=["test", "test"])
    p.add_argument("--num_beams", type=int, default=50)
    p.add_argument("--beam_batch", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_suffix", type=str, default="_perpool",
                   help="Suffix appended to query_codes_topN filename.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Load shared codes + cand_ids once --------------------------------
    shared_cand_dir_clipsf = os.path.join(
        args.clipsf_root, "cand_pool", "SHARED")
    shared_cand_dir_geniusr = os.path.join(
        args.out_root, "cand_pool", "SHARED")
    with open(os.path.join(shared_cand_dir_clipsf, "cand_ids.json"),
              "r", encoding="utf-8") as f:
        cand_ids_full: List[str] = [str(x) for x in json.load(f)]
    codes_full: torch.Tensor = torch.load(
        os.path.join(shared_cand_dir_geniusr, "cand_codes.pt"),
        map_location="cpu", weights_only=True,
    )
    print(f"[shared] cand pool total={len(cand_ids_full)}  "
          f"codes shape={tuple(codes_full.shape)} dtype={codes_full.dtype}")

    slices: Dict[str, Tuple[int, int]] = {
        ds: compute_slice(cand_ids_full, ds) for ds in args.datasets
    }
    for ds, (s, e) in slices.items():
        print(f"[slice] {ds}: [{s}, {e})  size={e - s}")

    # ---- Persist per-dataset cand_codes / code_to_rows / cand_ids ---------
    for ds, (s, e) in slices.items():
        ds_dir = os.path.join(args.out_root, "cand_pool", ds)
        os.makedirs(ds_dir, exist_ok=True)
        codes_pt = os.path.join(ds_dir, "cand_codes.pt")
        ids_json = os.path.join(ds_dir, "cand_ids.json")
        idx_pkl = os.path.join(ds_dir, "code_to_rows.pkl")
        if args.overwrite or not os.path.isfile(codes_pt):
            torch.save(codes_full[s:e].contiguous(), codes_pt)
            print(f"[cand:{ds}] codes -> {codes_pt}  "
                  f"shape={(e - s, codes_full.shape[1])}")
        if args.overwrite or not os.path.isfile(ids_json):
            with open(ids_json, "w", encoding="utf-8") as f:
                json.dump(cand_ids_full[s:e], f)
            print(f"[cand:{ds}] ids   -> {ids_json}")
        if args.overwrite or not os.path.isfile(idx_pkl):
            sub_codes = torch.load(codes_pt, map_location="cpu",
                                   weights_only=True)
            idx = build_code_to_rows(sub_codes)
            with open(idx_pkl, "wb") as f:
                pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[cand:{ds}] code_to_rows  -> {idx_pkl}  "
                  f"unique_codes={len(idx)}")

    # ---- Which datasets still need beam search ----------------------------
    todo: List[Tuple[str, str]] = []
    for ds, sp in zip(args.datasets, args.splits):
        out_pt = os.path.join(args.out_root, "queries", ds, sp,
                              f"query_codes_topN{args.out_suffix}.pt")
        if (not args.overwrite) and os.path.isfile(out_pt):
            print(f"[skip] {out_pt}")
            continue
        todo.append((ds, sp))
    if not todo:
        print("[done] nothing to do -- all per-pool beams already exist.")
        return

    # ---- Load T5 generator ONCE -------------------------------------------
    cfg = _build_genius_config()
    model, _tok = load_generator(cfg, device)
    print(f"[gen] T5 generator loaded on {device}")

    # ---- Per-dataset trie + beam ------------------------------------------
    for ds, sp in todo:
        s, e = slices[ds]
        sub_codes = codes_full[s:e].contiguous()
        n_unique = install_trie(model, sub_codes)
        print(f"[trie:{ds}] built triecpp with {n_unique} unique codes "
              f"(pool size {sub_codes.shape[0]})")

        raw_dir = os.path.join(args.clipsf_raw_root, "queries", ds, sp)
        q_img = torch.load(os.path.join(raw_dir, "query_img_raw.pt"),
                           map_location="cpu", weights_only=True)
        q_txt = torch.load(os.path.join(raw_dir, "query_txt_raw.pt"),
                           map_location="cpu", weights_only=True)
        print(f"[query:{ds}/{sp}] img_raw={tuple(q_img.shape)}  "
              f"txt_raw={tuple(q_txt.shape)}")

        t0 = time.time()
        top = beam_search_queries(
            model, q_img, q_txt, device,
            num_beams=args.num_beams,
            batch_size=args.beam_batch,
            tag=f"{ds}/{sp}[perpool]",
        )
        dt = time.time() - t0
        print(f"[query:{ds}/{sp}] beam done in {dt:.1f}s  "
              f"top.shape={tuple(top.shape)}")

        q_out_dir = os.path.join(args.out_root, "queries", ds, sp)
        os.makedirs(q_out_dir, exist_ok=True)
        out_pt = os.path.join(q_out_dir,
                              f"query_codes_topN{args.out_suffix}.pt")
        torch.save(top, out_pt)
        print(f"[query:{ds}/{sp}] saved -> {out_pt}")

    print("[done] run_genius_r_perpool.py")


if __name__ == "__main__":
    main()
