# -*- coding: utf-8 -*-
"""Extract Qwen3-VL-Embedding features for M-BEIR task 8 (OVEN + Infoseek).

Unlike the CLIP baseline (dual-tower with score-fusion), Qwen3-VL-Embedding is
a **single-tower** multimodal encoder: image + text are fed together into one
Qwen3-VL forward pass, and the [EOS] hidden state is L2-normalised as the
final vector (dim = 4096 for the 8B variant).

Design
------
* One process per GPU.  ``--rank R --world W`` partitions the index space so
  each worker owns ``list(indices)[R::W]`` (interleaved striding for better
  load balance; heavy images are spread across workers).
* Shard outputs land in
    outputs/cand_pool/<POOL>/shards/shard_{R:02d}_of_{W:02d}.pt
  as ``{"feats": [Nr, D] fp16, "indices": [Nr] int64}`` (indices refer to the
  original ordered list).
* A final ``--stage merge`` step gathers all shards, reassembles the [N, D]
  tensor in original order and dumps
    cand_pool/<POOL>/cand_feats.pt   (fp16)
    cand_pool/<POOL>/cand_ids.json
    cand_pool/<POOL>/meta.json
  which is exactly the layout consumed by ``eval_qwen3vl.py`` (a symlink to
  the CLIP baseline's evaluator).
* Queries are typically small (~32K); by default we run rank=0/world=1 and
  the same code path handles them without sharding.

Usage
-----
  # smoke test (single GPU, tiny slice)
  python3 extract_qwen3vl_feats.py --stage cand --pools SHARED --limit 20 \
        --rank 0 --world 1

  # 8-GPU parallel cand pool encoding (invoked from run_qwen3vl_baseline.sh)
  for R in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$R python3 extract_qwen3vl_feats.py \
          --stage cand --pools SHARED --rank $R --world 8 &
  done
  wait
  python3 extract_qwen3vl_feats.py --stage merge --pools SHARED

  # queries (fast, single card is enough)
  CUDA_VISIBLE_DEVICES=0 python3 extract_qwen3vl_feats.py --stage query \
      --datasets OVEN INFOSEEK --splits test test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Sequence

import torch
from PIL import Image

# --- import the Qwen3-VL-Embedding framework -------------------------------
# The repo ships its own package layout ("src/models/..."); we add the repo
# root to sys.path so `from src.models.qwen3_vl_embedding import ...` works
# without needing an install.
HERE = os.path.dirname(os.path.abspath(__file__))
GM2M_ROOT = os.path.dirname(HERE)
WORKSPACE = os.path.dirname(GM2M_ROOT)
QWEN3VL_ROOT = os.path.join(WORKSPACE, "baseline", "Qwen3-VL-Embedding")
if QWEN3VL_ROOT not in sys.path:
    sys.path.insert(0, QWEN3VL_ROOT)

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402

from mbeir_data import (  # noqa: E402
    MbeirCandBundle,
    MbeirQueryBundle,
    build_shared_cand_bundle,
    load_cand_bundle,
    load_query_bundle,
)


DEFAULT_MODEL = os.path.join(WORKSPACE, "baseline", "Qwen3-VL-Embedding",
                             "models", "Qwen3-VL-Embedding-8B")
DEFAULT_OUT_ROOT = os.path.join(HERE, "outputs")

# Qwen3-VL-Embedding official default instruction for retrieval.  UniIR's
# task instruction (from query_instructions.tsv) is folded into the query
# **text** by mbeir_data.py; we still pass a top-level "instruction" here
# because Qwen3VL-Embedding treats it as the system prompt.
DEFAULT_INSTR_QUERY = "Retrieve images or text relevant to the user's query."
DEFAULT_INSTR_CAND = "Represent the document for retrieval."


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------
def _build_input(
    text: str,
    image_getter: Callable[[], Image.Image],
    instruction: str,
) -> Dict:
    """Assemble one Qwen3-VL-Embedding input dict.

    We resolve the PIL image at call-time (lazy) since Qwen3-VL-Embedding
    accepts PIL.Image.Image in the "image" field.
    """
    img = image_getter()
    return {"text": text, "image": img, "instruction": instruction}


@torch.no_grad()
def encode_qwen3vl(
    embedder: Qwen3VLEmbedder,
    image_getters: Sequence[Callable[[], Image.Image]],
    texts: Sequence[str],
    instruction: str,
    batch_size: int,
    tag: str,
    log_every: int = 5,
) -> torch.Tensor:
    """Encode (image, text) pairs into fp16 [N, D] L2-normalised features."""
    assert len(image_getters) == len(texts)
    n = len(texts)
    all_feats: List[torch.Tensor] = []
    t0 = time.time()
    seen = 0
    print(f"  [{tag}] starting encode: n={n}, batch_size={batch_size}", flush=True)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_inputs = []
        for idx in range(start, end):
            try:
                item = _build_input(texts[idx], image_getters[idx], instruction)
            except Exception as e:
                # Fallback: text-only if image fails to load
                print(f"  [{tag}] warn: image load failed at idx={idx}: {e}; "
                      f"falling back to text-only.", flush=True)
                item = {"text": texts[idx], "instruction": instruction}
            batch_inputs.append(item)

        embs = embedder.process(batch_inputs, normalize=True)   # [b, D] fp32/bf16 on GPU
        all_feats.append(embs.detach().half().cpu())

        seen = end
        step = start // batch_size
        if step % log_every == 0 or seen == n:
            dt = time.time() - t0
            rate = seen / max(dt, 1e-9)
            eta = (n - seen) / max(rate, 1e-9)
            print(f"  [{tag}] {seen}/{n} ({rate:.2f}/s, elapsed {dt:.1f}s, "
                  f"eta {eta/60:.1f}min)", flush=True)
    return torch.cat(all_feats, dim=0)


# ---------------------------------------------------------------------------
# Slice helper
# ---------------------------------------------------------------------------
def _slice_indices(n: int, rank: int, world: int, limit: int = 0) -> List[int]:
    """Return this rank's index list: [rank, rank+world, rank+2*world, ...].

    ``limit`` (>0) caps the total number of *original* indices considered,
    for smoke tests.
    """
    total = n if limit <= 0 else min(n, limit)
    return list(range(rank, total, world))


# ---------------------------------------------------------------------------
# Stage: cand (per-rank shard)
# ---------------------------------------------------------------------------
def extract_cand_shard(
    pool_name: str,
    embedder: Qwen3VLEmbedder,
    batch_size: int,
    out_root: str,
    rank: int,
    world: int,
    instruction: str,
    limit: int,
    overwrite: bool,
) -> dict:
    out_dir = os.path.join(out_root, "cand_pool", pool_name)
    shard_dir = os.path.join(out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    shard_pt = os.path.join(shard_dir, f"shard_{rank:02d}_of_{world:02d}.pt")
    ids_all_json = os.path.join(out_dir, "cand_ids.json")   # only rank 0 writes this
    if (not overwrite) and os.path.isfile(shard_pt):
        print(f"[skip] cand shard {shard_pt} already exists.")
        return {"pool": pool_name, "rank": rank, "world": world, "skipped": True}

    print(f"\n=== [rank={rank}/{world}] cand:{pool_name} loading bundle ===", flush=True)
    t_bundle = time.time()
    if pool_name == "SHARED":
        bundle: MbeirCandBundle = build_shared_cand_bundle()
    else:
        bundle = load_cand_bundle(pool_name)
    print(f"    n_cand(total)={bundle.n}  (bundle build took {time.time()-t_bundle:.1f}s)",
          flush=True)

    # Rank 0 dumps the full ordered dids list once (used by merge + evaluator).
    if rank == 0 and (overwrite or not os.path.isfile(ids_all_json)):
        with open(ids_all_json, "w", encoding="utf-8") as f:
            json.dump(bundle.dids, f)
        print(f"    [rank=0] wrote cand_ids.json ({len(bundle.dids)} dids)")

    idxs = _slice_indices(bundle.n, rank, world, limit)
    print(f"    [rank={rank}] owns {len(idxs)} items "
          f"(first={idxs[:3]}, last={idxs[-3:] if len(idxs)>=3 else idxs})")

    sub_getters = [bundle.cand_image_getters[i] for i in idxs]
    sub_texts = [bundle.cand_texts[i] for i in idxs]

    t0 = time.time()
    feats = encode_qwen3vl(
        embedder=embedder,
        image_getters=sub_getters,
        texts=sub_texts,
        instruction=instruction,
        batch_size=batch_size,
        tag=f"cand:{pool_name}:r{rank}/{world}",
    )
    dt = time.time() - t0

    torch.save(
        {"feats": feats, "indices": torch.tensor(idxs, dtype=torch.long)},
        shard_pt,
    )
    meta = {
        "pool": pool_name, "rank": rank, "world": world,
        "n_local": len(idxs), "n_total": bundle.n,
        "feat_dim": int(feats.shape[-1]), "dtype": str(feats.dtype),
        "batch_size": batch_size, "time_sec": round(dt, 2),
        "instruction": instruction, "limit": limit,
    }
    with open(os.path.join(shard_dir, f"shard_{rank:02d}_of_{world:02d}.meta.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"    done: shard r{rank} -> {shard_pt}  ({dt/60:.2f}min)")
    return meta


# ---------------------------------------------------------------------------
# Stage: merge shards -> cand_feats.pt
# ---------------------------------------------------------------------------
def merge_cand_shards(pool_name: str, out_root: str, world: int) -> dict:
    out_dir = os.path.join(out_root, "cand_pool", pool_name)
    shard_dir = os.path.join(out_dir, "shards")
    ids_all_json = os.path.join(out_dir, "cand_ids.json")
    if not os.path.isfile(ids_all_json):
        raise FileNotFoundError(f"cand_ids.json missing at {ids_all_json}; "
                                f"run cand stage with rank=0 first.")
    with open(ids_all_json, "r", encoding="utf-8") as f:
        dids = json.load(f)
    n_total = len(dids)

    shards = []
    covered = torch.zeros(n_total, dtype=torch.bool)
    dim = None
    for r in range(world):
        p = os.path.join(shard_dir, f"shard_{r:02d}_of_{world:02d}.pt")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"missing shard: {p}")
        obj = torch.load(p, map_location="cpu")
        shards.append(obj)
        if dim is None:
            dim = obj["feats"].shape[-1]
        elif obj["feats"].shape[-1] != dim:
            raise RuntimeError(f"dim mismatch: shard r{r} dim={obj['feats'].shape[-1]} vs {dim}")
        idxs = obj["indices"]
        if covered[idxs].any():
            dup = int(covered[idxs].sum().item())
            raise RuntimeError(f"shard r{r}: {dup} indices overlap with earlier shards.")
        covered[idxs] = True

    n_covered = int(covered.sum().item())
    if n_covered != n_total:
        missing = int(n_total - n_covered)
        raise RuntimeError(f"merge coverage incomplete: {n_covered}/{n_total} "
                           f"({missing} rows missing)")

    print(f"[merge] all {world} shards cover {n_covered}/{n_total} rows, dim={dim}")

    full = torch.zeros(n_total, dim, dtype=torch.float16)
    for obj in shards:
        full[obj["indices"]] = obj["feats"]
    feat_pt = os.path.join(out_dir, "cand_feats.pt")
    torch.save(full, feat_pt)

    meta = {"pool": pool_name, "n_cand": n_total, "feat_dim": int(dim),
            "dtype": "torch.float16", "world": world, "merged_from": "shards/"}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[merge] wrote {feat_pt} (shape={tuple(full.shape)})")
    return meta


# ---------------------------------------------------------------------------
# Stage: query (sharded like cand, but usually world=1 since queries are small)
# ---------------------------------------------------------------------------
def extract_query_shard(
    dataset: str,
    split: str,
    embedder: Qwen3VLEmbedder,
    batch_size: int,
    out_root: str,
    rank: int,
    world: int,
    instruction: str,
    limit: int,
    overwrite: bool,
) -> dict:
    out_dir = os.path.join(out_root, "queries", dataset, split)
    os.makedirs(out_dir, exist_ok=True)
    feat_pt = os.path.join(out_dir, "query_feats.pt")
    qids_json = os.path.join(out_dir, "qids.json")
    pos_json = os.path.join(out_dir, "qid_to_positives.json")
    shard_dir = os.path.join(out_dir, "shards") if world > 1 else None

    if world == 1:
        if (not overwrite) and os.path.isfile(feat_pt) and os.path.isfile(qids_json):
            print(f"[skip] query {dataset}/{split} already extracted at {out_dir}")
            return {"dataset": dataset, "split": split, "skipped": True}
    else:
        os.makedirs(shard_dir, exist_ok=True)
        shard_pt = os.path.join(shard_dir, f"shard_{rank:02d}_of_{world:02d}.pt")
        if (not overwrite) and os.path.isfile(shard_pt):
            print(f"[skip] query shard {shard_pt}")
            return {"dataset": dataset, "split": split, "rank": rank, "skipped": True}

    print(f"\n=== [rank={rank}/{world}] query:{dataset}/{split} loading ===")
    bundle: MbeirQueryBundle = load_query_bundle(dataset, split)
    print(f"    n_query(total)={bundle.n}, uniir_instr={bundle.instruction!r}")

    if rank == 0:
        with open(qids_json, "w", encoding="utf-8") as f:
            json.dump(bundle.qids, f)
        with open(pos_json, "w", encoding="utf-8") as f:
            json.dump(bundle.qid_to_positives, f)

    idxs = _slice_indices(bundle.n, rank, world, limit)
    print(f"    [rank={rank}] owns {len(idxs)} queries")

    sub_getters = [bundle.query_image_getters[i] for i in idxs]
    # NOTE: bundle.query_texts already has the UniIR task instruction prepended.
    # We *also* pass Qwen3VL's own instruction as system prompt, which is
    # separate from the query text.  This double-instruction mirrors the way
    # UniIR-style benchmarks compare with instruction-aware embedders.
    sub_texts = [bundle.query_texts[i] for i in idxs]

    t0 = time.time()
    feats = encode_qwen3vl(
        embedder=embedder,
        image_getters=sub_getters,
        texts=sub_texts,
        instruction=instruction,
        batch_size=batch_size,
        tag=f"query:{dataset}/{split}:r{rank}/{world}",
    )
    dt = time.time() - t0

    if world == 1:
        torch.save(feats, feat_pt)
        target = feat_pt
    else:
        torch.save({"feats": feats, "indices": torch.tensor(idxs, dtype=torch.long)},
                   shard_pt)
        target = shard_pt

    meta = {"dataset": dataset, "split": split, "rank": rank, "world": world,
            "n_local": len(idxs), "n_total": bundle.n,
            "feat_dim": int(feats.shape[-1]), "dtype": str(feats.dtype),
            "batch_size": batch_size, "time_sec": round(dt, 2),
            "uniir_instruction": bundle.instruction,
            "qwen3vl_instruction": instruction, "limit": limit}
    meta_path = (os.path.join(out_dir, "meta.json") if world == 1
                 else os.path.join(shard_dir, f"shard_{rank:02d}_of_{world:02d}.meta.json"))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"    done: query {dataset}/{split} r{rank} -> {target}  ({dt/60:.2f}min)")
    return meta


def merge_query_shards(dataset: str, split: str, out_root: str, world: int) -> dict:
    out_dir = os.path.join(out_root, "queries", dataset, split)
    shard_dir = os.path.join(out_dir, "shards")
    qids_json = os.path.join(out_dir, "qids.json")
    if not os.path.isfile(qids_json):
        raise FileNotFoundError(f"qids.json missing at {qids_json}")
    with open(qids_json, "r", encoding="utf-8") as f:
        qids = json.load(f)
    n_total = len(qids)

    shards = []
    covered = torch.zeros(n_total, dtype=torch.bool)
    dim = None
    for r in range(world):
        p = os.path.join(shard_dir, f"shard_{r:02d}_of_{world:02d}.pt")
        obj = torch.load(p, map_location="cpu")
        shards.append(obj)
        dim = dim or obj["feats"].shape[-1]
        covered[obj["indices"]] = True
    if int(covered.sum().item()) != n_total:
        raise RuntimeError("query merge: incomplete coverage")

    full = torch.zeros(n_total, dim, dtype=torch.float16)
    for obj in shards:
        full[obj["indices"]] = obj["feats"]
    feat_pt = os.path.join(out_dir, "query_feats.pt")
    torch.save(full, feat_pt)
    meta = {"dataset": dataset, "split": split, "n_query": n_total,
            "feat_dim": int(dim), "dtype": "torch.float16", "world": world}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[merge] query {dataset}/{split} -> {feat_pt} (shape={tuple(full.shape)})")
    return meta


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def build_embedder(model_path: str, dtype: str, attn_impl: str,
                   max_length: int, max_pixels: int) -> Qwen3VLEmbedder:
    torch_dtype = {
        "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32,
    }[dtype]
    kwargs = {"torch_dtype": torch_dtype}
    if attn_impl and attn_impl != "eager":
        kwargs["attn_implementation"] = attn_impl
    print(f"[init] loading Qwen3VLEmbedder from {model_path}", flush=True)
    print(f"       dtype={dtype}, attn={attn_impl}, max_length={max_length}, "
          f"max_pixels={max_pixels}", flush=True)
    t0 = time.time()
    embedder = Qwen3VLEmbedder(
        model_name_or_path=model_path,
        max_length=max_length,
        max_pixels=max_pixels,
        **kwargs,
    )
    print(f"[init] embedder ready in {time.time()-t0:.1f}s  device={embedder.model.device}",
          flush=True)
    return embedder


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--stage", choices=["cand", "query", "merge", "merge_query"],
                   required=True)
    p.add_argument("--pools", nargs="+", default=["SHARED"])
    p.add_argument("--datasets", nargs="+", default=["OVEN", "INFOSEEK"])
    p.add_argument("--splits", nargs="+", default=["test", "test"])
    p.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--attn", type=str, default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--max_length", type=int, default=1024,
                   help="Truncate long candidate texts to this many tokens. "
                        "MBEIR wiki paragraphs mostly fit; 1024 keeps p99 intact.")
    p.add_argument("--max_pixels", type=int, default=1024 * 512,
                   help="Cap image resolution passed to Qwen3VL "
                        "(default ~0.5 MP; big enough for retrieval, saves time).")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only process the first `limit` items of the "
                        "ORIGINAL ordered list (for smoke tests).")
    p.add_argument("--query_instruction", type=str, default=DEFAULT_INSTR_QUERY)
    p.add_argument("--cand_instruction", type=str, default=DEFAULT_INSTR_CAND)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.splits) != len(args.datasets):
        raise ValueError("--splits length must match --datasets length.")

    # Merge stages don't need to load the model.
    if args.stage == "merge":
        for pool in args.pools:
            merge_cand_shards(pool, args.out_root, args.world)
        return
    if args.stage == "merge_query":
        for ds, sp in zip(args.datasets, args.splits):
            merge_query_shards(ds, sp, args.out_root, args.world)
        return

    embedder = build_embedder(
        model_path=args.model_path, dtype=args.dtype, attn_impl=args.attn,
        max_length=args.max_length, max_pixels=args.max_pixels,
    )

    if args.stage == "cand":
        for pool in args.pools:
            extract_cand_shard(
                pool_name=pool, embedder=embedder,
                batch_size=args.batch_size, out_root=args.out_root,
                rank=args.rank, world=args.world,
                instruction=args.cand_instruction,
                limit=args.limit, overwrite=args.overwrite,
            )
    elif args.stage == "query":
        for ds, sp in zip(args.datasets, args.splits):
            extract_query_shard(
                dataset=ds, split=sp, embedder=embedder,
                batch_size=args.batch_size, out_root=args.out_root,
                rank=args.rank, world=args.world,
                instruction=args.query_instruction,
                limit=args.limit, overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
