# -*- coding: utf-8 -*-
"""Extract UniIR CLIP-SF features for M-BEIR task 8 (OVEN + Infoseek).

Two variants are saved for every batch:
  * ``fused``  = L2Norm(L2Norm(v_img) + L2Norm(v_txt))    # for CLIP-SF eval
  * ``img``    = L2Norm(v_img)                            # for GENIUS RQ + reranker
  * ``txt``    = L2Norm(v_txt)                            # for GENIUS RQ + reranker

Layout (under --out_root, e.g. ``outputs/clipsf``):

    cand_pool/SHARED/
        cand_feats.pt      fp16 [Nc, D]  fused  (used by eval_clipsf.py)
        cand_img.pt        fp16 [Nc, D]  L2Norm(v_img)   (used by GENIUS)
        cand_txt.pt        fp16 [Nc, D]  L2Norm(v_txt)   (used by GENIUS)
        cand_ids.json      list[str]                     (aligned with rows)
        meta.json
    queries/<DATASET>/<SPLIT>/
        query_feats.pt     fp16 [Nq, D]  fused
        query_img.pt       fp16 [Nq, D]
        query_txt.pt       fp16 [Nq, D]
        qids.json          list[str]
        qid_to_positives.json  dict[qid -> list[did]]
        meta.json

Multi-GPU data parallel is supported via ``--rank R --world W``: each rank
processes shard ``i s.t. i % W == R`` (row-level striped partitioning),
writes into ``.../shards/shard_R_of_W.pt``, and a final ``--merge`` step
concatenates all shards back into the flat files above.

Usage
-----
  # Stage 1: 8-way SHARED cand pool extraction (one process per GPU)
  for r in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$r python extract_clipsf_feats.py \
          --stage cand --pool SHARED --rank $r --world 8 &
  done; wait
  python extract_clipsf_feats.py --stage cand --pool SHARED --merge --world 8

  # Stage 2: queries (fast, single GPU is enough)
  python extract_clipsf_feats.py --stage query \
      --datasets OVEN INFOSEEK --splits test test
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from typing import Callable, List, Sequence, Tuple

# ---- Thread-count caps ------------------------------------------------------
# When 8 ranks run in parallel each Python process was defaulting to ~45 CPU
# threads (OpenMP / MKL / PyTorch intraop), producing load-average ~750 and
# leaving GPUs at 0% util.  Force a very small per-rank thread budget BEFORE
# importing numpy / torch so their C libraries pick it up at init time.
_THREAD_BUDGET = os.environ.get("CLIPSF_THREAD_BUDGET", "4")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, _THREAD_BUDGET)

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Belt-and-suspenders: PyTorch intraop / interop caps (env vars alone don't
# always take effect after ``import torch`` in some builds).
try:
    torch.set_num_threads(int(_THREAD_BUDGET))
    torch.set_num_interop_threads(1)
except RuntimeError:
    # set_num_interop_threads is a one-shot; ignore if already set.
    pass

# ---- Default M-BEIR tar-based random-access image reader --------------------
# Must be set BEFORE ``mbeir_data`` is imported (its module-level init caches
# the reader based on these env vars).  We reuse the shared 174GB tar and its
# offset index built during the CLIP_baseline experiments.
os.environ.setdefault(
    "MBEIR_TAR_PATH",
    "xxx/mbeir_images.tar",
)
os.environ.setdefault(
    "MBEIR_TAR_INDEX",
    "xxx/mbeir_images_oven_index.pkl",
)

from clip_sf_model import load_clip_sf
from mbeir_data import (
    MbeirCandBundle,
    MbeirQueryBundle,
    build_shared_cand_bundle,
    load_cand_bundle,
    load_query_bundle,
)


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = "/dev/shm/genius_ckpt/clip_sf_large.pth"
DEFAULT_CLIP_DOWNLOAD_ROOT = "/dev/shm/genius_ckpt/clip_dl_root"
DEFAULT_OUT_ROOT = os.path.join(HERE, "outputs", "clipsf")


# ---------------------------------------------------------------------------
# Ensure openai-clip's ViT-L-14.pt is available in the local download root
# (we pre-copied it to /dev/shm; this function just guards + copies if missing)
# ---------------------------------------------------------------------------
def _ensure_clip_download_root(download_root: str, src_pt: str) -> None:
    os.makedirs(download_root, exist_ok=True)
    dst = os.path.join(download_root, "ViT-L-14.pt")
    if not os.path.exists(dst):
        if not os.path.exists(src_pt):
            raise FileNotFoundError(f"CLIP ViT-L/14 backbone not found at {src_pt}")
        shutil.copy(src_pt, dst)


# ---------------------------------------------------------------------------
# Streaming (image, text) dataset (identical spirit to CLIP_baseline)
# ---------------------------------------------------------------------------
class _ImageTextDataset(Dataset):
    def __init__(
        self,
        image_getters: Sequence[Callable[[], Image.Image]],
        texts: Sequence[str],
        indices: Sequence[int],
    ) -> None:
        assert len(image_getters) == len(texts)
        self._getters = image_getters
        self._texts = list(texts)
        self._indices = list(indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx):
        real_i = self._indices[idx]
        img = self._getters[real_i]()
        txt = self._texts[real_i]
        return real_i, img, txt


def _collate(batch):
    idxs = [b[0] for b in batch]
    imgs = [b[1] for b in batch]
    txts = [b[2] for b in batch]
    return idxs, imgs, txts


# ---------------------------------------------------------------------------
# Core encoding routine (returns 3 tensors: img/txt/fused, all L2-normalised)
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_all(
    clip_model,
    preprocess,
    tokenize,
    image_getters: Sequence[Callable[[], Image.Image]],
    texts: Sequence[str],
    indices: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    num_workers: int,
    tag: str,
    log_every: int = 20,
    save_raw: bool = False,
):
    dataset = _ImageTextDataset(image_getters, texts, indices)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=False,
    )
    n = len(dataset)
    all_img: List[torch.Tensor] = []
    all_txt: List[torch.Tensor] = []
    all_fused: List[torch.Tensor] = []
    all_img_raw: List[torch.Tensor] = []
    all_txt_raw: List[torch.Tensor] = []
    all_idx: List[int] = []
    t0 = time.time()
    seen = 0
    for step, (idx_batch, imgs, txts) in enumerate(loader):
        # Image branch --------------------------------------------------
        pixel = torch.stack([preprocess(img) for img in imgs], dim=0)
        pixel = pixel.to(device=device, dtype=dtype)
        v_img = clip_model.encode_image(pixel).float()

        # Text branch ---------------------------------------------------
        tok = tokenize(list(txts), context_length=77, truncate=True).to(device)
        v_txt = clip_model.encode_text(tok).float()

        # Normalise + fuse ---------------------------------------------
        v_img_n = torch.nn.functional.normalize(v_img, dim=-1)
        v_txt_n = torch.nn.functional.normalize(v_txt, dim=-1)
        fused   = torch.nn.functional.normalize(v_img_n + v_txt_n, dim=-1)

        all_img.append(v_img_n.half().cpu())
        all_txt.append(v_txt_n.half().cpu())
        all_fused.append(fused.half().cpu())
        if save_raw:
            all_img_raw.append(v_img.half().cpu())
            all_txt_raw.append(v_txt.half().cpu())
        all_idx.extend(idx_batch)

        seen += len(txts)
        if step % log_every == 0 or seen == n:
            dt = time.time() - t0
            rate = seen / max(dt, 1e-9)
            print(f"  [{tag}] {seen}/{n} ({rate:.1f}/s, elapsed {dt:.1f}s)", flush=True)

    out = {
        "img":   torch.cat(all_img,   dim=0),
        "txt":   torch.cat(all_txt,   dim=0),
        "fused": torch.cat(all_fused, dim=0),
        "idx":   all_idx,
    }
    if save_raw:
        out["img_raw"] = torch.cat(all_img_raw, dim=0)
        out["txt_raw"] = torch.cat(all_txt_raw, dim=0)
    return out


# ---------------------------------------------------------------------------
# Cand pool stage
# ---------------------------------------------------------------------------
def _build_bundle_cand(pool_name: str) -> MbeirCandBundle:
    if pool_name == "SHARED":
        return build_shared_cand_bundle()
    return load_cand_bundle(pool_name)


def run_cand(args, clip_model, preprocess, tokenize, device, dtype):
    bundle = _build_bundle_cand(args.pool)
    n = bundle.n
    print(f"[cand:{args.pool}] n_cand={n}")

    out_dir = os.path.join(args.out_root, "cand_pool", args.pool)
    shards_dir = os.path.join(out_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    # Row-striped partitioning: rank r processes indices [r, r+W, r+2W, ...]
    indices = list(range(args.rank, n, args.world))
    print(f"[cand:{args.pool}] rank={args.rank}/{args.world}, my share={len(indices)}")

    shard_file = os.path.join(shards_dir, f"shard_{args.rank:02d}_of_{args.world:02d}.pt")
    meta_file = os.path.join(shards_dir, f"shard_{args.rank:02d}_of_{args.world:02d}.meta.json")
    if (not args.overwrite) and os.path.isfile(shard_file):
        print(f"[skip] shard already exists: {shard_file}")
        return
    # cand_ids.json is written by rank 0 (holds the full ordered list)
    if args.rank == 0:
        with open(os.path.join(out_dir, "cand_ids.json"), "w", encoding="utf-8") as f:
            json.dump(bundle.dids, f)

    t0 = time.time()
    out = encode_all(
        clip_model, preprocess, tokenize,
        image_getters=bundle.cand_image_getters,
        texts=bundle.cand_texts,
        indices=indices,
        device=device, dtype=dtype,
        batch_size=args.batch_size, num_workers=args.num_workers,
        tag=f"cand:{args.pool}:r{args.rank}",
        save_raw=args.save_raw,
    )
    img_t, txt_t, fused_t, idx_t = out["img"], out["txt"], out["fused"], out["idx"]
    dt = time.time() - t0

    payload = {
        "img": img_t, "txt": txt_t, "fused": fused_t,
        "indices": torch.tensor(idx_t, dtype=torch.long),
    }
    if args.save_raw:
        payload["img_raw"] = out["img_raw"]
        payload["txt_raw"] = out["txt_raw"]
    torch.save(payload, shard_file)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({
            "pool": args.pool, "n_shard": len(idx_t), "n_total": n,
            "rank": args.rank, "world": args.world,
            "time_sec": round(dt, 2), "feat_dim": int(img_t.shape[-1]),
        }, f, indent=2)
    print(f"[cand:{args.pool}:r{args.rank}] done in {dt:.1f}s -> {shard_file}")


def merge_cand(args):
    out_dir = os.path.join(args.out_root, "cand_pool", args.pool)
    shards_dir = os.path.join(out_dir, "shards")

    with open(os.path.join(out_dir, "cand_ids.json"), "r", encoding="utf-8") as f:
        dids = json.load(f)
    n_total = len(dids)

    # Determine feature dim from any shard
    world = args.world
    first_shard = torch.load(os.path.join(shards_dir, f"shard_00_of_{world:02d}.pt"),
                              map_location="cpu", weights_only=False)
    D = int(first_shard["img"].shape[-1])
    has_raw = ("img_raw" in first_shard) and ("txt_raw" in first_shard)
    print(f"[merge:{args.pool}] world={world}, n_total={n_total}, dim={D}, has_raw={has_raw}")

    img_full   = torch.zeros(n_total, D, dtype=torch.float16)
    txt_full   = torch.zeros(n_total, D, dtype=torch.float16)
    fused_full = torch.zeros(n_total, D, dtype=torch.float16)
    if has_raw:
        img_raw_full = torch.zeros(n_total, D, dtype=torch.float16)
        txt_raw_full = torch.zeros(n_total, D, dtype=torch.float16)

    filled = torch.zeros(n_total, dtype=torch.bool)
    for r in range(world):
        sf = os.path.join(shards_dir, f"shard_{r:02d}_of_{world:02d}.pt")
        d = torch.load(sf, map_location="cpu", weights_only=False)
        idx = d["indices"].long()
        img_full[idx]   = d["img"]
        txt_full[idx]   = d["txt"]
        fused_full[idx] = d["fused"]
        if has_raw:
            img_raw_full[idx] = d["img_raw"]
            txt_raw_full[idx] = d["txt_raw"]
        filled[idx] = True
        print(f"  merged {sf}: {idx.numel()} rows")

    if not filled.all():
        n_missing = int((~filled).sum())
        raise RuntimeError(f"[merge:{args.pool}] {n_missing} rows never filled")

    torch.save(fused_full, os.path.join(out_dir, "cand_feats.pt"))
    torch.save(img_full,   os.path.join(out_dir, "cand_img.pt"))
    torch.save(txt_full,   os.path.join(out_dir, "cand_txt.pt"))
    if has_raw:
        torch.save(img_raw_full, os.path.join(out_dir, "cand_img_raw.pt"))
        torch.save(txt_raw_full, os.path.join(out_dir, "cand_txt_raw.pt"))
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "pool": args.pool, "n_cand": n_total, "feat_dim": D,
            "dtype": "float16",
            "fusion": "score_fusion (L2Norm(L2Norm(v_img)+L2Norm(v_txt)))",
            "encoder": "UniIR CLIP-SF ViT-L/14",
            "world": world,
        }, f, indent=2)
    print(f"[merge:{args.pool}] -> {out_dir}/cand_{{feats,img,txt}}.pt")


# ---------------------------------------------------------------------------
# Query stage
# ---------------------------------------------------------------------------
def run_query(args, clip_model, preprocess, tokenize, device, dtype):
    for ds, sp in zip(args.datasets, args.splits):
        out_dir = os.path.join(args.out_root, "queries", ds, sp)
        os.makedirs(out_dir, exist_ok=True)
        feat_pt = os.path.join(out_dir, "query_feats.pt")
        if (not args.overwrite) and os.path.isfile(feat_pt):
            print(f"[skip] query {ds}/{sp}")
            continue

        bundle: MbeirQueryBundle = load_query_bundle(ds, sp)
        print(f"[query:{ds}/{sp}] n_query={bundle.n}, instruction={bundle.instruction!r}")

        with open(os.path.join(out_dir, "qids.json"), "w", encoding="utf-8") as f:
            json.dump(bundle.qids, f)
        with open(os.path.join(out_dir, "qid_to_positives.json"), "w", encoding="utf-8") as f:
            json.dump(bundle.qid_to_positives, f)

        t0 = time.time()
        out = encode_all(
            clip_model, preprocess, tokenize,
            image_getters=bundle.query_image_getters,
            texts=bundle.query_texts,
            indices=list(range(bundle.n)),
            device=device, dtype=dtype,
            batch_size=args.batch_size, num_workers=args.num_workers,
            tag=f"query:{ds}/{sp}",
            save_raw=args.save_raw,
        )
        img_t, txt_t, fused_t = out["img"], out["txt"], out["fused"]
        dt = time.time() - t0

        torch.save(fused_t, feat_pt)
        torch.save(img_t,   os.path.join(out_dir, "query_img.pt"))
        torch.save(txt_t,   os.path.join(out_dir, "query_txt.pt"))
        if args.save_raw:
            torch.save(out["img_raw"], os.path.join(out_dir, "query_img_raw.pt"))
            torch.save(out["txt_raw"], os.path.join(out_dir, "query_txt_raw.pt"))
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "dataset": ds, "split": sp, "n_query": bundle.n,
                "feat_dim": int(fused_t.shape[-1]), "dtype": "float16",
                "instruction": bundle.instruction,
                "fusion": "score_fusion (L2Norm(L2Norm(v_img)+L2Norm(v_txt)))",
                "encoder": "UniIR CLIP-SF ViT-L/14",
                "time_sec": round(dt, 2),
            }, f, indent=2)
        print(f"[query:{ds}/{sp}] done in {dt:.1f}s -> {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["cand", "query"], required=True)
    p.add_argument("--pool", type=str, default="SHARED",
                   help="Which cand pool (SHARED | OVEN | INFOSEEK).")
    p.add_argument("--datasets", nargs="+", default=["OVEN", "INFOSEEK"])
    p.add_argument("--splits", nargs="+", default=["test", "test"])
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--merge", action="store_true", help="Only merge shards.")

    p.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    p.add_argument("--clip_download_root", type=str, default=DEFAULT_CLIP_DOWNLOAD_ROOT)
    p.add_argument("--vitl14_src", type=str,
                   default="/dev/shm/genius_ckpt/ViT-L-14.pt")

    p.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_raw", action="store_true",
                   help="Also save un-normalised v_img / v_txt (needed for GENIUS RQ input).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.splits) != len(args.datasets):
        raise ValueError("--splits length must match --datasets length.")

    # Merge-only path (no GPU / model needed)
    if args.stage == "cand" and args.merge:
        merge_cand(args)
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if (args.dtype == "fp16" and device.type == "cuda") else torch.float32

    _ensure_clip_download_root(args.clip_download_root, args.vitl14_src)
    print(f"[init] loading CLIP-SF from {args.ckpt} (dtype={dtype}, device={device})")
    clip_model, preprocess, tokenize = load_clip_sf(
        ckpt_path=args.ckpt,
        base_model="ViT-L/14",
        device=device,
        download_root=args.clip_download_root,
        fp16=(dtype == torch.float16),
    )
    clip_model.eval()

    if args.stage == "cand":
        run_cand(args, clip_model, preprocess, tokenize, device, dtype)
    elif args.stage == "query":
        run_query(args, clip_model, preprocess, tokenize, device, dtype)


if __name__ == "__main__":
    main()
