# -*- coding: utf-8 -*-
"""Extract WeMM-Embedding-9B features for M-BEIR task 8 (OVEN + Infoseek).

WeMM-Embedding-9B is a single-tower multimodal embedding model built on top of
Qwen3.5-9B.  It accepts interleaved multimodal inputs (image + text) via a
chat-style message list, tokenises through `AutoProcessor`, and produces a
4096-dim L2-normalised embedding by pooling the ``<embedding>`` token position
from the last hidden state (see ``modeling_wemm_embedding.WeMMEmbedding``).

Pipeline
--------
* 8-GPU data-parallel encoding: ``--rank R --world W`` partitions the index
  space, each worker owns ``list(indices)[R::W]``.
* Shard outputs -> ``outputs/cand_pool/<POOL>/shards/shard_{R:02d}_of_{W:02d}.pt``
  with ``{"feats": [Nr, D] fp16, "indices": [Nr] int64}``.
* ``--stage merge`` gathers shards, reassembles the [N, D] tensor in original
  order, and dumps ``cand_feats.pt`` + ``cand_ids.json`` (same layout as
  Qwen3VL / CLIP-SF baselines, so ``eval_wemm.py`` reuses the evaluator).
* Queries are small (~32K), single-card is enough.

Usage
-----
  # smoke test (single GPU, tiny slice)
  python3 extract_wemm_feats.py --stage cand --pools SHARED --limit 20 \
        --rank 0 --world 1

  # 8-GPU parallel encoding
  for R in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$R python3 extract_wemm_feats.py \
          --stage cand --pools SHARED --rank $R --world 8 &
  done
  wait
  python3 extract_wemm_feats.py --stage merge --pools SHARED

  # queries (single card is enough)
  CUDA_VISIBLE_DEVICES=0 python3 extract_wemm_feats.py --stage query \
      --datasets OVEN INFOSEEK --splits test test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from PIL import Image

# WeMM depends on transformers==5.2.0 and qwen_vl_utils.  These live in the
# llama_factory conda env; the launcher script picks the right interpreter.
from transformers import AutoModel, AutoProcessor
from qwen_vl_utils import process_vision_info

from mbeir_data import (
    MbeirCandBundle,
    MbeirQueryBundle,
    build_shared_cand_bundle,
    load_cand_bundle,
    load_query_bundle,
)


HERE = os.path.dirname(os.path.abspath(__file__))
GM2M_ROOT = os.path.dirname(HERE)
WORKSPACE = os.path.dirname(GM2M_ROOT)

DEFAULT_MODEL = os.path.join(WORKSPACE, "baseline", "WeMM-Embedding-9B")
DEFAULT_OUT_ROOT = os.path.join(HERE, "outputs")

# Retrieval-style prompts.  WeMM's chat template lets us inject a system
# message; we prepend the "instruction" as an extra text item in the user
# content so the tokenizer serialises it via the same chat template used at
# training time.  UniIR's own task-8 instruction is folded into ``query_texts``
# already (see mbeir_data.py), so the fields below are the *outer* prompts,
# consistent with WeMM's official ``encode_query`` / ``encode_document`` split.
DEFAULT_INSTR_QUERY = "Retrieve images or text relevant to the user's query."
DEFAULT_INSTR_CAND = "Represent the document for retrieval."


# ---------------------------------------------------------------------------
# WeMM embedder wrapper
# ---------------------------------------------------------------------------
class WeMMEmbedder:
    """Thin wrapper around ``AutoModel(trust_remote_code=True).embedding()``.

    Uses the exact code path documented in the model card's "Transformers"
    section.  We keep image and text in a single user message; when image is
    missing (encode failure / robustness fallback) we send text-only.
    """

    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_impl: str = "flash_attention_2",
        image_patch_size: int = 16,
    ) -> None:
        print(f"[init] loading WeMM processor from {model_path}", flush=True)
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        print(f"[init] processor ready ({time.time()-t0:.1f}s)", flush=True)
        print(f"[init] loading WeMM model (dtype={torch_dtype}, attn={attn_impl}) ...",
              flush=True)
        t1 = time.time()
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": torch_dtype,
        }
        if attn_impl and attn_impl != "eager":
            model_kwargs["attn_implementation"] = attn_impl
        self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
        self.model = self.model.to("cuda").eval()
        # The vision tower uses a fixed patch size; expose it for
        # process_vision_info.  Fall back to 16 for WeMM-9B.
        vc = getattr(self.model.config, "vision_config", None)
        self.image_patch_size = int(getattr(vc, "patch_size", image_patch_size))
        self.device = next(self.model.parameters()).device
        print(f"[init] model ready in {time.time()-t1:.1f}s on {self.device}, "
              f"image_patch_size={self.image_patch_size}", flush=True)

    # ---- one input dict, chat-style ---------------------------------------
    @staticmethod
    def _make_message(instruction: str, text: str, image: Optional[Image.Image]) -> List[Dict[str, Any]]:
        """Build a single-turn user message.

        Order: (instruction as text) -> (image) -> (query/doc text).
        A missing image falls back to text-only.
        """
        content: List[Dict[str, Any]] = []
        # The chat template renders text items with a leading newline (except
        # the first).  Keeping instruction as its own text item is closest to
        # WeMM's own ``encode_query`` prompt-injection path.
        content.append({"type": "text", "text": instruction})
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": text if text else ""})
        return [{"role": "user", "content": content}]

    # ---- batched embed ----------------------------------------------------
    @torch.no_grad()
    def embed_batch(
        self,
        instruction: str,
        texts: Sequence[str],
        image_getters: Sequence[Callable[[], Image.Image]],
    ) -> torch.Tensor:
        """Return [B, D] fp16 L2-normalised embeddings on CPU.

        We process one sample at a time to sidestep processor padding issues
        with heterogeneous image resolutions (WeMM/Qwen3VL images have variable
        token counts, and 5.x transformers' processor batches them by simply
        concatenating pixel_values with a per-sample grid_thw tensor).  The
        model itself is amortised (loaded once), so per-sample forwards are
        acceptable for a 9B encoder.
        """
        outs: List[torch.Tensor] = []
        for txt, get_img in zip(texts, image_getters):
            img: Optional[Image.Image] = None
            try:
                img = get_img()
            except Exception as e:
                # Robust fallback: text-only.
                print(f"[warn] image load failed ({e}); using text-only.", flush=True)
                img = None

            messages = self._make_message(instruction, txt, img)
            chat_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            # process_vision_info reads the PIL image from the messages and
            # returns a list of images / videos.  We always have image OR None.
            images_list, videos_list, video_kwargs = process_vision_info(
                messages,
                image_patch_size=self.image_patch_size,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if videos_list is not None:
                # M-BEIR task 8 has no video; guard defensively.
                videos_list, video_metadata = zip(*videos_list)
                videos_list = list(videos_list)
                video_kwargs = {**video_kwargs, "video_metadata": list(video_metadata)}
            else:
                videos_list = None

            inputs = self.processor(
                text=chat_text,
                images=images_list,
                videos=videos_list,
                return_tensors="pt",
                **video_kwargs,
            )
            inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                      for k, v in inputs.items()}

            with torch.inference_mode():
                emb = self.model.embedding(**inputs)   # [1, D]

            outs.append(emb.detach().to(dtype=torch.float16, device="cpu"))

        return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# Encoding driver (shared with Qwen3VL layout)
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_wemm(
    embedder: WeMMEmbedder,
    image_getters: Sequence[Callable[[], Image.Image]],
    texts: Sequence[str],
    instruction: str,
    batch_size: int,
    tag: str,
    log_every: int = 5,
) -> torch.Tensor:
    """Encode (image, text) pairs into [N, D] fp16 L2-normalised features."""
    assert len(image_getters) == len(texts)
    n = len(texts)
    all_feats: List[torch.Tensor] = []
    t0 = time.time()
    print(f"  [{tag}] starting encode: n={n}, batch_size={batch_size}", flush=True)
    seen = 0
    step = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        feats = embedder.embed_batch(
            instruction=instruction,
            texts=[texts[i] for i in range(start, end)],
            image_getters=[image_getters[i] for i in range(start, end)],
        )
        all_feats.append(feats)
        seen = end
        step += 1
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
    total = n if limit <= 0 else min(n, limit)
    return list(range(rank, total, world))


# ---------------------------------------------------------------------------
# Stage: cand (per-rank shard)
# ---------------------------------------------------------------------------
def extract_cand_shard(
    pool_name: str,
    embedder: WeMMEmbedder,
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
    ids_all_json = os.path.join(out_dir, "cand_ids.json")
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
    feats = encode_wemm(
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
            raise RuntimeError(f"dim mismatch at shard r{r}")
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
# Stage: query
# ---------------------------------------------------------------------------
def extract_query_shard(
    dataset: str,
    split: str,
    embedder: WeMMEmbedder,
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
    sub_texts = [bundle.query_texts[i] for i in idxs]

    t0 = time.time()
    feats = encode_wemm(
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
            "wemm_instruction": instruction, "limit": limit}
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
# CLI
# ---------------------------------------------------------------------------
def build_embedder(model_path: str, dtype: str, attn_impl: str) -> WeMMEmbedder:
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                   "fp32": torch.float32}[dtype]
    return WeMMEmbedder(model_path=model_path, torch_dtype=torch_dtype,
                        attn_impl=attn_impl)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--stage", choices=["cand", "query", "merge", "merge_query"],
                   required=True)
    p.add_argument("--pools", nargs="+", default=["SHARED"])
    p.add_argument("--datasets", nargs="+", default=["OVEN", "INFOSEEK"])
    p.add_argument("--splits", nargs="+", default=["test", "test"])
    p.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Per-forward samples; kept at 1 to avoid processor "
                        "padding pitfalls with heterogeneous image resolutions.")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--attn", type=str, default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--query_instruction", type=str, default=DEFAULT_INSTR_QUERY)
    p.add_argument("--cand_instruction", type=str, default=DEFAULT_INSTR_CAND)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.splits) != len(args.datasets):
        raise ValueError("--splits length must match --datasets length.")

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
