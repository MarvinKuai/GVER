# -*- coding: utf-8 -*-
"""GENIUS_R retrieval pipeline for M-BEIR task 8 (OVEN / Infoseek).

This script consumes pre-extracted CLIP-SF features (produced by
``extract_clipsf_feats.py --save_raw``) and executes the *inference-only*
GENIUS_R path:

  Stage A  quantize cand pool:
             img_raw + txt_raw + mask -> Combiner -> RQ -> code[9]
             also compute rerank feat = L2Norm(img_n + txt_n)  (already saved
             as cand_feats.pt in the CLIP_SF run -- we reuse it directly).

  Stage B  build code->did index and Trie for constrained decoding.

  Stage C  quantize queries (same Combiner + RQ) to get query encode features.

  Stage D  for each query, run T5 constrained beam search (num_beams) over
             the Trie to obtain up to ``num_beams`` candidate code sequences.
             Look each code up in the code->did index.  If a bucket is bigger
             than ``max_per_bucket``, keep the whole bucket (they will be
             reranked in Stage E of ``eval_genius_r.py``).

Outputs (under --out_root, default ``outputs/genius_r``):
    cand_pool/SHARED/
        cand_codes.pt        int32 [Nc, 9]     RQ code per candidate
        cand_ids.json        list[str]         aligned with rows
        code_to_rows.pkl     dict[tuple[int]*9 -> list[int]]  bucket index
        meta.json
    queries/<DS>/<SPLIT>/
        query_codes_topN.pt  int32 [Nq, num_beams, 9]  T5 beam outputs
        query_encode.pt      fp16  [Nq, 768]            Combiner+normalize (unused)
        qids.json            list[str]
        qid_to_positives.json  copied from CLIP_SF stage
        meta.json

Downstream ``eval_genius_r.py`` does the CLIP-embedding rerank + metrics.

Prerequisites
-------------
Run these first (they're chained in ``run_baseline.sh`` when ``STAGES=genius_r``):
  1. ``extract_clipsf_feats.py --stage cand --pool SHARED --save_raw``  x8
  2. ``extract_clipsf_feats.py --stage cand --pool SHARED --merge``
  3. ``extract_clipsf_feats.py --stage query --datasets OVEN INFOSEEK \
         --splits test test --save_raw``  (single GPU)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from typing import Dict, List, Sequence, Tuple

# ---- Thread caps (single-GPU pipeline but T5 beam search is CPU-heavy) ------
_THREAD_BUDGET = os.environ.get("GENIUSR_THREAD_BUDGET", "8")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, _THREAD_BUDGET)

import numpy as np
import torch
import torch.nn.functional as F

try:
    torch.set_num_threads(int(_THREAD_BUDGET))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# ---- PyTorch 2.6+ default ``weights_only=True`` chokes on the GENIUS ckpt
# because it stashes an ``omegaconf.DictConfig`` alongside the tensors.  Add
# it to the safe-globals allowlist so the internal ``torch.load`` calls
# inside ``retriever.py`` succeed without patching that file.
try:
    from omegaconf.dictconfig import DictConfig as _OCDictConfig
    from omegaconf.listconfig import ListConfig as _OCListConfig
    from omegaconf.base import ContainerMetadata as _OCContainerMetadata
    torch.serialization.add_safe_globals([
        _OCDictConfig, _OCListConfig, _OCContainerMetadata,
    ])
except Exception:  # noqa: BLE001
    pass

# Belt-and-suspenders: some code paths in GENIUS still call ``torch.load``
# without ``weights_only``, and even with the allowlist above, older
# transformers/omegaconf combos throw on nested class instances.  Monkey-patch
# ``torch.load`` to default to ``weights_only=False`` for the rest of this
# script's lifetime.  All ckpts we load here come from a trusted source
# (bundled with the GENIUS repository).
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load  # type: ignore[assignment]

# ---- Bring GENIUS-CVPR25 src on PYTHONPATH ---------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
GENIUS_REPO = os.environ.get(
    "GENIUS_REPO",
    "xxx/GENIUS-CVPR25",   # fill in the path to the original GENIUS-CVPR25 source tree
)
GENIUS_SRC = os.path.join(GENIUS_REPO, "src")
if GENIUS_SRC not in sys.path:
    sys.path.insert(0, GENIUS_SRC)

# GENIUS's RQ / retriever imports pull ``models.uniir_clip.utils`` which
# expects torch.distributed to be initialised for ``get_rank`` /
# ``get_world_size``.  In this single-GPU inference script we don't care, so
# monkey-patch them to safe stubs BEFORE importing the retriever.
def _patch_dist_utils() -> None:
    import models.uniir_clip.utils as _u
    _u.get_rank = lambda: 0
    _u.get_world_size = lambda: 1
    class _NoopGL(torch.autograd.Function):  # not used at inference
        @staticmethod
        def forward(ctx, x): return (x,)
        @staticmethod
        def backward(ctx, *grads): return grads
    _u.GatherLayer = _NoopGL

_patch_dist_utils()

from models.residual_quantization.residual_quantization import RQ  # noqa: E402
# Import ``retriever`` lazily inside main() because it also imports the C++
# trie extension (``trie_cpp``), which may not be built for our env.


# ---- Config shim ------------------------------------------------------------
# Both RQ and the retriever expect an OmegaConf-like config object.  We build
# a minimal ``types.SimpleNamespace`` tree that satisfies the accessed keys.
from types import SimpleNamespace


def _build_genius_config() -> SimpleNamespace:
    codebook_cfg = SimpleNamespace(
        codebook_level=8,   # +1 for modality index => 9-token codes
        codebook_vocab=4096,
        quantizer_path=os.path.join("checkpoint", "rq_clip_large.pth"),
    )
    ckpt_cfg = SimpleNamespace(
        ckpt_dir=os.path.join(HERE, "outputs", "genius_r", "trie"),
        ckpt_name="",
    )
    model_cfg = SimpleNamespace(
        ckpt_config=ckpt_cfg,
        trie_type="triecpp",
    )
    hyper_cfg = SimpleNamespace(alpha=2)
    cfg = SimpleNamespace(
        codebook_config=codebook_cfg,
        model=model_cfg,
        hyperparameter_config=hyper_cfg,
        genir_dir=GENIUS_REPO,
    )
    return cfg


# ---- Loading helpers --------------------------------------------------------
def load_rq(cfg: SimpleNamespace, device: torch.device) -> RQ:
    rq = RQ(config=cfg, feature_dim=768, clip_model=None,
            unique_code=False, modality_index=True)
    ckpt_path = os.path.join(GENIUS_REPO, "checkpoint", "rq_clip_large.pth")
    print(f"[rq] loading {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    missing, unexpected = rq.load_state_dict(sd, strict=False)
    if missing:
        print(f"  missing ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  unexpected ({len(unexpected)}): {unexpected[:5]}")
    rq.eval().to(device)
    return rq


def load_generator(cfg: SimpleNamespace, device: torch.device):
    """Load the T5-based generative retriever, but reuse an *already loaded*
    RQ so we don't double-instantiate the quantizer."""
    # Late import (see comment above).
    from models.generative_retriever.retriever import T5ForGenerativeRetrieval

    from transformers import T5Tokenizer
    tok_dir = os.path.join(GENIUS_REPO, "checkpoint", "t5-small")
    tokenizer = T5Tokenizer.from_pretrained(tok_dir)

    # The retriever ctor rebuilds an RQ + T5 stack from scratch and loads
    # the RQ ckpt itself.  It also *creates* a new tokenizer object.  We
    # rely on that so token-id assignments match training.
    #
    # NB: retriever requires ``clip_model`` only for ``get_img_preprocess_fn``
    # inside its multimodal forward path; since we feed pre-extracted
    # embeddings straight into ``quantizer.inference`` + ``embed_projector``
    # we can safely pass a stub with a dummy ``.eval()`` and no parameters.
    class _StubClip(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._dummy = torch.nn.Parameter(torch.zeros(1))
        def eval(self):
            super().eval()
            return self
        def get_img_preprocess_fn(self):
            raise NotImplementedError
        def get_tokenizer(self):
            raise NotImplementedError

    stub = _StubClip()
    model = T5ForGenerativeRetrieval(
        config=cfg, tokenizer=tokenizer, clip_model=stub,
        new_tokenizer=True, init_rq_codebook=True,
    )
    ckpt_path = os.path.join(GENIUS_REPO, "checkpoint", "GENIUS_t5small.pth")
    print(f"[gen] loading {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    # The training checkpoint stashes both the quantizer (prefix ``quantizer.``)
    # and the generator; loading strict=False keeps both.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Filter noise: the stub CLIP has no params so ``clip_model.*`` is missing.
    missing = [k for k in missing if not k.startswith("clip_model.")]
    if missing:
        print(f"  missing ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  unexpected ({len(unexpected)}): {unexpected[:5]}")

    # Materialise fp16 on GPU
    model.eval().to(device)
    return model, tokenizer


# ---- Stage A: quantize cand pool -------------------------------------------
def quantize_pool(rq: RQ, img_raw: torch.Tensor, txt_raw: torch.Tensor,
                  device: torch.device, batch_size: int = 4096) -> torch.Tensor:
    """Return ``codes`` int32 [N, codebook_level] on CPU."""
    n = img_raw.shape[0]
    all_codes: List[torch.Tensor] = []
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            v_img = img_raw[s:e].to(device=device, dtype=torch.float32,
                                     non_blocking=True)
            v_txt = txt_raw[s:e].to(device=device, dtype=torch.float32,
                                     non_blocking=True)
            # Task 8 always has both modalities: masks == 1.
            m = torch.ones(e - s, 1, device=device, dtype=torch.float32)
            out = rq.inference(v_img, v_txt, m, m)
            codes = out["code"]                                 # [bs, L]
            all_codes.append(codes.detach().cpu().to(torch.int32))
            if (s // batch_size) % 20 == 0:
                dt = time.time() - t0
                rate = e / max(dt, 1e-9)
                print(f"  [quantize] {e}/{n} ({rate:.0f}/s, {dt:.1f}s)",
                      flush=True)
    codes = torch.cat(all_codes, dim=0)
    return codes


def build_code_to_rows(codes: torch.Tensor) -> Dict[Tuple[int, ...], List[int]]:
    """Build inverted index: code tuple -> list of row indices (docs)."""
    idx: Dict[Tuple[int, ...], List[int]] = {}
    codes_list = codes.tolist()
    for i, c in enumerate(codes_list):
        key = tuple(c)
        idx.setdefault(key, []).append(i)
    return idx


# ---- Stage D: constrained beam search --------------------------------------
def _make_prefix_fn(model):
    """Constrained-decoding prefix function.

    Skips the initial ``decoder_start_token_id`` (== pad_token_id in T5),
    queries the C++ trie, and *falls back to EOS* when the beam wanders into
    a code prefix not present in the candidate pool.  Returning an empty
    list here would make transformers >= 4.30 raise ``ValueError`` and abort
    the whole batch, so we send that beam to EOS instead - it will still be
    ranked but its score will be dominated by the ``-inf`` mass at other
    positions and it therefore drops out of the top-N naturally.
    """
    pad_id = model.tokenizer.pad_token_id
    eos_id = model.tokenizer.eos_token_id
    trie = model.trie_index

    def _fn(batch_id, input_ids):
        prefix = input_ids.tolist()
        if prefix and prefix[0] == pad_id:
            prefix = prefix[1:]
        valid = trie.get(prefix)
        if not valid:
            return [eos_id]
        return valid
    return _fn


def beam_search_queries(model, q_img_raw: torch.Tensor, q_txt_raw: torch.Tensor,
                        device: torch.device, num_beams: int,
                        batch_size: int, tag: str) -> torch.Tensor:
    """Return ``code_topN`` int32 [Nq, num_beams, codebook_level] on CPU."""
    n = q_img_raw.shape[0]
    L = model.codebook_level
    out = torch.zeros(n, num_beams, L, dtype=torch.int32)
    pad_id = model.tokenizer.pad_token_id
    prefix_fn = _make_prefix_fn(model)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            v_img = q_img_raw[s:e].to(device=device, dtype=torch.float32,
                                       non_blocking=True)
            v_txt = q_txt_raw[s:e].to(device=device, dtype=torch.float32,
                                       non_blocking=True)
            m = torch.ones(e - s, 1, device=device, dtype=torch.float32)
            # Step 1: quantizer -> encode feature (used as T5 prefix)
            q_out = model.quantizer.inference(v_img, v_txt, m, m)
            emb = F.normalize(q_out["encode"], dim=-1)
            bs = emb.size(0)
            # Step 2: project to T5 prefix embeddings, then beam-search.
            #         We call the underlying ``id_generator.generate`` directly
            #         so we can inject our EOS-fallback prefix_allowed_tokens_fn.
            inputs_embeds = model.embed_projector(emb).reshape(
                bs, model.num_prefix, -1
            )
            gen = model.id_generator.generate(
                inputs_embeds=inputs_embeds,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                prefix_allowed_tokens_fn=prefix_fn,
                use_cache=True,
                max_new_tokens=model.codebook_level,
                early_stopping=True,
            )
            # gen shape: [bs*num_beams, L+1] (prepended pad token, then codes)
            gen_ids = gen.reshape(bs, num_beams, -1)
            for i in range(bs):
                for b in range(num_beams):
                    seq = gen_ids[i, b].tolist()
                    # Strip leading pad and any EOS-fallback tail.
                    if seq and seq[0] == pad_id:
                        seq = seq[1:]
                    row = model.tokenizer.decode(
                        seq, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    code = model.detransform_row(row)
                    if len(code) < L:
                        code = code + [-1] * (L - len(code))
                    elif len(code) > L:
                        code = code[:L]
                    out[s + i, b] = torch.tensor(code, dtype=torch.int32)
            if (s // batch_size) % 4 == 0:
                dt = time.time() - t0
                rate = e / max(dt, 1e-9)
                print(f"  [beam:{tag}] {e}/{n} ({rate:.1f}/s, {dt:.1f}s)",
                      flush=True)
    return out


# ---- Main pipeline ----------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clipsf_root", type=str,
                   default=os.path.join(HERE, "outputs", "clipsf"),
                   help="Where CLIP_SF features were dumped.")
    p.add_argument("--clipsf_raw_root", type=str,
                   default=os.path.join(HERE, "outputs", "clipsf_raw"),
                   help="Where CLIP_SF *un-normalised* raw features live.")
    p.add_argument("--out_root", type=str,
                   default=os.path.join(HERE, "outputs", "genius_r"))
    p.add_argument("--pool", type=str, default="SHARED")
    p.add_argument("--datasets", nargs="+", default=["OVEN", "INFOSEEK"])
    p.add_argument("--splits", nargs="+", default=["test", "test"])
    p.add_argument("--num_beams", type=int, default=50,
                   help="Beam width; also caps #code sequences returned/query.")
    p.add_argument("--quant_batch", type=int, default=8192,
                   help="Batch size for RQ quantisation (cand pool).")
    p.add_argument("--beam_batch", type=int, default=32,
                   help="Batch size for T5 constrained beam search.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--stages", nargs="+",
                   default=["cand", "trie", "query"],
                   choices=["cand", "trie", "query"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = _build_genius_config()
    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)
    trie_dir = os.path.join(out_root, "trie")
    os.makedirs(trie_dir, exist_ok=True)

    # ---- Stage A: cand pool codes ------------------------------------------
    cand_dir = os.path.join(out_root, "cand_pool", args.pool)
    os.makedirs(cand_dir, exist_ok=True)
    codes_pt = os.path.join(cand_dir, "cand_codes.pt")
    code_idx_pkl = os.path.join(cand_dir, "code_to_rows.pkl")

    if "cand" in args.stages and (args.overwrite or not os.path.isfile(codes_pt)):
        # Load raw features
        raw_dir = os.path.join(args.clipsf_raw_root, "cand_pool", args.pool)
        img_raw = torch.load(os.path.join(raw_dir, "cand_img_raw.pt"),
                             map_location="cpu", weights_only=True)
        txt_raw = torch.load(os.path.join(raw_dir, "cand_txt_raw.pt"),
                             map_location="cpu", weights_only=True)
        print(f"[cand] img_raw {tuple(img_raw.shape)} txt_raw {tuple(txt_raw.shape)}")

        rq = load_rq(cfg, device)
        codes = quantize_pool(rq, img_raw, txt_raw, device,
                              batch_size=args.quant_batch)
        torch.save(codes, codes_pt)
        # Also copy the ordered id list next to codes for convenience.
        cand_ids_src = os.path.join(args.clipsf_root, "cand_pool", args.pool,
                                     "cand_ids.json")
        cand_ids_dst = os.path.join(cand_dir, "cand_ids.json")
        if os.path.isfile(cand_ids_src) and not os.path.isfile(cand_ids_dst):
            import shutil
            shutil.copy(cand_ids_src, cand_ids_dst)
        print(f"[cand] codes -> {codes_pt}, shape={tuple(codes.shape)}")
        del rq
        torch.cuda.empty_cache()
    else:
        print(f"[skip] {codes_pt}")

    codes = torch.load(codes_pt, map_location="cpu", weights_only=True)
    print(f"[cand] codes {tuple(codes.shape)}, dtype={codes.dtype}")

    # ---- Stage B: build code->rows index + trie ---------------------------
    if "trie" in args.stages and (args.overwrite or not os.path.isfile(code_idx_pkl)):
        idx = build_code_to_rows(codes)
        with open(code_idx_pkl, "wb") as f:
            pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[trie] built code_to_rows: {len(idx)} unique codes over "
              f"{codes.shape[0]} rows -> {code_idx_pkl}")
    else:
        print(f"[skip] {code_idx_pkl}")

    # ---- Stage C+D: query beam search ------------------------------------
    if "query" in args.stages:
        model, _external_tokenizer = load_generator(cfg, device)
        # IMPORTANT: after T5ForGenerativeRetrieval.__init__(new_tokenizer=True),
        # the *model* installs a fresh PreTrainedTokenizerFast(WordLevel) at
        # ``model.tokenizer`` -- with special code tokens ``<a0>..<i4095>``.
        # The original T5 SentencePiece tokenizer we passed in stays untouched
        # externally.  Use the model's tokenizer for building the trie so the
        # token-id space matches what the T5 lm_head will emit at generation.
        tokenizer = model.tokenizer
        # Install trie
        trie_pkl = os.path.join(trie_dir, "cand_pool.triecpp.pkl")
        # We stash a distilled ``cand_codes`` list-of-lists to pass to
        # ``generative_index`` (which tokenises to token-ids on the fly).
        unique_codes = np.unique(codes.numpy(), axis=0).tolist()
        print(f"[trie] unique codes for trie: {len(unique_codes)} "
              f"(will build in-memory triecpp)")
        # Manually build the trie (no dist barriers).
        cand_codes_str = [model.transform_row(row) for row in unique_codes]
        cand_codes_ids = np.array(
            tokenizer(cand_codes_str, add_special_tokens=False).input_ids
        ).tolist()
        import models.generative_retriever.trie_cpp as Trie_Cpp
        model.trie_index = Trie_Cpp.Trie(cand_codes_ids)
        print(f"[trie] triecpp ready")

        for ds, sp in zip(args.datasets, args.splits):
            q_out_dir = os.path.join(out_root, "queries", ds, sp)
            os.makedirs(q_out_dir, exist_ok=True)
            topN_pt = os.path.join(q_out_dir, "query_codes_topN.pt")
            if (not args.overwrite) and os.path.isfile(topN_pt):
                print(f"[skip] {topN_pt}")
                continue
            raw_dir = os.path.join(args.clipsf_raw_root, "queries", ds, sp)
            q_img = torch.load(os.path.join(raw_dir, "query_img_raw.pt"),
                               map_location="cpu", weights_only=True)
            q_txt = torch.load(os.path.join(raw_dir, "query_txt_raw.pt"),
                               map_location="cpu", weights_only=True)
            print(f"[query {ds}/{sp}] img_raw {tuple(q_img.shape)} "
                  f"txt_raw {tuple(q_txt.shape)}")

            top = beam_search_queries(
                model, q_img, q_txt, device,
                num_beams=args.num_beams,
                batch_size=args.beam_batch,
                tag=f"{ds}/{sp}",
            )
            torch.save(top, topN_pt)
            # Copy qids + positives from CLIP_SF layout
            src_dir = os.path.join(args.clipsf_root, "queries", ds, sp)
            for n in ("qids.json", "qid_to_positives.json"):
                s = os.path.join(src_dir, n)
                d = os.path.join(q_out_dir, n)
                if os.path.isfile(s) and not os.path.isfile(d):
                    import shutil
                    shutil.copy(s, d)
            print(f"[query {ds}/{sp}] -> {topN_pt} shape={tuple(top.shape)}")

    print("[done] run_genius_r.py")


if __name__ == "__main__":
    main()
