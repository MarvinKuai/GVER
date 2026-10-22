# -*- coding: utf-8 -*-
"""Load UniIR CLIP-SF (ViT-L/14) and encode (image, text) with score-fusion.

CLIP-SF ('Score Fusion') is the UniIR-fine-tuned CLIP checkpoint used by
GENIUS as its Stage-0 encoder.  The state dict is stored under
``ckpt['model']`` with keys prefixed by ``clip_model.``, matching an
``openai/CLIP`` ViT-L/14 backbone verbatim.

Public API:
    load_clip_sf(ckpt_path, base_model='ViT-L/14', device='cpu')
        -> (clip_model, preprocess, tokenize)
    encode_score_fusion(clip_model, preprocess, tokenize, images, texts,
                         device, dtype)
        -> Tensor[N, D] L2-normalised fp16 features

Score-fusion follows GENIUS/UniIR:
    v_img = clip.encode_image(img)                          # [B, D]
    v_txt = clip.encode_text(tok)                           # [B, D]
    fused = L2Norm(L2Norm(v_img) + L2Norm(v_txt))           # [B, D]

For text-only or image-only candidates we set the missing branch to zero
by masking, which the datamodule already encodes via empty text /
placeholder image; but for M-BEIR task 8 both modalities are always
present, so we don't need masks here.
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import clip  # openai-clip
import torch
import torch.nn.functional as F
from PIL import Image


CLIP_SF_STATE_PREFIX = "clip_model."


def _strip_clip_prefix(state_dict: dict) -> dict:
    """Remove the ``clip_model.`` prefix so the SD can be loaded into the
    raw OpenAI CLIP module returned by ``clip.load``."""
    stripped = {}
    for k, v in state_dict.items():
        if k.startswith(CLIP_SF_STATE_PREFIX):
            stripped[k[len(CLIP_SF_STATE_PREFIX):]] = v
        else:
            stripped[k] = v
    return stripped


def load_clip_sf(
    ckpt_path: str,
    base_model: str = "ViT-L/14",
    device: str | torch.device = "cpu",
    download_root: str | None = None,
    fp16: bool = False,
) -> Tuple[torch.nn.Module, Callable, Callable]:
    """Return (clip_model, image_preprocess_fn, tokenize_fn) with CLIP-SF weights.

    Parameters
    ----------
    fp16 : bool
        If True, apply OpenAI-CLIP's ``convert_weights`` to selectively cast
        Conv/Linear/MHA/proj params to fp16 while keeping LayerNorm in fp32
        (as CLIP was originally trained).  Do NOT call ``model.half()`` on
        the whole module: it breaks LayerNorm's internal fp32 promotion.
    """
    clip_model, preprocess = clip.load(
        base_model, device="cpu", jit=False, download_root=download_root
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    sd = _strip_clip_prefix(sd)
    missing, unexpected = clip_model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[clip_sf] load_state_dict missing keys ({len(missing)}): "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[clip_sf] load_state_dict unexpected keys ({len(unexpected)}): "
              f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # Move to target device before mixed-precision casting so that Conv/Linear
    # weights are cast on the destination device (avoids an extra copy).
    clip_model = clip_model.to(device=device)
    if fp16:
        from clip.model import convert_weights
        convert_weights(clip_model)
    clip_model.eval()
    return clip_model, preprocess, clip.tokenize


@torch.no_grad()
def encode_score_fusion(
    clip_model: torch.nn.Module,
    preprocess: Callable,
    tokenize: Callable,
    images: Sequence[Image.Image],
    texts: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
    context_length: int = 77,
) -> torch.Tensor:
    """Encode a mini-batch of (image, text) pairs into L2-normalised fused features.

    Returns fp16 [B, D] on CPU (D = 768 for ViT-L/14).
    """
    assert len(images) == len(texts), "images/texts length mismatch"

    # Image branch --------------------------------------------------------
    pixel = torch.stack([preprocess(img) for img in images], dim=0)
    pixel = pixel.to(device=device, dtype=dtype)
    v_img = clip_model.encode_image(pixel)                        # [B, D]

    # Text branch ---------------------------------------------------------
    tok = tokenize(list(texts), context_length=context_length, truncate=True)
    tok = tok.to(device=device)
    v_txt = clip_model.encode_text(tok)                           # [B, D]

    # Score fusion (float32 for numerical stability) ----------------------
    v_img = v_img.float()
    v_txt = v_txt.float()
    v_img = F.normalize(v_img, dim=-1)
    v_txt = F.normalize(v_txt, dim=-1)
    fused = F.normalize(v_img + v_txt, dim=-1)                    # [B, D]
    return fused.half().cpu()


def encode_score_fusion_split(
    clip_model: torch.nn.Module,
    preprocess: Callable,
    tokenize: Callable,
    images: Sequence[Image.Image],
    texts: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
    context_length: int = 77,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same as ``encode_score_fusion`` but *also* returns the un-fused per-branch
    embeddings needed by GENIUS's residual quantizer and generator.

    Returns (img_emb, txt_emb, fused_emb), all L2-normalised fp16 [B, D] on CPU.
    """
    assert len(images) == len(texts), "images/texts length mismatch"

    pixel = torch.stack([preprocess(img) for img in images], dim=0)
    pixel = pixel.to(device=device, dtype=dtype)
    v_img = clip_model.encode_image(pixel).float()

    tok = tokenize(list(texts), context_length=context_length, truncate=True)
    tok = tok.to(device=device)
    v_txt = clip_model.encode_text(tok).float()

    v_img_n = F.normalize(v_img, dim=-1)
    v_txt_n = F.normalize(v_txt, dim=-1)
    fused   = F.normalize(v_img_n + v_txt_n, dim=-1)

    return v_img_n.half().cpu(), v_txt_n.half().cpu(), fused.half().cpu()


@torch.no_grad()
def encode_score_fusion_raw(
    clip_model: torch.nn.Module,
    preprocess: Callable,
    tokenize: Callable,
    images: Sequence[Image.Image],
    texts: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
    context_length: int = 77,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the *raw* (un-normalised) CLIP per-branch outputs plus a
    score-fusion feature.  GENIUS's residual quantizer / generator were
    trained on the raw ``encode_multimodal_input`` outputs of CLIP-SF (no
    normalisation before the Combiner), so we must expose them for
    downstream quantisation.

    Returns
    -------
    (img_raw, txt_raw, fused_norm) each fp16 [B, D] on CPU, where
    ``fused_norm = L2Norm(L2Norm(v_img) + L2Norm(v_txt))``.
    """
    assert len(images) == len(texts), "images/texts length mismatch"

    pixel = torch.stack([preprocess(img) for img in images], dim=0)
    pixel = pixel.to(device=device, dtype=dtype)
    v_img = clip_model.encode_image(pixel).float()               # raw

    tok = tokenize(list(texts), context_length=context_length, truncate=True)
    tok = tok.to(device=device)
    v_txt = clip_model.encode_text(tok).float()                  # raw

    v_img_n = F.normalize(v_img, dim=-1)
    v_txt_n = F.normalize(v_txt, dim=-1)
    fused = F.normalize(v_img_n + v_txt_n, dim=-1)

    return v_img.half().cpu(), v_txt.half().cpu(), fused.half().cpu()


__all__ = [
    "load_clip_sf",
    "encode_score_fusion",
    "encode_score_fusion_split",
    "encode_score_fusion_raw",
]
