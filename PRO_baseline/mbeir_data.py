# -*- coding: utf-8 -*-
"""Data loading utilities for the CLIP-SF (Score Fusion) baseline on
M-BEIR task 8 (image+text -> image+text), covering OVEN and Infoseek.

For task 8, both the query and the candidate are multimodal (image + text).
CLIP does not natively support fused image+text encoding, so this module only
exposes the raw components (PIL image + string text) and lets the extractor
encode them separately; fusion (image_emb + text_emb, L2 normalised) is done
downstream.

Public entry points
-------------------
* ``load_query_bundle(dataset, split)``   -> MbeirQueryBundle
* ``load_cand_bundle(pool_name)``         -> MbeirCandBundle
* ``load_qrels(dataset, split)``          -> dict[qid -> list[did]] (multi-positive)
* ``build_shared_cand_bundle()``          -> merged (OVEN ∪ Infoseek) task-8 pool

Design decisions
----------------
1. Instruction prepending: UniIR's official M-BEIR protocol prepends a task
   instruction sampled from `instructions/query_instructions.tsv` to each
   query text.  For **deterministic evaluation** we always use ``prompt_1``
   (the first template), matching the UniIR eval convention.
2. Multi-positive: ``qrels`` files list *all* relevant did per qid; we honour
   this by returning ``list[did]`` and letting the evaluator compute hit if
   any positive is retrieved.
3. Shared gallery: task-8 qrels cross-reference OVEN and Infoseek did
   (they share the Wikipedia entity pool).  The recommended M-BEIR-local
   evaluation therefore uses the UNION of both cand pools (816,917 rows).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from mbeir_tar_reader import get_default_reader


HERE = os.path.dirname(os.path.abspath(__file__))                   # .../PRO_baseline
GM2M_ROOT = os.path.dirname(HERE)                                   # .../GM2M_opensource
WORKSPACE = os.path.dirname(GM2M_ROOT)                              # .../xxx
# M-BEIR dataset root. Override with the MBEIR_ROOT env var on your machine.
MBEIR_ROOT = os.environ.get("MBEIR_ROOT", os.path.join(WORKSPACE, "Dataset", "m-beir"))
IMAGES_ROOT = MBEIR_ROOT                                            # img_path already starts with "mbeir_images/..."
INSTRUCTIONS_TSV = os.path.join(MBEIR_ROOT, "instructions", "query_instructions.tsv")

# task 8 dataset_name -> dataset_id  (see query_instructions.tsv)
DATASET_ID = {"OVEN": 5, "INFOSEEK": 6}
TASK_ID = 8

# File name conventions -------------------------------------------------------
QUERY_JSONL = {
    ("OVEN",     "train"): "query/train/mbeir_oven_task8_train.jsonl",
    ("OVEN",     "val"):   "query/val/mbeir_oven_task8_val.jsonl",
    ("OVEN",     "test"):  "query/test/mbeir_oven_task8_test.jsonl",
    ("INFOSEEK", "train"): "query/train/mbeir_infoseek_task8_train.jsonl",
    ("INFOSEEK", "val"):   "query/val/mbeir_infoseek_task8_val.jsonl",
    ("INFOSEEK", "test"):  "query/test/mbeir_infoseek_task8_test.jsonl",
}
CAND_JSONL = {
    "OVEN":     "cand_pool/local/mbeir_oven_task8_cand_pool.jsonl",
    "INFOSEEK": "cand_pool/local/mbeir_infoseek_task8_cand_pool.jsonl",
}
QRELS_TXT = {
    ("OVEN",     "train"): "qrels/train/mbeir_oven_train_qrels.txt",
    ("OVEN",     "val"):   "qrels/val/mbeir_oven_task8_val_qrels.txt",
    ("OVEN",     "test"):  "qrels/test/mbeir_oven_task8_test_qrels.txt",
    ("INFOSEEK", "train"): "qrels/train/mbeir_infoseek_train_qrels.txt",
    ("INFOSEEK", "val"):   "qrels/val/mbeir_infoseek_task8_val_qrels.txt",
    ("INFOSEEK", "test"): "qrels/test/mbeir_infoseek_task8_test_qrels.txt",
}


# ---------------------------------------------------------------------------
# Instruction template loader
# ---------------------------------------------------------------------------
def _load_instructions() -> Dict[Tuple[str, str, str], str]:
    """Return (query_modality, cand_modality, dataset_name) -> prompt_1.

    Only the first prompt template is used (deterministic evaluation).
    """
    with open(INSTRUCTIONS_TSV, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx_qm = header.index("query_modality")
        idx_cm = header.index("cand_modality")
        idx_ds = header.index("dataset_name")
        idx_p1 = header.index("prompt_1")
        table: Dict[Tuple[str, str, str], str] = {}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= idx_p1:
                continue
            table[(parts[idx_qm], parts[idx_cm], parts[idx_ds])] = parts[idx_p1]
    return table


_INSTR_TABLE = _load_instructions()


def get_task8_instruction(dataset: str) -> str:
    """Instruction for M-BEIR task 8 (image,text -> image,text)."""
    key = ("image,text", "image,text", dataset)
    if key not in _INSTR_TABLE:
        raise KeyError(f"No task-8 instruction template for dataset={dataset}")
    return _INSTR_TABLE[key]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class MbeirQueryBundle:
    dataset: str
    split: str
    qids: List[str]                                                 # length N
    query_texts: List[str]                                          # length N (instruction + txt)
    query_image_getters: List[Callable[[], Image.Image]]            # length N
    raw_query_texts: List[str]                                      # length N (without instruction)
    instruction: str
    # From qrels: qid -> list[did] with relevance >= 1
    qid_to_positives: Dict[str, List[str]]
    n: int = field(init=False)

    def __post_init__(self) -> None:
        self.n = len(self.qids)


@dataclass
class MbeirCandBundle:
    name: str                                                       # "OVEN" / "INFOSEEK" / "SHARED"
    dids: List[str]                                                 # length N
    cand_texts: List[str]                                           # length N (raw wiki txt)
    cand_image_getters: List[Callable[[], Image.Image]]             # length N
    n: int = field(init=False)

    def __post_init__(self) -> None:
        self.n = len(self.dids)


# ---------------------------------------------------------------------------
# Image getter factories
# ---------------------------------------------------------------------------
def _resolve_image(rel_path: str) -> str:
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(IMAGES_ROOT, rel_path)


def _make_getter(rel_path: str) -> Callable[[], Image.Image]:
    """Return a zero-arg callable that yields a PIL image.

    Priority:
    1. If a tar-based reader is configured (env MBEIR_TAR_PATH / MBEIR_TAR_INDEX)
       and it knows about ``rel_path``, read from the tar (O(1) random access).
    2. Otherwise fall back to reading the loose file from ``IMAGES_ROOT``.

    ``rel_path`` is expected to start with ``mbeir_images/`` (as stored in the
    query/cand jsonl).  Absolute paths are also accepted and always read from
    disk.
    """
    reader = get_default_reader()
    if reader is not None and not os.path.isabs(rel_path) and rel_path in reader:
        return reader.make_getter(rel_path)

    abs_path = _resolve_image(rel_path)

    def _get() -> Image.Image:
        return Image.open(abs_path).convert("RGB")

    return _get


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_qrels(dataset: str, split: str) -> Dict[str, List[str]]:
    """Parse a TREC-style qrels file into qid -> [did, ...] (positives only)."""
    key = (dataset, split)
    if key not in QRELS_TXT:
        raise KeyError(f"No qrels for ({dataset}, {split})")
    path = os.path.join(MBEIR_ROOT, QRELS_TXT[key])
    qid_to_pos: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _iter, did, rel = parts[0], parts[1], parts[2], parts[3]
            try:
                if int(rel) <= 0:
                    continue
            except ValueError:
                continue
            qid_to_pos.setdefault(qid, []).append(did)
    return qid_to_pos


def load_query_bundle(dataset: str, split: str) -> MbeirQueryBundle:
    """Load task-8 queries for OVEN or Infoseek at a given split."""
    if dataset not in DATASET_ID:
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    if (dataset, split) not in QUERY_JSONL:
        raise ValueError(f"No query file for ({dataset}, {split})")
    rows = _read_jsonl(os.path.join(MBEIR_ROOT, QUERY_JSONL[(dataset, split)]))
    instruction = get_task8_instruction(dataset)

    qid_to_pos = load_qrels(dataset, split) if (dataset, split) in QRELS_TXT else {}

    qids: List[str] = []
    query_texts: List[str] = []
    raw_texts: List[str] = []
    getters: List[Callable[[], Image.Image]] = []
    n_qrels_fallback = 0
    n_no_positive = 0
    for r in rows:
        qid = str(r["qid"])
        raw = str(r.get("query_txt") or "").strip()
        img_rel = r.get("query_img_path") or ""
        # Instruction prepended (space-joined), consistent with UniIR eval.
        full_text = f"{instruction} {raw}".strip() if raw else instruction
        qids.append(qid)
        raw_texts.append(raw)
        query_texts.append(full_text)
        getters.append(_make_getter(img_rel))
        # Fallback: if qrels does not cover this qid (typical for train split
        # whose qrels file may be per-dataset instead of per-task), use the
        # inline pos_cand_list carried by the jsonl row itself.  This keeps
        # (dataset, split)=("*", "train") usable for supervised training.
        if qid not in qid_to_pos:
            inline_pos = r.get("pos_cand_list") or []
            if inline_pos:
                qid_to_pos[qid] = [str(x) for x in inline_pos]
                n_qrels_fallback += 1
            else:
                n_no_positive += 1

    if n_qrels_fallback:
        print(f"[info][{dataset}/{split}] {n_qrels_fallback}/{len(qids)} queries "
              f"used inline pos_cand_list (no qrels entry).")
    if n_no_positive:
        print(f"[warn][{dataset}/{split}] {n_no_positive}/{len(qids)} queries "
              f"have neither qrels nor inline positives.")

    return MbeirQueryBundle(
        dataset=dataset,
        split=split,
        qids=qids,
        query_texts=query_texts,
        query_image_getters=getters,
        raw_query_texts=raw_texts,
        instruction=instruction,
        qid_to_positives=qid_to_pos,
    )


def load_cand_bundle(pool_name: str) -> MbeirCandBundle:
    """Load a single-dataset task-8 candidate pool (OVEN or INFOSEEK)."""
    if pool_name not in CAND_JSONL:
        raise ValueError(f"Unknown pool: {pool_name!r}")
    rows = _read_jsonl(os.path.join(MBEIR_ROOT, CAND_JSONL[pool_name]))
    dids: List[str] = []
    texts: List[str] = []
    getters: List[Callable[[], Image.Image]] = []
    seen = set()
    for r in rows:
        did = str(r["did"])
        if did in seen:
            continue
        seen.add(did)
        dids.append(did)
        texts.append(str(r.get("txt") or ""))
        getters.append(_make_getter(r.get("img_path") or ""))
    return MbeirCandBundle(name=pool_name, dids=dids, cand_texts=texts, cand_image_getters=getters)


def build_shared_cand_bundle() -> MbeirCandBundle:
    """Merge OVEN and Infoseek task-8 cand pools (union of did).

    OVEN pool contains only dids with prefix '5:'; Infoseek only with '6:'.
    They are disjoint by construction, so a simple concatenation preserves
    uniqueness. Order: OVEN first, then Infoseek (deterministic).
    """
    oven = load_cand_bundle("OVEN")
    info = load_cand_bundle("INFOSEEK")
    dids = oven.dids + info.dids
    texts = oven.cand_texts + info.cand_texts
    getters = oven.cand_image_getters + info.cand_image_getters
    # Sanity: dids must be unique
    if len(set(dids)) != len(dids):
        raise RuntimeError("Shared task-8 pool has duplicate dids!")
    return MbeirCandBundle(name="SHARED", dids=dids, cand_texts=texts, cand_image_getters=getters)


__all__ = [
    "MbeirQueryBundle",
    "MbeirCandBundle",
    "load_query_bundle",
    "load_cand_bundle",
    "build_shared_cand_bundle",
    "load_qrels",
    "get_task8_instruction",
    "DATASET_ID",
    "TASK_ID",
    "MBEIR_ROOT",
]
