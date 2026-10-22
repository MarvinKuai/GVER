"""Evaluate MGR_PRO on OVEN + INFOSEEK, split-by-dataset,
   R@1 / R@5 / R@10 / R@50 / nDCG@10.

Key improvements over src/pro/evaluate.py:
  1. Split metrics by dataset (OVEN qids start with "5:", INFOSEEK with "6:").
  2. Add R@50 and nDCG@10 (binary relevance, IDCG = sum over min(pos, k)).
  3. Batched encode (avoids OOM on 800k items).
  4. **Bucket tie-break**: within a single beam prefix, multiple items share the
     same beam score. We re-rank them by cosine similarity between the query's
     fused CLIP-SF embedding and each item's fused embedding, breaking the
     trie-insertion-order bias that was hurting R@5..R@10.
  5. Save full ranked results to pickle for offline metric re-computation.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from src.pro.config import load_config, require_path, set_seed, is_placeholder
from src.pro.data import load_torch_file, load_feature_store, load_qrels, load_modalities
from src.pro.quantizer import ResidualQuantizer
from src.pro.decoder import CodeTokenizer, PrefixRetriever, PrefixTrie


# Same constants as data.py
DATASET_QUERY_BOUND = 500_000
DATASET_CANDIDATE_BOUND = 10_000_000


def qid_to_dataset(hashed_qid: int) -> str:
    ds_id = hashed_qid // DATASET_QUERY_BOUND
    if ds_id == 5:
        return "OVEN"
    if ds_id == 6:
        return "INFOSEEK"
    return f"DS{ds_id}"


def recall_at(results, qrels, k, subset=None):
    hits = 0
    total = 0
    for qid, relevant in qrels.items():
        if subset is not None and qid not in subset:
            continue
        predicted = [item for item, _ in results.get(qid, [])[:k]]
        hits += int(any(item in relevant for item in predicted))
        total += 1
    return hits / max(total, 1), total


def ndcg_at(results, qrels, k, subset=None):
    """Binary-relevance nDCG@k. IDCG = sum over min(#pos, k) ideal positions."""
    ndcg_sum = 0.0
    total = 0
    for qid, relevant in qrels.items():
        if subset is not None and qid not in subset:
            continue
        total += 1
        rel_set = set(int(x) for x in relevant)
        if not rel_set:
            continue
        predicted = [int(item) for item, _ in results.get(qid, [])[:k]]
        dcg = 0.0
        for pos, item in enumerate(predicted):
            if item in rel_set:
                dcg += 1.0 / math.log2(pos + 2)
        ideal_hits = min(len(rel_set), k)
        idcg = sum(1.0 / math.log2(p + 2) for p in range(ideal_hits))
        if idcg > 0:
            ndcg_sum += dcg / idcg
    return ndcg_sum / max(total, 1), total


def rerank_full_by_embedding(ranked, query_emb, id_to_idx, item_embs):
    """GENIUS-R style rerank: re-order the ENTIRE shortlist by query-item
    cosine similarity (item_embs are L2-normalised fused features, so a plain
    dot product == cosine sim). The output preserves the original list of item
    ids but with new scores == sim; downstream code truncates to topk.

    This is different from ``rerank_ties_by_embedding`` which only reorders
    within equal-score buckets. Use this for the "PRO + rerank" evaluation
    that parallels GENIUS -> GENIUS-R.
    """
    if not ranked:
        return ranked
    ids = [int(item) for item, _ in ranked]
    idxs = [id_to_idx.get(iid) for iid in ids]
    # keep items that have a fused embedding available
    valid_pairs = [(iid, idx) for iid, idx in zip(ids, idxs) if idx is not None]
    if not valid_pairs:
        return ranked
    id_tensor = torch.tensor([p[1] for p in valid_pairs],
                             dtype=torch.long, device=query_emb.device)
    emb = item_embs.index_select(0, id_tensor).to(query_emb.dtype)  # [G, D]
    sim = (emb @ query_emb).tolist()
    order = sorted(range(len(valid_pairs)), key=lambda i: sim[i], reverse=True)
    out = [(valid_pairs[i][0], float(sim[i])) for i in order]
    # Append items whose embedding was missing (should not happen in practice)
    if len(out) < len(ranked):
        seen = {p[0] for p in valid_pairs}
        for item, score in ranked:
            if int(item) not in seen:
                out.append((int(item), float(score)))
    return out


def rerank_ties_by_embedding(ranked, query_emb, id_to_idx, item_embs, tie_tol=1e-6):
    """Re-rank items sharing (near-)identical beam-search scores by
    query-item dot product. Preserves the original ordering across
    different score buckets but reorders within a bucket.
    """
    if not ranked:
        return ranked
    # group by score with a tolerance
    groups = []
    cur_score = None
    cur_grp = []
    for item, score in ranked:
        if cur_score is None or abs(score - cur_score) > tie_tol:
            if cur_grp:
                groups.append((cur_score, cur_grp))
            cur_grp = [(item, score)]
            cur_score = score
        else:
            cur_grp.append((item, score))
    if cur_grp:
        groups.append((cur_score, cur_grp))

    out = []
    for _, grp in groups:
        if len(grp) == 1:
            out.append(grp[0])
            continue
        idxs = [id_to_idx.get(int(item)) for item, _ in grp]
        valid_pairs = [(item, s, idx) for (item, s), idx in zip(grp, idxs) if idx is not None]
        if not valid_pairs:
            out.extend(grp)
            continue
        item_ids = [p[0] for p in valid_pairs]
        item_scores = [p[1] for p in valid_pairs]
        idx_tensor = torch.tensor([p[2] for p in valid_pairs],
                                  dtype=torch.long, device=query_emb.device)
        emb = item_embs.index_select(0, idx_tensor).to(query_emb.dtype)  # [G, D]
        sim = (emb @ query_emb).tolist()
        order = sorted(range(len(item_ids)), key=lambda i: sim[i], reverse=True)
        # keep the group's leading score for the primary key so cross-group
        # ordering is not perturbed; expose the tie-break sim as a secondary
        # component encoded into the returned score (score + eps*sim_rank).
        for rank_in_grp, i in enumerate(order):
            # subtract a tiny epsilon per rank so cross-group order remains
            out.append((item_ids[i], item_scores[i] - 1e-9 * rank_in_grp))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--limit_queries", type=int, default=-1,
                   help="If >0, only evaluate first N queries (debug/smoke)")
    p.add_argument("--no_tie_break", action="store_true",
                   help="Disable bucket tie-break rerank (default: enabled)")
    p.add_argument("--rerank_full", action="store_true",
                   help="GENIUS-R style: rerank the ENTIRE shortlist by "
                        "query-item fused-embedding cosine similarity, then "
                        "truncate to topk. Overrides --no_tie_break / "
                        "bucket tie-break (they become no-ops).")
    p.add_argument("--rerank_topk", type=int, default=50,
                   help="Only used with --rerank_full: how many items to keep "
                        "after full rerank (default: 50, matches R@50 ceiling).")
    p.add_argument("--output_suffix", type=str, default="",
                   help="Suffix appended before .json/.pkl to distinguish runs, e.g. 'tiebreak' or 'original'")
    p.add_argument("--save_results", type=str, default=None,
                   help="Path to pickle full per-query ranked results (default: alongside output_json)")
    args = p.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 2023))
    paths = config["paths"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}", flush=True)

    # ---- load quantizer ----
    print("[eval] loading quantizer...", flush=True)
    t0 = time.time()
    qz_ckpt = load_torch_file(require_path(paths["quantizer_checkpoint"], "quantizer_checkpoint"),
                              map_location="cpu")
    qz_cfg = qz_ckpt["config"]
    quantizer = ResidualQuantizer(
        qz_cfg["embedding_dim"],
        qz_cfg["codebook_sizes"],
        qz_cfg.get("combiner_projection_dim", 2560),
        qz_cfg.get("combiner_hidden_dim", 5120),
        qz_cfg.get("kmeans_init", True),
        qz_cfg.get("kmeans_iters", 1000),
        qz_cfg.get("codebook_decay", 0.9),
        qz_cfg.get("ema_update", True),
        qz_cfg.get("threshold_ema_dead_code", 2),
    ).to(device)
    quantizer.load_state_dict(qz_ckpt["state_dict"])
    quantizer.eval()
    print(f"[eval] quantizer loaded ({time.time()-t0:.1f}s)", flush=True)

    # ---- load retriever ----
    tokenizer = CodeTokenizer(qz_cfg["codebook_sizes"], qz_cfg.get("modality_tokens", 0))
    print("[eval] loading retriever...", flush=True)
    t0 = time.time()
    rt_ckpt = load_torch_file(require_path(paths["retriever_checkpoint"], "retriever_checkpoint"),
                              map_location="cpu")
    retriever = PrefixRetriever(config["retriever"]["model_name"],
                                config["retriever"]["embedding_dim"], tokenizer).to(device)
    retriever.load_state_dict(rt_ckpt["state_dict"])
    retriever.eval()
    print(f"[eval] retriever loaded ({time.time()-t0:.1f}s)", flush=True)

    # ---- load test data ----
    print("[eval] loading test embeddings & qrels...", flush=True)
    t0 = time.time()
    queries = load_feature_store(require_path(paths["test_query_embeddings"], "test_query_embeddings"))
    items = load_feature_store(require_path(paths["test_item_embeddings"], "test_item_embeddings"))
    qrels = load_qrels(require_path(paths["test_qrels"], "test_qrels"))
    print(f"[eval] queries={len(queries)}, items={len(items)}, qrels={len(qrels)} "
          f"({time.time()-t0:.1f}s)", flush=True)
    modality_path = paths.get("test_item_modalities")
    if modality_path is None or is_placeholder(modality_path):
        modality_path = paths.get("test_item_manifest")
    if modality_path is not None and is_placeholder(modality_path):
        modality_path = None
    item_modalities = None if modality_path is None else load_modalities(modality_path, len(items))

    # ---- encode all items + build trie (batched, keep fused item embeddings for tie-break) ----
    print("[eval] encoding all items + building prefix trie...", flush=True)
    t0 = time.time()
    all_items = items.all()
    n_items = len(items)
    encode_batch = int(config.get("eval", {}).get("encode_batch_size", 16384))
    codes_chunks = []
    fused_chunks = []  # fp16, kept on GPU for tie-break rerank
    with torch.no_grad():
        for start in range(0, n_items, encode_batch):
            end = min(n_items, start + encode_batch)
            if isinstance(all_items, dict):
                batch_feat = {k: v[start:end] for k, v in all_items.items()}
            else:
                batch_feat = all_items[start:end]
            fused = quantizer.encode_features(batch_feat)   # [B, D], already L2-normalized
            c, _, _ = quantizer.quantize(fused)              # [B, L]
            codes_chunks.append(c.cpu())
            fused_chunks.append(fused.to(torch.float16))
            if (start // encode_batch) % 4 == 0:
                print(f"[eval] encoded items {end}/{n_items} "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)
        codes = torch.cat(codes_chunks, dim=0).to(device)
        item_embs_fused = torch.cat(fused_chunks, dim=0)  # [N, D] fp16 on GPU
        if item_modalities is not None and not isinstance(item_modalities, dict):
            item_modalities = item_modalities.to(device)
        elif isinstance(item_modalities, dict):
            m_list = []
            for i, item_id in enumerate(items.ids):
                m_list.append(int(item_modalities.get(int(item_id), 2)))
            item_modalities = torch.tensor(m_list, dtype=torch.long, device=device)
        tokens = tokenizer.encode_codes(
            codes,
            item_modalities if tokenizer.modality_tokens > 0 else None,
        ).cpu().tolist()
    trie = PrefixTrie()
    for item_index, token_row in enumerate(tokens):
        trie.insert(token_row[:-1], int(items.ids[item_index]))
    # Build item_id -> position mapping for fast tie-break rerank
    id_to_idx = {int(items.ids[i]): i for i in range(n_items)}
    print(f"[eval] trie built with {len(tokens)} items, "
          f"fused_embs shape={tuple(item_embs_fused.shape)} dtype={item_embs_fused.dtype} "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ---- beam search ----
    qid_list = list(qrels.keys())
    if args.limit_queries > 0:
        qid_list = qid_list[:args.limit_queries]
        print(f"[eval] SMOKE: limiting to {len(qid_list)} queries", flush=True)

    beam_size = int(config["retriever"]["beam_size"])
    output_size = int(config["retriever"]["output_size"])
    sf_enabled = bool(config["retriever"].get("score_fusion_enabled", False))
    sf_weight = float(config["retriever"].get("score_fusion_weight", 0.0))
    sf_norm = bool(config["retriever"].get("score_fusion_normalize_bias", False))
    do_tie_break = (not args.no_tie_break) and (not args.rerank_full)
    do_rerank_full = bool(args.rerank_full)
    rerank_topk = int(args.rerank_topk)
    print(f"[eval] beam={beam_size}, out={output_size}, "
          f"score_fusion={sf_enabled} w={sf_weight} norm={sf_norm}, "
          f"tie_break={do_tie_break}, rerank_full={do_rerank_full} "
          f"(topk={rerank_topk if do_rerank_full else 'n/a'})", flush=True)

    results = {}
    log_every = max(1, int(args.log_every))
    search_start = time.time()
    for i, qid in enumerate(qid_list):
        with torch.no_grad():
            query = quantizer.encode_features(queries.get(int(qid))).squeeze(0)
        ranked = retriever.beam_search(
            query, trie, quantizer, beam_size, output_size,
            sf_enabled, sf_weight, sf_norm,
        )
        if do_rerank_full and len(ranked) > 1:
            with torch.no_grad():
                ranked = rerank_full_by_embedding(
                    ranked, query, id_to_idx, item_embs_fused,
                )
            ranked = ranked[:rerank_topk]
        elif do_tie_break and len(ranked) > 1:
            with torch.no_grad():
                ranked = rerank_ties_by_embedding(
                    ranked, query, id_to_idx, item_embs_fused,
                )
        results[int(qid)] = ranked
        if (i + 1) % log_every == 0:
            elapsed = time.time() - search_start
            eta = elapsed / (i + 1) * (len(qid_list) - i - 1)
            print(f"[eval] {i+1}/{len(qid_list)} queries done "
                  f"elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
    print(f"[eval] beam search done ({time.time()-search_start:.1f}s)", flush=True)

    # ---- metrics: per-dataset R@1/R@5/R@10/R@50 + nDCG@10 ----
    if args.limit_queries > 0:
        qrels_eval = {q: qrels[q] for q in qid_list if q in qrels}
    else:
        qrels_eval = qrels

    oven_qids = {q for q in qrels_eval if qid_to_dataset(q) == "OVEN"}
    infoseek_qids = {q for q in qrels_eval if qid_to_dataset(q) == "INFOSEEK"}
    metrics = {}
    for name, subset in [("ALL", None), ("OVEN", oven_qids), ("INFOSEEK", infoseek_qids)]:
        r_at = {}
        for k in (1, 5, 10, 50):
            v, n = recall_at(results, qrels_eval, k, subset)
            r_at[f"R@{k}"] = round(v * 100, 4)
        v, n = ndcg_at(results, qrels_eval, 10, subset)
        r_at["nDCG@10"] = round(v * 100, 4)
        r_at["n_queries"] = n
        metrics[name] = r_at

    out_json = require_path(paths["output_json"], "output_json")
    out_json_path = Path(out_json)
    if args.output_suffix:
        suffix = args.output_suffix if args.output_suffix.startswith("_") else f"_{args.output_suffix}"
        out_json_path = out_json_path.with_name(out_json_path.stem + suffix + out_json_path.suffix)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "config": {
                "beam_size": beam_size, "output_size": output_size,
                "score_fusion_enabled": sf_enabled,
                "score_fusion_weight": sf_weight,
                "tie_break": do_tie_break,
                "rerank_full": do_rerank_full,
                "rerank_topk": rerank_topk if do_rerank_full else None,
            },
        }, f, indent=2)
    print(f"[eval] metrics saved: {out_json_path}", flush=True)

    # ---- save full ranked results for offline metric re-computation ----
    save_results = args.save_results
    if save_results is None:
        save_results = str(out_json_path.with_suffix("")) + "_ranked.pkl"
    Path(save_results).parent.mkdir(parents=True, exist_ok=True)
    with open(save_results, "wb") as f:
        pickle.dump({"results": results, "qrels": qrels_eval}, f)
    print(f"[eval] ranked results pickle saved: {save_results}", flush=True)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
