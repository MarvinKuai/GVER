"""M2M-SID - Fast batched evaluator (strict no-rerank, no-tie-break).

Key design decisions:

  * Trie is flattened level-by-level. For every trie node we store the set
    of allowed children as (parent_node_id, child_token_in_level) pairs.
    This is stored in CSR form so the mask at each step only lives in the
    current level's codebook_size dimension (<= 2048) instead of the full
    T5 vocabulary (which can be 10k+ -> OOM).
  * All T5 decoder forward passes run in a single call of shape
    ``[B*beam, step+1]`` -> ``[B*beam, vocab]``.
  * Score fusion (geometric bias) is fully vectorised over
    ``[B*beam, codebook_size]`` and only applied on the current level slice.
  * Per-query ``prefix_reconstruction`` is maintained as a float tensor of
    shape ``[B, beam, D]``.

K-view handling: queries are tiled V times along the batch axis so the
beam search treats every view as an independent query; afterwards results
are grouped back by qid and merged by maximum score (same as the slow
version).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.m2m_sid.config import load_config, require_path, set_seed, is_placeholder
from src.m2m_sid.data import load_feature_store, load_qrels, load_modalities, load_torch_file
from src.m2m_sid.quantizer import ResidualQuantizer
from src.m2m_sid.decoder import CodeTokenizer, PrefixRetriever, PrefixTrie


DATASET_QUERY_BOUND = 500_000


def qid_to_dataset(hashed_qid):
    ds_id = hashed_qid // DATASET_QUERY_BOUND
    if ds_id == 5:
        return "OVEN"
    if ds_id == 6:
        return "INFOSEEK"
    return f"DS{ds_id}"


# ---------- metrics ----------

def recall_at(results, qrels, k, subset=None):
    hits, total = 0, 0
    for qid, relevant in qrels.items():
        if subset is not None and qid not in subset:
            continue
        total += 1
        predicted = [item for item, _ in results.get(qid, [])[:k]]
        hits += int(any(item in relevant for item in predicted))
    return hits / max(total, 1), total


def ndcg_at(results, qrels, k, subset=None):
    ndcg_sum, total = 0.0, 0
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


# ---------- level-wise flattened trie ----------

class LevelTrie:
    """Flat per-level trie representation.

    For each level ``L`` we store a dense table
    ``child_table[L]: [num_nodes_at_level_L, codebook_sizes[L]]``
    where entry ``[n, c]`` is the child node id in level ``L+1`` for parent
    node ``n`` taking code ``c``, or ``-1`` if that transition is not in the
    trie.

    * ``num_nodes_at_level_L`` is bounded by the number of distinct SID
      prefixes of length L, which is <= num_items (335K / 481K) and in
      practice much smaller because many items share prefixes.
    * ``codebook_sizes[L]`` <= 2048, so each table is at most
      ``num_items * 2048 * 4 bytes`` ~ 2.7 GB; we cap it with int32 and
      keep tables on CPU, moving slices to GPU per step.

    If modality tokens are used, level 0 of this trie actually represents
    the modality step (3 options). We treat the modality step as "level
    -1" and store it as ``modality_table`` of shape ``[1, modality_tokens]``.
    """

    def __init__(self, trie: PrefixTrie, tokenizer: CodeTokenizer, device: torch.device):
        self.tokenizer = tokenizer
        self.device = device
        self.levels = len(tokenizer.codebook_sizes)
        self.modality_tokens = tokenizer.modality_tokens
        self.total_steps = self.levels + (1 if self.modality_tokens > 0 else 0)

        # per-step tables: list of tensors, each shape [num_parents_at_step, step_vocab]
        self.child_tables = []            # on CPU int32
        self.step_vocab_offset = []       # token id offset for step s (so token = offset + code_within_level)
        self.step_vocab_size = []         # vocab size for step s
        self.step_level = []              # the quantizer level L corresponding to step s (or None for modality step)

        # node_prefix[step][node_id] -> prefix tuple (for leaf item lookup).
        self.node_prefix = []
        # items at each node (keyed by (step, node_id)): leaf items.
        self.leaf_items = {}

        # BFS the trie level by level.
        # root is node 0 at step 0.
        current_nodes = [((), trie.children)]   # list of (prefix_tuple, children_dict)
        self.node_prefix.append([()])
        if trie.items.get((), []):
            self.leaf_items[(0, 0)] = list(trie.items[()])

        for step in range(self.total_steps):
            level = tokenizer.level_for_step(step)
            if level is None:
                step_offset = tokenizer.modality_offset
                step_size = int(tokenizer.modality_tokens)
            else:
                step_offset = int(tokenizer.level_offsets[int(level)])
                step_size = int(tokenizer.codebook_sizes[int(level)])
            self.step_vocab_offset.append(step_offset)
            self.step_vocab_size.append(step_size)
            self.step_level.append(level)

            # Build child_table for this step.
            next_nodes = []
            table = torch.full((len(current_nodes), step_size), -1, dtype=torch.int32)
            for parent_id, (prefix, child_dict) in enumerate(current_nodes):
                for tok, grand_children in child_dict.items():
                    code = int(tok) - step_offset
                    if code < 0 or code >= step_size:
                        continue
                    child_id = len(next_nodes)
                    child_prefix = prefix + (int(tok),)
                    next_nodes.append((child_prefix, grand_children))
                    table[parent_id, code] = child_id
                    # Record leaf items at this child if any.
                    items_here = trie.items.get(child_prefix, [])
                    if items_here:
                        self.leaf_items[(step + 1, child_id)] = list(items_here)
            self.child_tables.append(table)
            self.node_prefix.append([n[0] for n in next_nodes])
            current_nodes = next_nodes
            if not current_nodes:
                # trie exhausted earlier than expected (shouldn't happen for a
                # well-formed SID trie); break early.
                break

        self.num_steps = len(self.child_tables)


# ---------- batched beam search ----------

@torch.no_grad()
def batched_beam_search(
    retriever: PrefixRetriever,
    quantizer: ResidualQuantizer,
    tokenizer: CodeTokenizer,
    query_embeddings: torch.Tensor,   # [N, D] (already view-flattened)
    level_trie: LevelTrie,
    beam_size: int,
    output_size: int,
    score_fusion_enabled: bool,
    score_fusion_weight: float,
    score_fusion_normalize_bias: bool,
    batch_size: int,
    device: torch.device,
    log_every: int = 500,
    log_prefix: str = "batched_beam",
    rank_aware_weight: bool = False,
    rank_aware_head_weight: float = None,
    rank_aware_tail_weight: float = None,
    rank_aware_head_frac: float = 0.2,
):
    """Return ``results_per_query: list[list[(item_id, score)]]`` of length N."""
    N = int(query_embeddings.shape[0])
    beam = int(beam_size)
    out = int(output_size)
    vocab = tokenizer.vocab_size
    total_steps = level_trie.num_steps
    bos = tokenizer.bos_token_id

    codebooks = [cb.to(device) for cb in quantizer.codebooks]
    # child_tables stay on CPU (int32). For each beam-search step we only
    # need (a) the rows pointed to by the current B*beam parents, which is
    # tiny; and (b) a single scalar lookup per (parent, chosen code) when
    # advancing node_id. Materialising the full table on GPU would cost
    # ``parents * step_vocab * 8B`` which blows up to tens of GB under the
    # bigcb config (post-4 levels: 335K parents x 4096 codes x 8B = 11GB).
    child_tables_cpu = level_trie.child_tables  # list of int32 CPU tensors

    def _gather_child_rows(step_idx: int, flat_node_cpu: torch.Tensor) -> torch.Tensor:
        """Return child-id rows for the given parents at this step, on device.

        flat_node_cpu: [M] int64 on CPU. Output: [M, step_vocab] int64 on device.
        """
        rows = child_tables_cpu[step_idx].index_select(0, flat_node_cpu)  # CPU int32
        return rows.to(device, non_blocking=True).long()

    def _gather_child_scalars(step_idx: int, parent_cpu: torch.Tensor, code_cpu: torch.Tensor) -> torch.Tensor:
        """Return child-id scalars for (parent, code) pairs, on device.

        parent_cpu / code_cpu: [M] int64 on CPU. Output: [M] int64 on device.
        """
        tbl = child_tables_cpu[step_idx]
        # tbl[parent_cpu, code_cpu] -> [M] int32 on CPU
        vals = tbl[parent_cpu, code_cpu]
        return vals.to(device, non_blocking=True).long()

    step_offsets = level_trie.step_vocab_offset
    step_sizes = level_trie.step_vocab_size
    step_levels = level_trie.step_level

    results = [None] * N
    search_start = time.time()
    seen = 0
    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        B = end - start
        q = query_embeddings[start:end].to(device)                              # [B, D]
        enc_inputs = retriever.query_projection(q).unsqueeze(1)                 # [B, 1, dm]
        attn_single = torch.ones((B, 1), dtype=torch.long, device=device)
        enc_hidden = retriever.model.encoder(
            inputs_embeds=enc_inputs, attention_mask=attn_single,
        ).last_hidden_state                                                     # [B, 1, dm]

        # Beam state:
        #   node_id   : [B, beam] (position in current step's node list)
        #   score     : [B, beam]
        #   prefix    : [B, beam, total_steps]
        #   recon     : [B, beam, D]
        node_id = torch.zeros((B, beam), dtype=torch.long, device=device)
        score = torch.full((B, beam), float("-inf"), device=device)
        score[:, 0] = 0.0
        prefix = torch.zeros((B, beam, total_steps), dtype=torch.long, device=device)
        recon = torch.zeros((B, beam, int(q.shape[-1])), device=device)

        for step in range(total_steps):
            level = step_levels[step]
            step_offset = step_offsets[step]
            step_size = step_sizes[step]

            # Build decoder input [B, beam, step+1] = BOS + prefix[:step].
            dec_in = torch.full((B, beam, step + 1), bos, dtype=torch.long, device=device)
            if step > 0:
                dec_in[:, :, 1:step + 1] = prefix[:, :, :step]
            dec_in_flat = dec_in.reshape(B * beam, step + 1)                    # [B*beam, L]

            enc_rep = enc_hidden.unsqueeze(1).expand(B, beam, 1, enc_hidden.shape[-1])
            enc_rep = enc_rep.reshape(B * beam, 1, enc_hidden.shape[-1])
            attn_rep = torch.ones((B * beam, 1), dtype=torch.long, device=device)
            out_fwd = retriever.model(
                encoder_outputs=(enc_rep,),
                attention_mask=attn_rep,
                decoder_input_ids=dec_in_flat,
                use_cache=False,
            )
            logits = out_fwd.logits[:, -1, :]                                   # [B*beam, V]
            del out_fwd
            # Need log_softmax over the full vocabulary to be numerically
            # faithful to the original beam search. Compute log_softmax over
            # all V and then slice (equivalent to original single-beam code
            # which does log_softmax over all V and then looks up allowed).
            full_log_probs = F.log_softmax(logits, dim=-1)
            del logits
            log_probs_slice = full_log_probs[:, step_offset:step_offset + step_size].contiguous()  # [B*beam, step_size]
            del full_log_probs

            # Mask forbidden codes using child_table (-1 means not allowed).
            flat_node = node_id.reshape(B * beam)                               # [B*beam]
            # Pull just the needed rows from the CPU child table to GPU.
            # This avoids materialising the whole [num_parents x step_vocab]
            # table on GPU, which can be tens of GB under bigcb codebooks.
            flat_node_cpu = flat_node.detach().cpu()
            child_ids = _gather_child_rows(step, flat_node_cpu)                 # [B*beam, step_size] int64 on device
            NEG_INF = torch.finfo(log_probs_slice.dtype).min / 4
            allowed_mask = child_ids >= 0                                       # [B*beam, step_size]
            masked_log_probs = torch.where(
                allowed_mask, log_probs_slice, torch.full_like(log_probs_slice, NEG_INF),
            )

            # Score fusion.
            if score_fusion_enabled and level is not None:
                cb = codebooks[int(level)]                                      # [C, D]
                cb_size = int(cb.shape[0])
                recon_flat = recon.reshape(B * beam, -1)
                q_tiled = q.unsqueeze(1).expand(B, beam, q.shape[-1]).reshape(B * beam, -1)
                residual = q_tiled - recon_flat
                bias = 2.0 * residual @ cb.t() - cb.pow(2).sum(dim=1)           # [B*beam, C]
                if score_fusion_normalize_bias:
                    mean = bias.mean(dim=-1, keepdim=True)
                    std = bias.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
                    bias = (bias - mean) / std
                # Rank-aware weighting: give head-of-beam slots a stronger
                # geometric prior (sharpens R@1) while leaving tail slots with
                # a lighter prior (preserves trie coverage -> R@50).
                if rank_aware_weight:
                    head_w = float(rank_aware_head_weight if rank_aware_head_weight is not None else score_fusion_weight * 1.5)
                    tail_w = float(rank_aware_tail_weight if rank_aware_tail_weight is not None else score_fusion_weight * 0.25)
                    head_cut = max(1, int(round(float(rank_aware_head_frac) * beam)))
                    rank_weights = torch.full((beam,), tail_w, device=device, dtype=bias.dtype)
                    rank_weights[:head_cut] = head_w
                    w_vec = rank_weights.unsqueeze(0).expand(B, beam).reshape(B * beam, 1)
                    bias = w_vec * bias
                else:
                    bias = float(score_fusion_weight) * bias
                # cb_size should == step_size when score_fusion applies to a code level.
                masked_log_probs = masked_log_probs + bias[:, :step_size]

            # Combine with cumulative beam scores: [B, beam, step_size].
            beam_score = score.unsqueeze(-1) + masked_log_probs.reshape(B, beam, step_size)
            flat_scores = beam_score.reshape(B, beam * step_size)
            topk_scores, topk_idx = flat_scores.topk(beam, dim=-1)              # [B, beam]
            parent = topk_idx // step_size
            next_code = topk_idx % step_size

            batch_idx = torch.arange(B, device=device).unsqueeze(-1).expand(B, beam)
            new_prefix = prefix[batch_idx, parent].clone()
            new_recon = recon[batch_idx, parent].clone()
            # Record the token (= step_offset + code) at this step.
            new_prefix[:, :, step] = next_code + int(step_offset)
            # Update node_id via child_table. Dead beams (parent came from a
            # -inf position) may read bogus child ids; we clamp them to 0 and
            # keep their score at -inf so they never contribute.
            parent_node = node_id[batch_idx, parent]
            # Scalar lookup of child id for the chosen (parent, code) pair.
            # Done via CPU index since the full table is CPU-resident.
            parent_node_cpu = parent_node.detach().reshape(-1).cpu()
            next_code_cpu = next_code.detach().reshape(-1).cpu()
            new_node = _gather_child_scalars(step, parent_node_cpu, next_code_cpu).reshape(parent_node.shape)
            dead = ~torch.isfinite(topk_scores)
            new_node = torch.where(
                new_node >= 0, new_node, torch.zeros_like(new_node),
            )
            new_node = torch.where(
                dead, torch.zeros_like(new_node), new_node,
            )
            # Update reconstruction accumulator with chosen codebook row.
            if level is not None:
                code = next_code.clamp(min=0, max=int(codebooks[int(level)].shape[0]) - 1)
                vec = codebooks[int(level)].index_select(0, code.reshape(-1)).reshape(B, beam, -1)
                # Zero out contribution from dead beams.
                vec = torch.where(dead.unsqueeze(-1), torch.zeros_like(vec), vec)
                new_recon = new_recon + vec
            prefix = new_prefix
            recon = new_recon
            node_id = new_node
            score = topk_scores

            if not torch.isfinite(score).any():
                break

        # ----- gather leaf items -----
        final_step = min(step + 1, total_steps)
        node_id_cpu = node_id.cpu().tolist()
        score_cpu = score.cpu().tolist()
        for b in range(B):
            best = {}
            for k in range(beam):
                nid = int(node_id_cpu[b][k])
                sc = float(score_cpu[b][k])
                if not math.isfinite(sc):
                    continue
                items = level_trie.leaf_items.get((final_step, nid), ())
                for it in items:
                    prev = best.get(int(it))
                    if prev is None or sc > prev:
                        best[int(it)] = sc
            ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:out]
            results[start + b] = ranked
        seen += B
        if seen >= log_every or end == N:
            elapsed = time.time() - search_start
            qps = (end) / elapsed if elapsed > 0 else 0.0
            eta = (N - end) / qps if qps > 0 else 0.0
            print(f"[{log_prefix}] {end}/{N} done elapsed={elapsed:.1f}s qps={qps:.1f} eta={eta:.1f}s", flush=True)
            seen = 0
    return results


# ---------- helpers ----------

def merge_view_results(view_lists, output_size):
    best = {}
    for lst in view_lists:
        for item, score in lst:
            prev = best.get(int(item))
            if prev is None or score > prev:
                best[int(item)] = float(score)
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:int(output_size)]


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--limit_queries", type=int, default=-1)
    parser.add_argument("--output_suffix", type=str, default="")
    parser.add_argument("--save_results", type=str, default=None)
    parser.add_argument("--query_batch_size", type=int, default=0,
                        help="Number of (query, view) pairs processed together. 0 = auto by vocab/items.")
    # CLI overrides for fusion params (so one ckpt can be evaluated under
    # many fusion configurations without duplicating YAML files).
    parser.add_argument("--override_head_weight", type=float, default=None)
    parser.add_argument("--override_tail_weight", type=float, default=None)
    parser.add_argument("--override_head_frac", type=float, default=None)
    parser.add_argument("--override_fusion_weight", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 2023))
    paths = config["paths"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_fast] device={device}", flush=True)

    t0 = time.time()
    qz_ckpt = load_torch_file(require_path(paths["quantizer_checkpoint"], "quantizer_checkpoint"), map_location="cpu")
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
        qz_cfg.get("n_views", 1),
    ).to(device)
    quantizer.load_state_dict(qz_ckpt["state_dict"])
    quantizer.eval()
    n_views = int(getattr(quantizer, "n_views", 1))
    print(f"[eval_fast] quantizer loaded, n_views={n_views} ({time.time()-t0:.1f}s)", flush=True)

    tokenizer = CodeTokenizer(qz_cfg["codebook_sizes"], qz_cfg.get("modality_tokens", 0))

    t0 = time.time()
    rt_ckpt = load_torch_file(require_path(paths["retriever_checkpoint"], "retriever_checkpoint"), map_location="cpu")
    retriever = PrefixRetriever(config["retriever"]["model_name"], config["retriever"]["embedding_dim"], tokenizer).to(device)
    retriever.load_state_dict(rt_ckpt["state_dict"])
    retriever.eval()
    print(f"[eval_fast] retriever loaded ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    queries = load_feature_store(require_path(paths["test_query_embeddings"], "test_query_embeddings"))
    items = load_feature_store(require_path(paths["test_item_embeddings"], "test_item_embeddings"))
    qrels = load_qrels(require_path(paths["test_qrels"], "test_qrels"))
    print(f"[eval_fast] queries={len(queries)} items={len(items)} qrels={len(qrels)} ({time.time()-t0:.1f}s)", flush=True)

    modality_path = paths.get("test_item_modalities")
    if modality_path is None or is_placeholder(modality_path):
        modality_path = paths.get("test_item_manifest")
    if modality_path is not None and is_placeholder(modality_path):
        modality_path = None
    item_modalities = None if modality_path is None else load_modalities(modality_path, len(items))

    print("[eval_fast] encoding items + building trie...", flush=True)
    t0 = time.time()
    all_items = items.all()
    n_items = len(items)
    encode_batch = int(config.get("eval", {}).get("encode_batch_size", 16384))
    codes_chunks = []
    with torch.no_grad():
        for start in range(0, n_items, encode_batch):
            end = min(n_items, start + encode_batch)
            if isinstance(all_items, dict):
                batch_feat = {k: v[start:end] for k, v in all_items.items()}
            else:
                batch_feat = all_items[start:end]
            fused = quantizer.encode_features(batch_feat)
            if fused.dim() == 3:
                fused = fused[:, 0, :]
            c, _, _ = quantizer.quantize(fused)
            codes_chunks.append(c.cpu())
        codes = torch.cat(codes_chunks, dim=0).to(device)
        if item_modalities is not None and not isinstance(item_modalities, dict):
            item_modalities = item_modalities.to(device)
        elif isinstance(item_modalities, dict):
            m_list = [int(item_modalities.get(int(items.ids[i]), 2)) for i in range(n_items)]
            item_modalities = torch.tensor(m_list, dtype=torch.long, device=device)
        tokens = tokenizer.encode_codes(codes, item_modalities if tokenizer.modality_tokens > 0 else None).cpu().tolist()
    trie = PrefixTrie()
    for item_index, token_row in enumerate(tokens):
        trie.insert(token_row[:-1], int(items.ids[item_index]))
    print(f"[eval_fast] python-trie built with {len(tokens)} items ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    level_trie = LevelTrie(trie, tokenizer, device=device)
    sizes = [t.shape[0] for t in level_trie.child_tables]
    print(f"[eval_fast] level-trie built: steps={level_trie.num_steps} parents/step={sizes} ({time.time()-t0:.1f}s)", flush=True)

    qid_list = list(qrels.keys())
    if args.limit_queries > 0:
        qid_list = qid_list[:args.limit_queries]
        print(f"[eval_fast] SMOKE: limiting to {len(qid_list)} queries", flush=True)

    print("[eval_fast] encoding all queries...", flush=True)
    t0 = time.time()
    all_q = queries.all()
    if isinstance(all_q, dict):
        if queries.id_to_index is not None:
            idx = torch.tensor([queries._index(q) for q in qid_list], dtype=torch.long)
        else:
            idx = torch.tensor(qid_list, dtype=torch.long)
        batch_feat = {k: v[idx] for k, v in all_q.items()}
    else:
        idx = torch.tensor([queries._index(q) for q in qid_list], dtype=torch.long)
        batch_feat = all_q[idx]
    with torch.no_grad():
        q_enc_chunks = []
        bs = int(config.get("eval", {}).get("query_encode_batch_size", 8192))
        for s in range(0, len(qid_list), bs):
            e = min(len(qid_list), s + bs)
            if isinstance(batch_feat, dict):
                sub = {k: v[s:e] for k, v in batch_feat.items()}
            else:
                sub = batch_feat[s:e]
            fused = quantizer.encode_features(sub)
            q_enc_chunks.append(fused.detach())
    q_enc = torch.cat(q_enc_chunks, dim=0)
    if q_enc.dim() == 2:
        q_enc = q_enc.unsqueeze(1)
    N, V, D = q_enc.shape
    q_flat = q_enc.reshape(N * V, D)
    print(f"[eval_fast] encoded {N} queries, V={V} ({time.time()-t0:.1f}s)", flush=True)

    beam_size = int(config["retriever"]["beam_size"])
    output_size = int(config["retriever"]["output_size"])
    sf_enabled = bool(config["retriever"].get("score_fusion_enabled", False))
    sf_weight = float(config["retriever"].get("score_fusion_weight", 0.0))
    sf_norm = bool(config["retriever"].get("score_fusion_normalize_bias", False))
    # Rank-aware score fusion: head-of-beam gets a stronger geometric
    # prior than tail-of-beam. If disabled we fall back to the uniform-weight
    # baseline fusion variant.
    sf_rank_aware = bool(config["retriever"].get("score_fusion_rank_aware", False))
    sf_head_weight = config["retriever"].get("score_fusion_head_weight", None)
    sf_tail_weight = config["retriever"].get("score_fusion_tail_weight", None)
    sf_head_frac = float(config["retriever"].get("score_fusion_head_frac", 0.2))
    # Apply CLI overrides if provided (allows one-shot fusion sweeps on a
    # frozen checkpoint without having to clone the YAML).
    if args.override_fusion_weight is not None:
        sf_weight = float(args.override_fusion_weight)
    if args.override_head_weight is not None:
        sf_head_weight = float(args.override_head_weight)
    if args.override_tail_weight is not None:
        sf_tail_weight = float(args.override_tail_weight)
    if args.override_head_frac is not None:
        sf_head_frac = float(args.override_head_frac)

    # Auto-select batch size when the user did not override it.
    # The dominant GPU cost per micro-batch is the T5 lm_head output of shape
    # [B*beam, L, vocab] in fp32; with L up to total_steps (~17) that is
    # 4 * B * beam * 17 * vocab bytes. We budget ~16 GB for this activation
    # and clamp batch to a safe window [8, 256].
    if int(args.query_batch_size) > 0:
        batch_size_eff = int(args.query_batch_size)
    else:
        vocab = int(tokenizer.vocab_size)
        total_steps_est = int(level_trie.num_steps)
        # Bytes per sample: 4 (fp32) * beam * L * vocab.
        per_sample_bytes = 4.0 * float(beam_size) * float(total_steps_est) * float(vocab)
        budget_bytes = 16.0 * (1024 ** 3)   # ~16 GB headroom for lm_head output
        auto_bs = int(max(8, min(256, budget_bytes / max(1.0, per_sample_bytes))))
        # Round to multiple of 8 for cleaner shapes.
        batch_size_eff = max(8, (auto_bs // 8) * 8)
        print(f"[eval_fast] auto query_batch_size={batch_size_eff} "
              f"(vocab={vocab}, beam={beam_size}, steps={total_steps_est})", flush=True)

    print(f"[eval_fast] beam={beam_size} out={output_size} n_views={V} "
          f"score_fusion={sf_enabled} w={sf_weight} norm={sf_norm} "
          f"rank_aware={sf_rank_aware} head_w={sf_head_weight} tail_w={sf_tail_weight} head_frac={sf_head_frac} "
          f"batch={batch_size_eff}", flush=True)

    flat_results = batched_beam_search(
        retriever, quantizer, tokenizer, q_flat, level_trie,
        beam_size, output_size, sf_enabled, sf_weight, sf_norm,
        batch_size=batch_size_eff, device=device,
        log_every=int(args.log_every), log_prefix="eval_fast",
        rank_aware_weight=sf_rank_aware,
        rank_aware_head_weight=(None if sf_head_weight is None else float(sf_head_weight)),
        rank_aware_tail_weight=(None if sf_tail_weight is None else float(sf_tail_weight)),
        rank_aware_head_frac=sf_head_frac,
    )

    results = {}
    for i, qid in enumerate(qid_list):
        view_lists = [flat_results[i * V + v] for v in range(V)]
        results[int(qid)] = (view_lists[0] if V == 1
                             else merge_view_results(view_lists, output_size))

    qrels_eval = {q: qrels[q] for q in qid_list if q in qrels} if args.limit_queries > 0 else qrels
    oven_qids = {q for q in qrels_eval if qid_to_dataset(q) == "OVEN"}
    infoseek_qids = {q for q in qrels_eval if qid_to_dataset(q) == "INFOSEEK"}
    metrics = {}
    for name, subset in [("ALL", None), ("OVEN", oven_qids), ("INFOSEEK", infoseek_qids)]:
        r = {}
        for k in (1, 5, 10, 50):
            v, n = recall_at(results, qrels_eval, k, subset)
            r[f"R@{k}"] = round(v * 100, 4)
        v, n = ndcg_at(results, qrels_eval, 10, subset)
        r["nDCG@10"] = round(v * 100, 4)
        r["n_queries"] = n
        metrics[name] = r

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
                "n_views": V,
                "score_fusion_enabled": sf_enabled,
                "score_fusion_weight": sf_weight,
                "score_fusion_rank_aware": sf_rank_aware,
                "score_fusion_head_weight": sf_head_weight,
                "score_fusion_tail_weight": sf_tail_weight,
                "score_fusion_head_frac": sf_head_frac,
                "tie_break": False,
                "rerank_full": False,
                "evaluator": "evaluate_m2m_fast",
                "query_batch_size": int(batch_size_eff),
            },
        }, f, indent=2)
    print(f"[eval_fast] metrics saved: {out_json_path}", flush=True)

    save_results = args.save_results or (str(out_json_path.with_suffix("")) + "_ranked.pkl")
    Path(save_results).parent.mkdir(parents=True, exist_ok=True)
    with open(save_results, "wb") as f:
        pickle.dump({"results": results, "qrels": qrels_eval}, f)
    print(f"[eval_fast] ranked results saved: {save_results}", flush=True)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
