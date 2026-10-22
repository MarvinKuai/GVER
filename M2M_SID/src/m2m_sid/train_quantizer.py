import argparse
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .config import load_config, require_path, set_seed, is_placeholder
from .data import load_feature_store, load_pairs, PairEmbeddingDataset
from .quantizer import ResidualQuantizer
from .objectives import contrastive_loss, reconstruction_alignment_loss, prefix_ranking_distillation, neighbor_prefix_consistency_loss, hard_negative_contrastive_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--log_every", type=int, default=20)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 2023))
    paths = config["paths"]
    print("[quantizer] loading query embeddings...", flush=True)
    t0 = time.time()
    q = load_feature_store(require_path(paths["train_query_embeddings"], "train_query_embeddings"))
    print(f"[quantizer] query embeddings loaded: n={len(q)} ({time.time()-t0:.1f}s)", flush=True)
    print("[quantizer] loading item embeddings...", flush=True)
    t0 = time.time()
    x = load_feature_store(require_path(paths["train_item_embeddings"], "train_item_embeddings"))
    print(f"[quantizer] item embeddings loaded: n={len(x)} ({time.time()-t0:.1f}s)", flush=True)
    pair_path = paths.get("train_pairs")
    if pair_path is None or is_placeholder(pair_path):
        pair_path = paths.get("train_query_manifest")
    if pair_path is not None and is_placeholder(pair_path):
        pair_path = None
    if pair_path is None and (q.id_to_index is not None or x.id_to_index is not None):
        raise ValueError("Set train_pairs or train_query_manifest when using embedding dictionaries with id_to_index")
    pairs = load_pairs(pair_path, min(len(q), len(x)))
    print(f"[quantizer] pairs loaded: {len(pairs)}", flush=True)
    dataset = PairEmbeddingDataset(q, x, pairs)
    loader = DataLoader(dataset, batch_size=int(config["quantizer"]["batch_size"]), shuffle=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[quantizer] device={device}, batch_size={int(config['quantizer']['batch_size'])}, steps_per_epoch={len(loader)}, epochs={int(config['quantizer']['epochs'])}", flush=True)
    print("[quantizer] initializing ResidualQuantizer (kmeans init may take a while)...", flush=True)
    t0 = time.time()
    quantizer = ResidualQuantizer(
        config["quantizer"]["embedding_dim"],
        config["quantizer"]["codebook_sizes"],
        config["quantizer"].get("combiner_projection_dim", 2560),
        config["quantizer"].get("combiner_hidden_dim", 5120),
        config["quantizer"].get("kmeans_init", True),
        config["quantizer"].get("kmeans_iters", 1000),
        config["quantizer"].get("codebook_decay", 0.9),
        config["quantizer"].get("ema_update", True),
        config["quantizer"].get("threshold_ema_dead_code", 2),
        config["quantizer"].get("n_views", 1),
    ).to(device)
    print(f"[quantizer] quantizer ready ({time.time()-t0:.1f}s)", flush=True)
    optimizer = torch.optim.AdamW(quantizer.parameters(), lr=float(config["quantizer"]["learning_rate"]))
    levels = config["quantizer"].get("prefix_ranking_levels", [])
    accumulation_steps = int(config["quantizer"].get("gradient_accumulation_steps", 1))
    total_steps = max(1, len(loader) * int(config["quantizer"]["epochs"]) // max(1, accumulation_steps))
    warmup_steps = int(total_steps * float(config["quantizer"].get("prefix_ranking_warmup_fraction", 0.0)))
    n_views = int(getattr(quantizer, "n_views", 1))
    view_loss_weight = float(config["quantizer"].get("view_contrastive_weight", 1.0))
    print(f"[quantizer] n_views={n_views}, view_contrastive_weight={view_loss_weight}", flush=True)
    # Neighbour Prefix Consistency Loss: pulls true neighbours (cos>=0.85)
    # into the same shallow prefix cells. Setting weight=0 disables it.
    npc_weight_cfg = float(config["quantizer"].get("neighbor_consistency_weight", 0.0))
    npc_levels = list(config["quantizer"].get("neighbor_consistency_levels", []))
    npc_neighbors = int(config["quantizer"].get("neighbor_consistency_neighbors", 4))
    npc_sim_threshold = float(config["quantizer"].get("neighbor_consistency_sim_threshold", 0.85))
    npc_warmup_fraction = float(config["quantizer"].get("neighbor_consistency_warmup_fraction", 0.0))
    npc_warmup_steps = int(total_steps * npc_warmup_fraction)
    print(f"[quantizer] neighbor_prefix_consistency: weight={npc_weight_cfg} levels={npc_levels} neighbors={npc_neighbors} sim_threshold={npc_sim_threshold} warmup_steps={npc_warmup_steps}", flush=True)
    # Hard-Negative Contrastive Loss: pushes mid-similarity confusers
    # apart at DEEP prefix levels. Complements NPC (shallow pull). weight=0
    # keeps baseline behaviour.
    hn_weight_cfg = float(config["quantizer"].get("hard_neg_contrastive_weight", 0.0))
    hn_levels = list(config["quantizer"].get("hard_neg_contrastive_levels", []))
    hn_num = int(config["quantizer"].get("hard_neg_contrastive_num", 16))
    hn_sim_low = float(config["quantizer"].get("hard_neg_contrastive_sim_low", 0.60))
    hn_sim_high = float(config["quantizer"].get("hard_neg_contrastive_sim_high", 0.85))
    hn_temperature = float(config["quantizer"].get("hard_neg_contrastive_temperature", 0.07))
    hn_warmup_fraction = float(config["quantizer"].get("hard_neg_contrastive_warmup_fraction", 0.0))
    hn_warmup_steps = int(total_steps * hn_warmup_fraction)
    print(f"[quantizer] hard_neg_contrastive: weight={hn_weight_cfg} levels={hn_levels} num={hn_num} sim_low={hn_sim_low} sim_high={hn_sim_high} temperature={hn_temperature} warmup_steps={hn_warmup_steps}", flush=True)
    step = 0
    total_epochs = int(config["quantizer"]["epochs"])
    log_every = max(1, int(args.log_every))
    for epoch in range(total_epochs):
        quantizer.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        epoch_batches = 0
        epoch_start = time.time()
        for batch_index, (query, item, _, _) in enumerate(loader):
            raw_query_embedding = quantizer.encode_features(query)
            raw_item_embedding = quantizer.encode_features(item)
            # Keep item side single-view (view 0) so the shared codebook / trie
            # structure stays identical to PRO. Only the query side exposes
            # multiple views for the trie-constrained beam search at inference.
            if raw_item_embedding.dim() == 3:
                item_embedding = raw_item_embedding[:, 0, :]
            else:
                item_embedding = raw_item_embedding
            if raw_query_embedding.dim() == 3:
                query_views = raw_query_embedding
                query_embedding = raw_query_embedding[:, 0, :]
            else:
                query_views = raw_query_embedding.unsqueeze(1)
                query_embedding = raw_query_embedding
            batch_size = int(query_embedding.shape[0])
            _, joint_reconstruction, joint_prefixes, rq_loss = quantizer.quantize_train(torch.cat((query_embedding, item_embedding), dim=0))
            query_reconstruction = joint_reconstruction[:batch_size]
            item_reconstruction = joint_reconstruction[batch_size:]
            item_prefixes = [prefix[batch_size:] for prefix in joint_prefixes]
            loss = contrastive_loss(query_embedding, item_embedding)
            # Extra per-view contrastive loss (K>1 only): each additional view
            # of the query must still align with the single item view. This
            # pushes the K query heads to spread into complementary modes
            # while all remaining tied to the same codebook.
            if query_views.shape[1] > 1:
                view_loss = 0.0
                for v in range(1, query_views.shape[1]):
                    view_loss = view_loss + contrastive_loss(query_views[:, v, :], item_embedding)
                loss = loss + view_loss_weight * (view_loss / float(query_views.shape[1] - 1))
            loss = loss + float(config["quantizer"]["rq_loss_weight"]) * rq_loss
            loss = loss + float(config["quantizer"]["mse_loss_weight"]) * reconstruction_alignment_loss(F.normalize(query_reconstruction, dim=-1), F.normalize(item_reconstruction, dim=-1))
            if warmup_steps > 0:
                rank_weight = float(config["quantizer"]["prefix_ranking_weight"]) * min(1.0, float(step) / float(warmup_steps))
            else:
                rank_weight = float(config["quantizer"]["prefix_ranking_weight"])
            loss = loss + rank_weight * prefix_ranking_distillation(query_embedding, item_embedding, item_prefixes, levels, config["quantizer"]["prefix_ranking_temperature"], config["quantizer"]["prefix_ranking_topk"])
            if npc_weight_cfg > 0.0 and npc_levels:
                if npc_warmup_steps > 0:
                    npc_weight = npc_weight_cfg * min(1.0, float(step) / float(npc_warmup_steps))
                else:
                    npc_weight = npc_weight_cfg
                loss = loss + npc_weight * neighbor_prefix_consistency_loss(
                    item_embedding, item_prefixes, npc_levels,
                    num_neighbors=npc_neighbors, sim_threshold=npc_sim_threshold,
                )
            if hn_weight_cfg > 0.0 and hn_levels:
                if hn_warmup_steps > 0:
                    hn_weight = hn_weight_cfg * min(1.0, float(step) / float(hn_warmup_steps))
                else:
                    hn_weight = hn_weight_cfg
                loss = loss + hn_weight * hard_negative_contrastive_loss(
                    item_embedding, item_prefixes, hn_levels,
                    num_hard_negatives=hn_num,
                    sim_low=hn_sim_low, sim_high=hn_sim_high,
                    temperature=hn_temperature,
                )
            loss = loss / max(1, accumulation_steps)
            loss.backward()
            loss_value = float(loss.detach().item()) * max(1, accumulation_steps)
            epoch_loss_sum += loss_value
            epoch_batches += 1
            if (batch_index + 1) % max(1, accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            if (batch_index + 1) % log_every == 0:
                print(f"[quantizer] epoch {epoch+1}/{total_epochs} batch {batch_index+1}/{len(loader)} loss={loss_value:.4f} elapsed={time.time()-epoch_start:.1f}s", flush=True)
        if len(loader) % max(1, accumulation_steps) != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
        avg_loss = epoch_loss_sum / max(1, epoch_batches)
        print(f"[quantizer] epoch {epoch+1}/{total_epochs} done avg_loss={avg_loss:.4f} time={time.time()-epoch_start:.1f}s", flush=True)
    output = require_path(paths["quantizer_checkpoint"], "quantizer_checkpoint")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": quantizer.state_dict(), "config": config["quantizer"]}, output)
    print(f"[quantizer] checkpoint saved: {output}", flush=True)


if __name__ == "__main__":
    main()
