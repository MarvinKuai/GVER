import argparse
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .config import load_config, require_path, set_seed, is_placeholder
from .data import load_feature_store, load_pairs, PairEmbeddingDataset
from .quantizer import ResidualQuantizer
from .objectives import contrastive_loss, reconstruction_alignment_loss, prefix_ranking_distillation


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
    ).to(device)
    print(f"[quantizer] quantizer ready ({time.time()-t0:.1f}s)", flush=True)
    optimizer = torch.optim.AdamW(quantizer.parameters(), lr=float(config["quantizer"]["learning_rate"]))
    levels = config["quantizer"].get("prefix_ranking_levels", [])
    accumulation_steps = int(config["quantizer"].get("gradient_accumulation_steps", 1))
    total_steps = max(1, len(loader) * int(config["quantizer"]["epochs"]) // max(1, accumulation_steps))
    warmup_steps = int(total_steps * float(config["quantizer"].get("prefix_ranking_warmup_fraction", 0.0)))
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
            query_embedding = quantizer.encode_features(query)
            item_embedding = quantizer.encode_features(item)
            batch_size = int(query_embedding.shape[0])
            _, joint_reconstruction, joint_prefixes, rq_loss = quantizer.quantize_train(torch.cat((query_embedding, item_embedding), dim=0))
            query_reconstruction = joint_reconstruction[:batch_size]
            item_reconstruction = joint_reconstruction[batch_size:]
            item_prefixes = [prefix[batch_size:] for prefix in joint_prefixes]
            loss = contrastive_loss(query_embedding, item_embedding)
            loss = loss + float(config["quantizer"]["rq_loss_weight"]) * rq_loss
            loss = loss + float(config["quantizer"]["mse_loss_weight"]) * reconstruction_alignment_loss(F.normalize(query_reconstruction, dim=-1), F.normalize(item_reconstruction, dim=-1))
            if warmup_steps > 0:
                rank_weight = float(config["quantizer"]["prefix_ranking_weight"]) * min(1.0, float(step) / float(warmup_steps))
            else:
                rank_weight = float(config["quantizer"]["prefix_ranking_weight"])
            loss = loss + rank_weight * prefix_ranking_distillation(query_embedding, item_embedding, item_prefixes, levels, config["quantizer"]["prefix_ranking_temperature"], config["quantizer"]["prefix_ranking_topk"])
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
