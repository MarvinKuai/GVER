import argparse
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .config import load_config, require_path, set_seed, is_placeholder
from .data import load_torch_file, load_feature_store, load_pairs, load_modalities, PairEmbeddingDataset
from .quantizer import ResidualQuantizer
from .decoder import CodeTokenizer, PrefixRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--log_every", type=int, default=20)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 2023))
    paths = config["paths"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[retriever] device={device}", flush=True)
    print("[retriever] loading quantizer checkpoint...", flush=True)
    t0 = time.time()
    checkpoint = load_torch_file(require_path(paths["quantizer_checkpoint"], "quantizer_checkpoint"), map_location="cpu")
    quantizer_config = checkpoint["config"]
    quantizer = ResidualQuantizer(
        quantizer_config["embedding_dim"],
        quantizer_config["codebook_sizes"],
        quantizer_config.get("combiner_projection_dim", 2560),
        quantizer_config.get("combiner_hidden_dim", 5120),
        quantizer_config.get("kmeans_init", True),
        quantizer_config.get("kmeans_iters", 1000),
        quantizer_config.get("codebook_decay", 0.9),
        quantizer_config.get("ema_update", True),
        quantizer_config.get("threshold_ema_dead_code", 2),
    ).to(device)
    quantizer.load_state_dict(checkpoint["state_dict"])
    quantizer.eval()
    print(f"[retriever] quantizer loaded ({time.time()-t0:.1f}s)", flush=True)
    tokenizer = CodeTokenizer(quantizer_config["codebook_sizes"], quantizer_config.get("modality_tokens", 0))
    print(f"[retriever] building PrefixRetriever ({config['retriever']['model_name']})...", flush=True)
    t0 = time.time()
    retriever = PrefixRetriever(config["retriever"]["model_name"], config["retriever"]["embedding_dim"], tokenizer).to(device)
    print(f"[retriever] PrefixRetriever ready ({time.time()-t0:.1f}s)", flush=True)
    print("[retriever] loading train query embeddings...", flush=True)
    t0 = time.time()
    q = load_feature_store(require_path(paths["train_query_embeddings"], "train_query_embeddings"))
    print(f"[retriever] query embeddings loaded: n={len(q)} ({time.time()-t0:.1f}s)", flush=True)
    print("[retriever] loading train item embeddings...", flush=True)
    t0 = time.time()
    x = load_feature_store(require_path(paths["train_item_embeddings"], "train_item_embeddings"))
    print(f"[retriever] item embeddings loaded: n={len(x)} ({time.time()-t0:.1f}s)", flush=True)
    pair_path = paths.get("train_pairs")
    if pair_path is None or is_placeholder(pair_path):
        pair_path = paths.get("train_query_manifest")
    if pair_path is not None and is_placeholder(pair_path):
        pair_path = None
    if pair_path is None and (q.id_to_index is not None or x.id_to_index is not None):
        raise ValueError("Set train_pairs or train_query_manifest when using embedding dictionaries with id_to_index")
    pairs = load_pairs(pair_path, min(len(q), len(x)))
    modality_path = paths.get("train_item_modalities")
    if modality_path is None or is_placeholder(modality_path):
        modality_path = paths.get("train_item_manifest")
    if modality_path is not None and is_placeholder(modality_path):
        modality_path = None
    item_modalities = None if modality_path is None else load_modalities(modality_path, len(x))
    dataset = PairEmbeddingDataset(q, x, pairs, item_modalities)
    loader = DataLoader(dataset, batch_size=int(config["retriever"]["batch_size"]), shuffle=True, drop_last=True)
    total_epochs = int(config["retriever"]["epochs"])
    print(f"[retriever] pairs={len(pairs)}, batch_size={int(config['retriever']['batch_size'])}, steps_per_epoch={len(loader)}, epochs={total_epochs}", flush=True)
    optimizer = torch.optim.AdamW(retriever.parameters(), lr=float(config["retriever"]["learning_rate"]))
    log_every = max(1, int(args.log_every))
    for epoch in range(total_epochs):
        retriever.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        epoch_start = time.time()
        for batch_index, (query, item, _, modality) in enumerate(loader):
            modality = modality.to(device)
            with torch.no_grad():
                codes, _, _ = quantizer.encode(item)
                query_embedding = quantizer.encode_features(query)
                labels = tokenizer.encode_codes(codes, modality if tokenizer.modality_tokens > 0 else None)
            loss = retriever(query_embedding, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().item())
            epoch_loss_sum += loss_value
            epoch_batches += 1
            if (batch_index + 1) % log_every == 0:
                print(f"[retriever] epoch {epoch+1}/{total_epochs} batch {batch_index+1}/{len(loader)} loss={loss_value:.4f} elapsed={time.time()-epoch_start:.1f}s", flush=True)
        avg_loss = epoch_loss_sum / max(1, epoch_batches)
        print(f"[retriever] epoch {epoch+1}/{total_epochs} done avg_loss={avg_loss:.4f} time={time.time()-epoch_start:.1f}s", flush=True)
    output = require_path(paths["retriever_checkpoint"], "retriever_checkpoint")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": retriever.state_dict(), "config": config["retriever"], "tokenizer": {"codebook_sizes": tokenizer.codebook_sizes, "modality_tokens": tokenizer.modality_tokens}}, output)
    print(f"[retriever] checkpoint saved: {output}", flush=True)


if __name__ == "__main__":
    main()
