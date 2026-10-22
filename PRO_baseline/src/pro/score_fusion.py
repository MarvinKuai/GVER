import torch


def geometric_score_bias(query_embedding, prefix_reconstruction, codebook, normalize=False, eps=1e-6):
    residual = query_embedding - prefix_reconstruction
    bias = 2.0 * residual @ codebook.t() - codebook.pow(2).sum(dim=1)
    if normalize:
        bias = bias - bias.mean(dim=-1, keepdim=True)
        bias = bias / bias.std(dim=-1, keepdim=True, unbiased=False).clamp_min(float(eps))
    return bias


def add_geometric_score(logits, allowed_tokens, token_to_code, query_embedding, prefix_reconstruction, codebook, weight, normalize=False):
    bias = geometric_score_bias(query_embedding, prefix_reconstruction, codebook, normalize)
    selected_bias = []
    for token in allowed_tokens:
        selected_bias.append(bias[int(token_to_code[int(token)])])
    if selected_bias:
        logits = logits.clone()
        logits[allowed_tokens] = logits[allowed_tokens] + float(weight) * torch.stack(selected_bias).to(logits.device)
    return logits
