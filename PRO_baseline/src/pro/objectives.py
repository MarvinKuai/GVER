import torch
import torch.nn.functional as F


def contrastive_loss(query_embeddings, item_embeddings, temperature=0.01):
    query_embeddings = F.normalize(query_embeddings, dim=-1)
    item_embeddings = F.normalize(item_embeddings, dim=-1)
    logits = query_embeddings @ item_embeddings.t() / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def reconstruction_alignment_loss(query_reconstructions, item_reconstructions):
    return F.mse_loss(query_reconstructions, item_reconstructions)


def prefix_ranking_distillation(query_embeddings, item_embeddings, item_prefixes, levels, temperature=0.05, topk=128):
    if not levels:
        return query_embeddings.new_zeros(())
    query_embeddings = F.normalize(query_embeddings, dim=-1)
    item_embeddings = F.normalize(item_embeddings, dim=-1)
    teacher_scores = query_embeddings @ item_embeddings.t()
    k = min(int(topk), teacher_scores.shape[1])
    selected = torch.topk(teacher_scores, k=k, dim=1).indices
    teacher_selected = teacher_scores.gather(1, selected)
    teacher_distribution = F.softmax(teacher_selected / float(temperature), dim=1).detach()
    losses = []
    for level in levels:
        prefix = F.normalize(item_prefixes[int(level) - 1], dim=-1)
        student_scores = query_embeddings @ prefix.t()
        student_selected = student_scores.gather(1, selected)
        student_log_distribution = F.log_softmax(student_selected / float(temperature), dim=1)
        losses.append(F.kl_div(student_log_distribution, teacher_distribution, reduction="batchmean") * float(temperature) ** 2)
    return torch.stack(losses).mean()
