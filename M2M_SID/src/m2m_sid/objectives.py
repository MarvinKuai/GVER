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


def candidate_aware_label_distribution(
    labels,
    smoothing=0.1,
    vocab_size=None,
):
    """Build a soft target distribution for teacher-forced T5 training.

    For each (sample, step) we produce a mixture:

        (1 - smoothing) * one_hot(labels[sample, step])
      +      smoothing  * batch_frequency_at_step(labels[:, step])

    where ``batch_frequency_at_step`` is the empirical token distribution of
    all other samples in the batch at the same decoder step. This explicitly
    tells the model *where the batch is likely to disagree* and provides a
    second-order teacher signal exactly on the ambiguous SID branches, while
    all unambiguous positions degenerate to (1-eps) * onehot (same as plain
    label smoothing with eps -> 0 for those positions).

    Returns a tensor of shape ``[B, L, V]`` summing to 1 along the last
    axis. Rows for pad positions (labels == -100 or labels == 0 for T5) are
    left as uniform zero vectors so the caller can mask them out via the
    attention/label mask.
    """
    if labels.dim() != 2:
        raise ValueError("candidate_aware_label_distribution expects labels of shape [B, L]")
    B, L = labels.shape
    V = int(vocab_size)
    smoothing = float(smoothing)
    device = labels.device
    pad_mask = (labels == -100) | (labels == 0)  # pad / BOS positions
    safe_labels = labels.masked_fill(pad_mask, 0).clamp(min=0, max=V - 1)
    # One-hot term.
    one_hot = F.one_hot(safe_labels, num_classes=V).to(dtype=torch.float32)   # [B, L, V]
    # Empirical frequency at each step over the current mini-batch (excluding
    # pads). Shape: [L, V].
    valid_mask = (~pad_mask).to(dtype=torch.float32)                          # [B, L]
    step_counts = torch.zeros(L, V, device=device, dtype=torch.float32)
    step_counts.scatter_add_(1, safe_labels.t(), valid_mask.t())
    # Normalise per step; empty steps fall back to one_hot.
    step_totals = step_counts.sum(dim=-1, keepdim=True).clamp(min=1.0)
    step_freq = step_counts / step_totals                                     # [L, V]
    step_freq = step_freq.unsqueeze(0).expand(B, L, V)                        # [B, L, V]
    soft = (1.0 - smoothing) * one_hot + smoothing * step_freq
    # Zero out pad positions so the loss can mask them.
    soft = soft * valid_mask.unsqueeze(-1)
    return soft


def neighbor_prefix_consistency_loss(
    item_embeddings,
    item_prefixes,
    prefix_levels,
    num_neighbors=4,
    sim_threshold=0.85,
    temperature=0.05,
):
    """Neighbour Prefix *Consistency* loss (shallow pull).

    Hypothesis (design):
        The soft-margin discriminative codebook loss hurts R@50 on INFOSEEK
        because entity-level neighbours in CLIP space are the *right* answer
        for many queries. Pushing their SIDs apart shrinks the recall surface
        of trie beam search. The correct SID philosophy is the opposite:
        truly similar items SHOULD share early-prefix codes so a single
        beam path can retrieve all of them (R@50 up); the discriminator is
        the retriever (CE loss + candidate-aware smoothing), not the
        codebook.

    Mechanics:
        For each anchor i, we locate its top-k true confusers (cos > thr),
        and at every shallow prefix level ``l`` we **minimise** the
        divergence between the anchor's own prefix ``p_i^{(l)}`` and the
        neighbour's prefix ``p_j^{(l)}`` -- i.e. encourage them to end up in
        the same codebook cell. This is exactly the opposite sign of
        ``discriminative_codebook_loss``.

    Loss form (per surviving pair):
        1 - cos(p_i^{(l)}, p_j^{(l)})

    which lies in [0, 2] and is zero iff the two prefixes point in the same
    direction.

    ``prefix_levels`` should list the SHALLOW levels where sharing is
    desirable (e.g. [1, 2, 3, 4] for a 16-level SID). ``temperature`` is
    accepted for config symmetry but currently unused.
    """
    del temperature  # kept for config symmetry
    if not prefix_levels:
        return item_embeddings.new_zeros(())
    batch_size = int(item_embeddings.shape[0])
    k = int(num_neighbors)
    if batch_size <= 1 or k <= 0:
        return item_embeddings.new_zeros(())
    k = min(k, batch_size - 1)
    anchors = F.normalize(item_embeddings, dim=-1)
    similarity = anchors @ anchors.t()
    diag_mask = torch.eye(batch_size, device=similarity.device, dtype=torch.bool)
    masked_similarity = similarity.masked_fill(diag_mask, float("-inf"))
    nbr_sim, nbr_idx = torch.topk(masked_similarity, k=k, dim=1)
    confuser_mask = (nbr_sim >= float(sim_threshold)).float()
    total_pairs = confuser_mask.sum()
    if float(total_pairs.item()) <= 0.0:
        return item_embeddings.new_zeros(())
    losses = []
    for level in prefix_levels:
        prefix = F.normalize(item_prefixes[int(level) - 1], dim=-1)
        own = prefix                       # (B, D)
        nbr_prefix = prefix[nbr_idx]       # (B, k, D)
        # cos(own_i, nbr_prefix_j) summed over k neighbours of each anchor
        cos_pairs = torch.einsum("bd,bkd->bk", own, nbr_prefix)  # (B, k)
        divergence = 1.0 - cos_pairs                             # (B, k)
        losses.append((divergence * confuser_mask).sum() / total_pairs.clamp(min=1.0))
    return torch.stack(losses).mean()


def hard_negative_contrastive_loss(
    item_embeddings,
    item_prefixes,
    prefix_levels,
    num_hard_negatives=16,
    sim_low=0.60,
    sim_high=0.85,
    temperature=0.07,
):
    """Hard-Negative Contrastive loss at DEEP prefix levels.

    Companion to ``neighbor_prefix_consistency_loss`` (NPC) which pulls
    *true* neighbours (cos>=0.85) together at SHALLOW prefix levels. This
    loss operates on the opposite end: at the DEEPEST prefix levels
    (e.g. [13,14,15,16] for a 16-level SID), it pushes AWAY the
    *medium-similarity* confusers (0.60 <= cos < 0.85) — items that look
    similar in CLIP space but belong to different entities. Those are
    precisely the pairs that hurt R@1 on INFOSEEK (same beam path, wrong
    answer at the leaf).

    Why "mid-band" sim and not "all non-positives":
        - cos >= sim_high are TRUE neighbours (handled by NPC, must stay
          together even at the leaf).
        - cos < sim_low are easy negatives, random noise; pushing them
          costs capacity and hurts R@50.
        - sim_low <= cos < sim_high is the sweet spot: pairs the retriever
          actually confuses at top-1.

    Formulation (InfoNCE on deep prefixes):
        For each anchor i, build a logit vector over the whole batch,
        where positives are i itself and *true* neighbours (cos>=sim_high),
        and the "others" set includes every HARD NEGATIVE (sim_low <= cos
        < sim_high) with the rest masked out of the denominator. We then
        pay an InfoNCE loss computed with the anchor's CLIP embedding as
        the query and the prefix vectors as keys.

    This is numerically zero in two cases: (a) no pair in the batch falls
    into the mid band, (b) ``prefix_levels`` is empty.
    """
    if not prefix_levels:
        return item_embeddings.new_zeros(())
    batch_size = int(item_embeddings.shape[0])
    if batch_size <= 1:
        return item_embeddings.new_zeros(())
    anchors = F.normalize(item_embeddings, dim=-1)
    similarity = anchors @ anchors.t()                                # (B, B)
    eye = torch.eye(batch_size, device=similarity.device, dtype=torch.bool)
    # Hard negatives: strictly within the mid band and not self.
    hard_mask = ((similarity >= float(sim_low)) &
                 (similarity <  float(sim_high)) &
                 (~eye))                                              # (B, B) bool
    # True positives for the denominator cushion: self + true neighbours.
    pos_mask = (similarity >= float(sim_high)) | eye                  # (B, B) bool
    # Cap the number of hard negatives per anchor to ``num_hard_negatives``.
    if int(num_hard_negatives) > 0 and int(num_hard_negatives) < batch_size:
        # Among the hard set we keep the top-num_hard_negatives by similarity
        # per anchor; this stops a single query batch from being dominated by
        # very-dense clusters.
        hard_sim = similarity.masked_fill(~hard_mask, float("-inf"))
        topk_vals, topk_idx = torch.topk(hard_sim, k=int(num_hard_negatives), dim=1)
        keep_mask = torch.zeros_like(hard_mask)
        valid_rows = (topk_vals > float("-inf"))
        keep_mask.scatter_(1, topk_idx, valid_rows)
        hard_mask = hard_mask & keep_mask
    # Anchors that still have at least one hard negative contribute.
    row_has_hn = hard_mask.any(dim=1)                                 # (B,)
    if not bool(row_has_hn.any().item()):
        return item_embeddings.new_zeros(())
    losses = []
    for level in prefix_levels:
        prefix = F.normalize(item_prefixes[int(level) - 1], dim=-1)   # (B, D)
        # Logits for every (anchor, candidate_prefix): higher = more aligned.
        logits = (anchors @ prefix.t()) / float(temperature)          # (B, B)
        # In-batch InfoNCE where numerator = sum over positives (self + true
        # neighbours), denominator = positives + hard negatives.
        mask_for_softmax = pos_mask | hard_mask                       # (B, B) bool
        logits = logits.masked_fill(~mask_for_softmax, float("-inf"))
        # log-sum-exp over the kept columns (positives + hard negatives).
        log_denom = torch.logsumexp(logits, dim=1)                    # (B,)
        # Numerator: logsumexp over positives only.
        logits_pos = logits.masked_fill(~pos_mask, float("-inf"))
        log_num = torch.logsumexp(logits_pos, dim=1)                  # (B,)
        per_row = log_denom - log_num                                 # (B,) >= 0
        # Only count rows that actually have at least one hard negative;
        # otherwise the loss is trivially 0 for that row and shouldn't drag
        # the mean down.
        row_valid = row_has_hn.float()
        total = row_valid.sum().clamp(min=1.0)
        losses.append((per_row * row_valid).sum() / total)
    return torch.stack(losses).mean()
