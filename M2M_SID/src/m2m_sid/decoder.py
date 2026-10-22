import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration
from .score_fusion import add_geometric_score


class CodeTokenizer:
    def __init__(self, codebook_sizes, modality_tokens=0):
        self.codebook_sizes = [int(x) for x in codebook_sizes]
        self.modality_tokens = int(modality_tokens)
        self.pad_token_id = 0
        self.bos_token_id = 0
        self.eos_token_id = 1
        offset = 2
        self.modality_offset = offset
        offset += self.modality_tokens
        self.level_offsets = []
        for size in self.codebook_sizes:
            self.level_offsets.append(offset)
            offset += size
        self.vocab_size = offset

    def encode_codes(self, codes, modality=None):
        rows = []
        for row_index in range(codes.shape[0]):
            tokens = []
            if self.modality_tokens > 0:
                if modality is None:
                    raise ValueError("modality is required when modality tokens are enabled")
                value = int(modality[row_index])
                if value < 0 or value >= self.modality_tokens:
                    raise ValueError("modality index is outside the configured modality vocabulary")
                tokens.append(self.modality_offset + value)
            for level, code in enumerate(codes[row_index].tolist()):
                tokens.append(self.level_offsets[level] + int(code))
            tokens.append(self.eos_token_id)
            rows.append(tokens)
        return torch.tensor(rows, dtype=torch.long, device=codes.device)

    def level_for_step(self, step):
        if self.modality_tokens > 0:
            if step == 0:
                return None
            return step - 1
        return step

    def token_to_code(self, token, level):
        return int(token) - self.level_offsets[int(level)]

    def allowed_tokens_for_level(self, level):
        offset = self.level_offsets[int(level)]
        return list(range(offset, offset + self.codebook_sizes[int(level)]))


class PrefixTrie:
    def __init__(self):
        self.children = {}
        self.items = {}

    def insert(self, tokens, item_index):
        node = self.children
        prefix = []
        for token in tokens:
            token = int(token)
            prefix.append(token)
            node = node.setdefault(token, {})
        self.items.setdefault(tuple(prefix), []).append(int(item_index))

    def next_tokens(self, prefix):
        node = self.children
        for token in prefix:
            node = node.get(int(token), {})
        return list(node.keys())

    def items_for_prefix(self, prefix):
        return self.items.get(tuple(int(x) for x in prefix), [])


class PrefixRetriever(nn.Module):
    def __init__(self, model_name, embedding_dim, tokenizer, label_smoothing=0.0, candidate_aware_smoothing=False):
        super().__init__()
        self.tokenizer = tokenizer
        if str(model_name) == "t5-small":
            config = T5Config(vocab_size=tokenizer.vocab_size, d_model=512, d_ff=2048, num_layers=6, num_decoder_layers=6, num_heads=8, d_kv=64, dropout_rate=0.1, decoder_start_token_id=tokenizer.pad_token_id, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        else:
            config = T5Config(vocab_size=tokenizer.vocab_size, d_model=512, d_ff=2048, num_layers=6, num_decoder_layers=6, num_heads=8, d_kv=64, dropout_rate=0.1, decoder_start_token_id=tokenizer.pad_token_id, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        self.model = T5ForConditionalGeneration(config)
        self.query_projection = nn.Linear(int(embedding_dim), config.d_model)
        self.label_smoothing = float(label_smoothing)
        self.candidate_aware_smoothing = bool(candidate_aware_smoothing)

    def _smoothed_loss(self, lm_logits, labels):
        """Candidate-aware smoothed cross-entropy over SID tokens.

        When ``candidate_aware_smoothing`` is enabled we mix the one-hot
        teacher with the batch-wise empirical token distribution at each
        decoder step (see ``candidate_aware_label_distribution``). When
        only ``label_smoothing`` is set we fall back to uniform smoothing.
        If both are zero we delegate to the default T5 loss used by HF.
        """
        from .objectives import candidate_aware_label_distribution
        B, L, V = lm_logits.shape
        pad_id = int(self.tokenizer.pad_token_id)
        # Target distribution.
        if self.candidate_aware_smoothing:
            target = candidate_aware_label_distribution(labels, smoothing=self.label_smoothing, vocab_size=V)
        else:
            one_hot = F.one_hot(labels.clamp(min=0, max=V - 1), num_classes=V).to(dtype=lm_logits.dtype)
            uniform = torch.full_like(one_hot, 1.0 / float(V))
            target = (1.0 - self.label_smoothing) * one_hot + self.label_smoothing * uniform
            # zero pad positions
            pad_mask = (labels == pad_id) | (labels == -100)
            target = target * (~pad_mask).unsqueeze(-1).to(target.dtype)
        log_probs = F.log_softmax(lm_logits.to(torch.float32), dim=-1)
        pad_mask = (labels == pad_id) | (labels == -100)
        valid = (~pad_mask).to(dtype=log_probs.dtype)
        per_tok = -(target.to(log_probs.dtype) * log_probs).sum(dim=-1)
        return (per_tok * valid).sum() / valid.sum().clamp(min=1.0)

    def forward(self, query_embeddings, labels):
        # Accepts either (B, D) [single-view, PRO behaviour] or (B, V, D)
        # [M2M multi-view]. In the multi-view case we tile the labels V times
        # and treat each (view, label) pair as an independent training
        # example so each view head is pushed to generate the same SID.
        if query_embeddings.dim() == 3:
            batch, views, dim = query_embeddings.shape
            projected = self.query_projection(query_embeddings.reshape(batch * views, dim)).unsqueeze(1)
            tiled_labels = labels.unsqueeze(1).expand(-1, views, -1).reshape(batch * views, labels.shape[-1])
            attention_mask = torch.ones(projected.shape[:2], dtype=torch.long, device=projected.device)
            if self.candidate_aware_smoothing or self.label_smoothing > 0.0:
                outputs = self.model(inputs_embeds=projected, attention_mask=attention_mask, labels=tiled_labels)
                return self._smoothed_loss(outputs.logits, tiled_labels)
            return self.model(inputs_embeds=projected, attention_mask=attention_mask, labels=tiled_labels).loss
        inputs_embeds = self.query_projection(query_embeddings).unsqueeze(1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=query_embeddings.device)
        if self.candidate_aware_smoothing or self.label_smoothing > 0.0:
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            return self._smoothed_loss(outputs.logits, labels)
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss

    @torch.no_grad()
    def beam_search(self, query_embedding, trie, quantizer, beam_size, output_size, score_fusion_enabled=False, score_fusion_weight=0.0, score_fusion_normalize_bias=False):
        device = next(self.parameters()).device
        query_embedding = query_embedding.to(device)
        encoder_inputs = self.query_projection(query_embedding.unsqueeze(0)).unsqueeze(1)
        attention_mask_single = torch.ones((1, 1), dtype=torch.long, device=device)
        encoder_outputs = self.model.encoder(inputs_embeds=encoder_inputs, attention_mask=attention_mask_single)
        # cache encoder hidden states so we can expand it to batch for each step
        enc_hidden = encoder_outputs.last_hidden_state  # [1, 1, D]

        beams = [([], torch.zeros_like(query_embedding), 0.0)]
        total_steps = len(self.tokenizer.codebook_sizes) + (1 if self.tokenizer.modality_tokens > 0 else 0)
        bos = self.tokenizer.bos_token_id

        for step in range(total_steps):
            # filter beams that still have any next tokens
            active = []
            for prefix, reconstruction, score in beams:
                allowed = trie.next_tokens(prefix)
                if allowed:
                    active.append((prefix, reconstruction, score, allowed))
            if not active:
                break

            B = len(active)
            # build a padded decoder input batch: [B, 1 + step]
            #   note: all active prefixes at this step have identical length = step
            #   (because we run one token per step; every prefix grew from the same root)
            decoder_input = torch.tensor(
                [[bos] + a[0] for a in active], dtype=torch.long, device=device,
            )
            # expand encoder outputs to batch B
            enc_hidden_b = enc_hidden.expand(B, -1, -1)
            attn_b = torch.ones((B, 1), dtype=torch.long, device=device)
            # T5 forward for all beams in a single call
            output = self.model(
                encoder_outputs=(enc_hidden_b,),
                attention_mask=attn_b,
                decoder_input_ids=decoder_input,
                use_cache=False,
            )
            logits_all = F.log_softmax(output.logits[:, -1, :], dim=-1)  # [B, V]

            level = self.tokenizer.level_for_step(step)
            candidates = []
            for bi, (prefix, reconstruction, score, allowed) in enumerate(active):
                logits = logits_all[bi]
                if score_fusion_enabled and level is not None:
                    mapping = {token: self.tokenizer.token_to_code(token, level) for token in allowed}
                    logits = add_geometric_score(
                        logits, allowed, mapping, query_embedding, reconstruction,
                        quantizer.codebooks[int(level)], score_fusion_weight, score_fusion_normalize_bias,
                    )
                for token in allowed:
                    next_reconstruction = reconstruction
                    if level is not None:
                        code = self.tokenizer.token_to_code(token, level)
                        next_reconstruction = reconstruction + quantizer.codebooks[int(level)][int(code)].to(device)
                    candidates.append((prefix + [int(token)], next_reconstruction, score + float(logits[int(token)].item())))

            candidates.sort(key=lambda x: x[2], reverse=True)
            beams = candidates[: int(beam_size)]

        results = []
        for prefix, _, score in beams:
            for item in trie.items_for_prefix(prefix):
                results.append((item, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[: int(output_size)]
