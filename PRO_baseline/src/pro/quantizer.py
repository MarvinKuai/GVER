import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ


class FeatureCombiner(nn.Module):
    def __init__(self, feature_dim, projection_dim=2560, hidden_dim=5120, drop_rate=0.0):
        super().__init__()
        feature_dim = int(feature_dim)
        projection_dim = int(projection_dim)
        hidden_dim = int(hidden_dim)
        self.text_projection_layer = nn.Linear(feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(feature_dim, projection_dim)
        self.dropout1 = nn.Dropout(float(drop_rate))
        self.dropout2 = nn.Dropout(float(drop_rate))
        self.dropout3 = nn.Dropout(float(drop_rate))
        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, feature_dim)
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(drop_rate)),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.out_projection_layer = nn.Identity()

    def forward(self, image_features, text_features, image_masks=None, text_masks=None):
        if image_masks is not None:
            image_features = image_features * image_masks
        if text_masks is not None:
            text_features = text_features * text_masks
        image_projected = self.dropout2(F.relu(self.image_projection_layer(image_features)))
        text_projected = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        raw_combined = torch.cat((text_projected, image_projected), dim=-1)
        combined = self.dropout3(F.relu(self.combiner_layer(raw_combined)))
        scalar = self.dynamic_scalar(raw_combined)
        output = self.output_layer(combined) + scalar * text_features + (1.0 - scalar) * image_features
        return self.out_projection_layer(output)


class ResidualQuantizer(nn.Module):
    def __init__(self, embedding_dim, codebook_sizes, combiner_projection_dim=2560, combiner_hidden_dim=5120, kmeans_init=True, kmeans_iters=1000, codebook_decay=0.9, ema_update=True, threshold_ema_dead_code=2):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.codebook_sizes = [int(x) for x in codebook_sizes]
        self.levels = len(self.codebook_sizes)
        self.encoder = FeatureCombiner(self.embedding_dim, combiner_projection_dim, combiner_hidden_dim)
        self.residual_rq = ResidualVQ(
            dim=self.embedding_dim,
            codebook_dim=self.embedding_dim,
            num_quantizers=self.levels,
            codebook_size=tuple(self.codebook_sizes),
            kmeans_init=bool(kmeans_init),
            kmeans_iters=int(kmeans_iters),
            use_cosine_sim=False,
            learnable_codebook=False,
            ema_update=bool(ema_update),
            threshold_ema_dead_code=int(threshold_ema_dead_code),
            decay=float(codebook_decay),
        )

    @property
    def codebooks(self):
        values = []
        for level, layer in enumerate(self.residual_rq.layers):
            embed = layer._codebook.embed.squeeze(0)
            values.append(embed[: self.codebook_sizes[level]])
        return values

    def encode_features(self, features):
        if isinstance(features, dict):
            device = next(self.parameters()).device
            image = features["img"].to(device=device, dtype=torch.float32)
            text = features["text"].to(device=device, dtype=torch.float32)
            image_mask = features["img_mask"].to(device=device, dtype=torch.float32).view(-1, 1)
            text_mask = features["text_mask"].to(device=device, dtype=torch.float32).view(-1, 1)
            return F.normalize(self.encoder(image, text, image_mask, text_mask), dim=-1)
        tensor = features.to(device=next(self.parameters()).device, dtype=torch.float32)
        return tensor.unsqueeze(0) if tensor.dim() == 1 else tensor

    def _prefix_reconstructions(self, codes):
        reconstructions = []
        current = None
        for level, codebook in enumerate(self.codebooks):
            code = codes[:, level].clamp(min=0, max=codebook.shape[0] - 1)
            vector = codebook.index_select(0, code)
            current = vector if current is None else current + vector
            reconstructions.append(current)
        return reconstructions

    def _run_rq(self, embeddings):
        x = self.residual_rq.project_in(embeddings.unsqueeze(0))
        residual = x
        quantized_out = torch.zeros_like(x)
        all_codes = []
        all_losses = []
        reconstructions = []
        grad_fraction = float(getattr(self.residual_rq, "quant_grad_frac", 0.0) or 0.0)
        for layer in self.residual_rq.layers:
            quantized, codes, loss = layer(
                residual,
                indices=None,
                sample_codebook_temp=None,
                freeze_codebook=False,
                codebook_transform_fn=None,
                topk=None,
            )
            if grad_fraction <= 0.0:
                residual = residual - quantized.detach()
            elif grad_fraction >= 1.0:
                residual = residual - quantized
            else:
                residual = residual - (quantized.detach() * (1.0 - grad_fraction) + quantized * grad_fraction)
            quantized_out = quantized_out + quantized
            all_codes.append(codes)
            all_losses.append(loss.reshape(()))
            reconstructions.append(self.residual_rq.project_out(quantized_out).squeeze(0))
        quantized_out = self.residual_rq.project_out(quantized_out).squeeze(0)
        codes = torch.stack(all_codes, dim=-1).squeeze(0)
        loss = torch.stack(all_losses).mean()
        return codes, quantized_out, reconstructions, loss

    def quantize(self, embeddings):
        with torch.no_grad():
            codes, quantized, reconstructions, _ = self._run_rq(embeddings)
        return codes, quantized, reconstructions

    def quantize_by_nearest_code(self, embeddings):
        residual = embeddings
        codes = []
        reconstructions = []
        current = torch.zeros_like(embeddings)
        for codebook in self.codebooks:
            distance = torch.cdist(residual, codebook).pow(2)
            code = distance.argmin(dim=1)
            vector = codebook.index_select(0, code)
            current = current + vector
            residual = residual - vector
            codes.append(code)
            reconstructions.append(current)
        return torch.stack(codes, dim=1), current, reconstructions

    def encode(self, features):
        return self.quantize(self.encode_features(features))

    def quantize_train(self, embeddings):
        return self._run_rq(embeddings)

    def forward(self, features):
        return self.quantize_train(self.encode_features(features))

    def codebooks_as_tensors(self):
        return [x.detach() for x in self.codebooks]
