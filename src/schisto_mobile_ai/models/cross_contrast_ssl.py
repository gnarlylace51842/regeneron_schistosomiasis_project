"""Cross-contrast self-supervised learning for schistosomiasis microscopy.

Core idea: brightfield and darkfield images of the SAME slide are a naturally
occurring positive pair. BF measures light absorption; DF measures light
scattering. A shared encoder trained to align these two views of the same
biological sample must learn representations that are stable across optical
regimes — which forces it toward content-based (egg presence / morphology)
rather than illumination-based features.

This is physically motivated, not heuristic. Schistosome eggs have a distinctive
absorption+scattering signature that tissue debris lacks. Cross-contrast alignment
is therefore a harder and more informative pre-training task than within-contrast
augmentation (random crop, color jitter).

Architecture:
    Encoder f  : image → feature vector (shared TinyConvEncoder)
    Projector g : feature → L2-normalized embedding (2-layer MLP)

Training signal (NT-Xent / SimCLR loss):
    Positive pairs : (BF_i, DF_i)  — same slide
    Negative pairs : all other cross-image combinations within the batch

After pre-training, the projector is discarded. The encoder weights are used
to initialise supervised fine-tuning classifiers (single-contrast or dual).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from schisto_mobile_ai.models.simple_cnn import TinyConvEncoder


class ProjectionHead(nn.Module):
    """Two-layer MLP that maps encoder features to a normalised SSL embedding.

    Following SimCLR (Chen et al., 2020): hidden dim = feature_dim, output dim
    is configurable (128 by default). The output is L2-normalised so that the
    NT-Xent loss reduces to a cosine-similarity objective.
    """

    def __init__(self, *, feature_dim: int, projection_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.net(features)
        return F.normalize(projected, dim=1)


class CrossContrastSSLModel(nn.Module):
    """Shared encoder + projection head for cross-contrast contrastive pre-training.

    Both BF and DF images pass through the SAME encoder. The shared weights are
    forced to learn physics-invariant (illumination-invariant) representations.
    This is the key architectural choice: a separate encoder for each contrast
    could learn contrast-specific features; sharing forces content alignment.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        base_channels: int = 32,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = TinyConvEncoder(in_channels=in_channels, base_channels=base_channels)
        self.projector = ProjectionHead(
            feature_dim=self.encoder.feature_dim,
            projection_dim=projection_dim,
        )

    @property
    def feature_dim(self) -> int:
        return self.encoder.feature_dim

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Return raw encoder features (used for fine-tuning, not pre-training)."""
        return self.encoder(image)

    def project(self, image: torch.Tensor) -> torch.Tensor:
        """Return projected, normalised embedding (used during pre-training)."""
        return self.projector(self.encoder(image))

    def forward(
        self,
        brightfield: torch.Tensor,
        darkfield: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (z_bf, z_df) — normalised projections of both contrasts."""
        return self.project(brightfield), self.project(darkfield)


def nt_xent_loss(
    z_bf: torch.Tensor,
    z_df: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """NT-Xent (normalised temperature-scaled cross-entropy) contrastive loss.

    For a batch of N BF/DF pairs:
        - 2N embeddings total: [z_bf_0, ..., z_bf_{N-1}, z_df_0, ..., z_df_{N-1}]
        - Positive pair for z_bf_i is z_df_i and vice versa
        - All other 2N-2 embeddings in the batch are negatives

    The temperature τ=0.07 sharpens the similarity distribution. Lower temperature
    → model must more precisely distinguish positives from hard negatives within
    the batch. This value follows SimCLR recommendations for small batch sizes.

    Args:
        z_bf: (N, D) L2-normalised BF embeddings.
        z_df: (N, D) L2-normalised DF embeddings.
        temperature: Scaling factor for logits.

    Returns:
        Scalar loss averaged over both views.
    """
    n = z_bf.shape[0]
    if n < 2:
        raise ValueError("NT-Xent loss requires batch size >= 2.")

    # Concatenate both views: shape (2N, D)
    z = torch.cat([z_bf, z_df], dim=0)

    # Full cosine similarity matrix: (2N, 2N), already normalised so this is mm
    sim = torch.mm(z, z.T) / temperature

    # Mask out self-similarity on the diagonal (log(1) = 0 contribution but
    # softmax denominator should not include self)
    self_mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    # Positive pair indices:
    #   BF_i (row i)   → DF_i (column N+i)
    #   DF_i (row N+i) → BF_i (column i)
    labels = torch.cat([
        torch.arange(n, 2 * n, device=z.device),
        torch.arange(0, n, device=z.device),
    ])

    loss = F.cross_entropy(sim, labels)
    return loss


def cross_contrast_alignment_score(
    z_bf: torch.Tensor,
    z_df: torch.Tensor,
) -> torch.Tensor:
    """Per-sample cosine similarity between BF and DF embeddings.

    This is the INFERENCE-TIME gating signal for conditional dual-contrast
    processing. A high score means BF and DF agree in representation space
    (the case is easy, no need for DF). A low score means the two modalities
    disagree — the case is uncertain, and requesting DF processing is worth
    the compute cost.

    Returns:
        (N,) tensor of cosine similarities in [-1, 1].
    """
    z_bf_norm = F.normalize(z_bf, dim=1)
    z_df_norm = F.normalize(z_df, dim=1)
    return (z_bf_norm * z_df_norm).sum(dim=1)
