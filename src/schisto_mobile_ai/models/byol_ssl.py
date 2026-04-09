"""BYOL (Bootstrap Your Own Latent) SSL for cross-contrast schistosomiasis pre-training.

BYOL (Grill et al., 2020) is the fallback if SimCLR shows poor transfer. Unlike
SimCLR, BYOL does not require negative pairs, so it works well with small batches
and limited data — both constraints of our dataset.

Architecture:
    Online network  : encoder → projector → predictor
    Target network  : encoder_ema → projector_ema  (exponential moving average, no grad)

Training signal:
    Minimise MSE between predictor(online(view_a)) and stop_grad(target(view_b))
    and symmetrically predictor(online(view_b)) and stop_grad(target(view_a))

For cross-contrast pre-training:
    view_a = augmented BF image
    view_b = augmented DF image of the SAME slide

The EMA target network provides stable targets without needing negatives.
The predictor prevents representation collapse (without negatives, the network
could trivially minimise the loss by mapping everything to the same point —
the predictor + EMA asymmetry prevents this).

Key advantage over SimCLR for this dataset:
    SimCLR needs large batches for sufficient negatives.
    BYOL needs none. Works with batch_size=16 or even 8.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from schisto_mobile_ai.models.simple_cnn import TinyConvEncoder


class BYOLProjector(nn.Module):
    """Two-layer MLP projector with BN, used by both online and target networks."""

    def __init__(self, *, feature_dim: int, hidden_dim: int = 256, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BYOLPredictor(nn.Module):
    """Single-layer MLP predictor — only on the online network, key to preventing collapse."""

    def __init__(self, *, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossContrastBYOL(nn.Module):
    """BYOL for cross-contrast self-supervised pre-training.

    The ONLINE network processes one view; the TARGET (EMA) network processes
    the other. Gradients flow only through the online network.

    After pre-training, use online_encoder.state_dict() for initialising
    downstream fine-tuning — identical interface to SimCLR encoder_weights.pt.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        base_channels: int = 32,
        projection_dim: int = 128,
        projection_hidden_dim: int = 256,
        predictor_hidden_dim: int = 256,
        ema_decay: float = 0.996,
    ) -> None:
        super().__init__()
        self.ema_decay = ema_decay

        # Online network
        self.online_encoder = TinyConvEncoder(in_channels=in_channels, base_channels=base_channels)
        feature_dim = self.online_encoder.feature_dim
        self.online_projector = BYOLProjector(
            feature_dim=feature_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim,
        )
        self.predictor = BYOLPredictor(
            input_dim=projection_dim,
            hidden_dim=predictor_hidden_dim,
            output_dim=projection_dim,
        )

        # Target network — EMA copy of online, no gradient
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @property
    def feature_dim(self) -> int:
        return self.online_encoder.feature_dim

    @torch.no_grad()
    def update_target(self) -> None:
        """EMA update of target network parameters. Call after each optimiser step."""
        tau = self.ema_decay
        for online_p, target_p in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_p.data = tau * target_p.data + (1.0 - tau) * online_p.data
        for online_p, target_p in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            target_p.data = tau * target_p.data + (1.0 - tau) * online_p.data

    def _online_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.online_projector(self.online_encoder(x)))

    @torch.no_grad()
    def _target_forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.target_projector(self.target_encoder(x)), dim=1)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Raw online encoder features — used for fine-tuning."""
        return self.online_encoder(image)

    def forward(
        self,
        brightfield: torch.Tensor,
        darkfield: torch.Tensor,
    ) -> torch.Tensor:
        """Return BYOL loss for a BF/DF pair batch.

        Computes symmetric loss:
            L = regression(predict(online(BF)), stop_grad(target(DF)))
              + regression(predict(online(DF)), stop_grad(target(BF)))
        """
        # Online predictions
        pred_bf = self._online_forward(brightfield)
        pred_df = self._online_forward(darkfield)

        # Target representations (no gradient)
        tgt_bf = self._target_forward(brightfield)
        tgt_df = self._target_forward(darkfield)

        # Symmetric MSE in normalised space
        loss = byol_loss(pred_bf, tgt_df) + byol_loss(pred_df, tgt_bf)
        return loss


def byol_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Normalised MSE — equivalent to 2 - 2*cosine_similarity when both are unit vectors."""
    pred_norm = F.normalize(prediction, dim=1)
    # target is already normalised from _target_forward
    return 2.0 - 2.0 * (pred_norm * target).sum(dim=1).mean()


def cross_contrast_alignment_score_byol(
    model: CrossContrastBYOL,
    brightfield: torch.Tensor,
    darkfield: torch.Tensor,
) -> torch.Tensor:
    """Per-sample cosine similarity in ONLINE encoder space (for gating diagnostics)."""
    with torch.no_grad():
        z_bf = F.normalize(model.online_encoder(brightfield), dim=1)
        z_df = F.normalize(model.online_encoder(darkfield), dim=1)
    return (z_bf * z_df).sum(dim=1)
