"""Lightweight always-on BF+DF baseline without torchvision dependencies."""

from __future__ import annotations

import torch
import torch.nn as nn

from schisto_mobile_ai.models.simple_cnn import TinyConvEncoder


class AlwaysOnDualContrastClassifier(nn.Module):
    """Shared-encoder dual-contrast classifier with simple concatenation fusion."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 16,
        share_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.share_encoder = share_encoder

        self.encoder_a = TinyConvEncoder(in_channels=in_channels, base_channels=base_channels)
        if share_encoder:
            self.encoder_b = self.encoder_a
        else:
            self.encoder_b = TinyConvEncoder(in_channels=in_channels, base_channels=base_channels)

        feature_dim = self.encoder_a.feature_dim
        self.head = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, brightfield_image: torch.Tensor, darkfield_image: torch.Tensor) -> torch.Tensor:
        features_bf = self.encoder_a(brightfield_image)
        features_df = self.encoder_b(darkfield_image)
        fused = torch.cat([features_bf, features_df], dim=1)
        return self.head(fused)
