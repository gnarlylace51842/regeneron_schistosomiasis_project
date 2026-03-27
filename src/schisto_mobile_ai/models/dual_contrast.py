"""Simple placeholder architectures for single- and dual-contrast experiments."""

from __future__ import annotations

import torch
import torch.nn as nn

from schisto_mobile_ai.models.backbones import build_backbone


def _flatten_features(features: torch.Tensor) -> torch.Tensor:
    """Flatten feature maps if the encoder returns spatial tensors."""
    if features.ndim > 2:
        return torch.flatten(features, start_dim=1)
    return features


class SingleContrastClassifier(nn.Module):
    """Minimal single-image classifier for early prototyping."""

    def __init__(
        self,
        *,
        backbone_name: str = "resnet18",
        pretrained: bool = False,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.encoder, feature_dim = build_backbone(backbone_name, pretrained=pretrained)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = _flatten_features(self.encoder(image))
        return self.head(features)


class DualContrastClassifier(nn.Module):
    """Placeholder dual-branch classifier with simple fusion."""

    def __init__(
        self,
        *,
        backbone_name: str = "resnet18",
        pretrained: bool = False,
        num_classes: int = 1,
        share_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.share_encoder = share_encoder

        self.encoder_a, feature_dim = build_backbone(backbone_name, pretrained=pretrained)
        if share_encoder:
            self.encoder_b = self.encoder_a
        else:
            self.encoder_b, _ = build_backbone(backbone_name, pretrained=pretrained)

        fused_dim = feature_dim * 3
        self.head = nn.Sequential(
            nn.Linear(fused_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, contrast_a: torch.Tensor, contrast_b: torch.Tensor) -> torch.Tensor:
        embedding_a = _flatten_features(self.encoder_a(contrast_a))
        embedding_b = _flatten_features(self.encoder_b(contrast_b))
        fused = torch.cat(
            [embedding_a, embedding_b, torch.abs(embedding_a - embedding_b)],
            dim=1,
        )
        return self.head(fused)

