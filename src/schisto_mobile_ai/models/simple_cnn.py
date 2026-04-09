"""Very small convolutional classifier for smoke tests and lightweight baselines."""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyConvEncoder(nn.Module):
    """Compact convolutional encoder that outputs one pooled feature vector per image.

    Architecture: 4 conv blocks (3→6→12→24→48 channels for base_channels=12) with
    batch norm, then global average pool. At 224px input this gives a 48-d feature
    vector after 4 max-pool halvings (224→112→56→28→14→global). Enough capacity to
    detect schistosome eggs without torchvision dependencies.
    """

    def __init__(self, *, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.feature_dim = c * 4
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 2
            nn.Conv2d(c, c * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 3
            nn.Conv2d(c * 2, c * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 4 — extra depth for 224px inputs
            nn.Conv2d(c * 4, c * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.features(image)
        return torch.flatten(features, start_dim=1)


class TinyConvClassifier(nn.Module):
    """A compact CNN that avoids torchvision dependencies."""

    def __init__(self, *, in_channels: int = 3, num_classes: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        self.encoder = TinyConvEncoder(in_channels=in_channels, base_channels=base_channels)
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.encoder.feature_dim, num_classes),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(image))
