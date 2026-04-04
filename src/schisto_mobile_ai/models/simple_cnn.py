"""Very small convolutional classifier for smoke tests and lightweight baselines."""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyConvEncoder(nn.Module):
    """Compact convolutional encoder that outputs one pooled feature vector per image."""

    def __init__(self, *, in_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        self.feature_dim = base_channels * 4
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
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
