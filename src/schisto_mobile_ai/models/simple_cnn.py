"""Very small convolutional classifier for smoke tests and lightweight baselines."""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyConvClassifier(nn.Module):
    """A compact CNN that avoids torchvision dependencies."""

    def __init__(self, *, in_channels: int = 3, num_classes: int = 1, base_channels: int = 16) -> None:
        super().__init__()
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
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(base_channels * 4, num_classes),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(image))
