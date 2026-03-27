"""Small torchvision backbones suitable for CPU or MPS prototyping."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import get_model, get_model_weights


SUPPORTED_BACKBONES = {
    "mobilenet_v3_small",
    "resnet18",
}


def build_backbone(name: str = "resnet18", pretrained: bool = False) -> tuple[nn.Module, int]:
    """Build a feature extractor and return the extractor plus feature dimension."""
    if name not in SUPPORTED_BACKBONES:
        available = ", ".join(sorted(SUPPORTED_BACKBONES))
        raise ValueError(f"Unsupported backbone '{name}'. Choose from: {available}")

    weights = None
    if pretrained:
        try:
            weights = get_model_weights(name).DEFAULT
        except Exception:
            weights = None

    model = get_model(name, weights=weights)

    if hasattr(model, "fc"):
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim

    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Sequential) and len(classifier) > 0:
            last_layer = classifier[-1]
            feature_dim = getattr(last_layer, "in_features", None)
        else:
            feature_dim = getattr(classifier, "in_features", None)

        if feature_dim is None:
            raise ValueError(f"Could not infer feature dimension for backbone '{name}'.")

        model.classifier = nn.Identity()
        return model, feature_dim

    raise ValueError(f"Backbone '{name}' does not expose a supported classifier head.")

