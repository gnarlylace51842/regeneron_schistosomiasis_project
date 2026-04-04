"""Model helpers for lightweight experiments."""

from schisto_mobile_ai.models.dual_baseline import AlwaysOnDualContrastClassifier
from schisto_mobile_ai.models.patient_aggregation import aggregate_patient_predictions
from schisto_mobile_ai.models.simple_cnn import TinyConvClassifier, TinyConvEncoder

try:  # Optional torchvision-backed models remain available when torchvision is installed.
    from schisto_mobile_ai.models.backbones import build_backbone
    from schisto_mobile_ai.models.dual_contrast import DualContrastClassifier, SingleContrastClassifier
except ModuleNotFoundError:  # pragma: no cover - depends on local optional installs
    build_backbone = None
    DualContrastClassifier = None
    SingleContrastClassifier = None

__all__ = [
    "AlwaysOnDualContrastClassifier",
    "DualContrastClassifier",
    "SingleContrastClassifier",
    "TinyConvClassifier",
    "TinyConvEncoder",
    "aggregate_patient_predictions",
    "build_backbone",
]
