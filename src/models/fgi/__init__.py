"""FGI-inspired model components."""

from src.models.fgi.encoders import (
    FGIAudioEncoder,
    FGIEncoderPair,
    FGIVideoEncoder,
    build_fgi_encoders,
)
from src.models.fgi.model import (
    FGIInspiredClassifier,
    FGILocalInconsistency,
    FGIModelOutput,
    FGISpatialAttention,
    build_fgi_model,
)

__all__ = [
    "FGIAudioEncoder",
    "FGIEncoderPair",
    "FGIInspiredClassifier",
    "FGILocalInconsistency",
    "FGIModelOutput",
    "FGISpatialAttention",
    "FGIVideoEncoder",
    "build_fgi_encoders",
    "build_fgi_model",
]
