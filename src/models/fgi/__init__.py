"""FGI-inspired model components."""

from src.models.fgi.encoders import (
    FGIAudioEncoder,
    FGIEncoderPair,
    FGIVideoEncoder,
    build_fgi_encoders,
)

__all__ = [
    "FGIAudioEncoder",
    "FGIEncoderPair",
    "FGIVideoEncoder",
    "build_fgi_encoders",
]
