"""Backward-compatible imports for video models.

New code may import from ``src.models.video`` directly.
"""

from src.models.baselines.video import (
    R3D18VideoClassifier,
    VideoCNNBaseline,
    VideoConvBlock,
    build_video_model,
)

__all__ = [
    "R3D18VideoClassifier",
    "VideoCNNBaseline",
    "VideoConvBlock",
    "build_video_model",
]
