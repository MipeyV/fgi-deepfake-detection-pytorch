"""Checkpoint utilities for training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

__all__ = [
    "checkpoint_metric_is_better",
    "load_model_checkpoint",
    "load_training_checkpoint",
    "save_training_checkpoint",
]


def checkpoint_metric_is_better(
    current_value: float,
    best_value: float | None,
    metric_name: str,
) -> bool:
    """Return whether a metric value improves the current best value.

    Args:
        current_value: Metric value from the current epoch.
        best_value: Best metric value observed so far, or ``None``.
        metric_name: Metric key used to infer comparison direction.

    Returns:
        True if ``current_value`` should replace ``best_value``.
    """
    if best_value is None:
        return True

    if "loss" in metric_name.lower():
        return current_value < best_value

    return current_value > best_value


def save_training_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any] | None = None,
    best_metric_name: str | None = None,
    best_metric_value: float | None = None,
) -> None:
    """Save model and optimizer state for a training run.

    Args:
        checkpoint_path: Destination ``.pt`` file.
        model: Model whose weights should be saved.
        optimizer: Optimizer whose state should be saved.
        epoch: Epoch associated with this checkpoint.
        metrics: Metrics recorded for the checkpoint epoch.
        config: Optional experiment config snapshot.
        best_metric_name: Optional metric used to select the best checkpoint.
        best_metric_value: Optional best metric value at save time.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
    }
    torch.save(payload, checkpoint_path)


def load_model_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load model weights from a PyTorch checkpoint file.

    Args:
        model: Model instance to update in place.
        checkpoint_path: Path to a PyTorch checkpoint. The checkpoint may be a
            raw state dict or a dictionary containing ``model_state_dict``.

    Returns:
        Loaded checkpoint object.

    Raises:
        FileNotFoundError: If ``checkpoint_path`` does not exist.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint


def load_training_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Restore model and optimizer state for continued training."""
    checkpoint = load_model_checkpoint(model, checkpoint_path)

    if "optimizer_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain optimizer state: {checkpoint_path}")

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
