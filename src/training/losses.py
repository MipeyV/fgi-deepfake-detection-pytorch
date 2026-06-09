"""Classification loss construction and class-weight utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn


def compute_balanced_class_weights(
    manifest_path: str | Path,
    label_mapping: dict[str, int],
) -> torch.Tensor:
    """Compute inverse-frequency class weights from a training manifest."""
    samples = pd.read_csv(manifest_path, usecols=["label"], dtype={"label": str})
    num_classes = len(label_mapping)
    num_samples = len(samples)

    if num_samples == 0:
        raise ValueError("Cannot compute class weights from an empty manifest")

    weights = torch.empty(num_classes, dtype=torch.float32)
    for label, class_index in label_mapping.items():
        class_count = int((samples["label"] == label).sum())
        if class_count == 0:
            raise ValueError(
                f"Cannot compute balanced class weights: class '{label}' is absent"
            )
        weights[class_index] = num_samples / (num_classes * class_count)

    return weights


def build_classification_criterion(
    loss_config: dict,
    train_manifest_path: str | Path,
    label_mapping: dict[str, int],
    device: torch.device,
) -> nn.Module:
    """Build the configured cross-entropy loss, optionally with class weights."""
    loss_name = str(loss_config.get("name", "cross_entropy")).lower()
    if loss_name != "cross_entropy":
        raise ValueError(f"Unsupported loss: {loss_config.get('name')}")

    class_weights_config = loss_config.get("class_weights")
    if class_weights_config in (None, "none"):
        class_weights = None
    elif class_weights_config == "balanced":
        class_weights = compute_balanced_class_weights(
            train_manifest_path,
            label_mapping,
        )
    elif isinstance(class_weights_config, list):
        class_weights = torch.tensor(class_weights_config, dtype=torch.float32)
        if class_weights.numel() != len(label_mapping):
            raise ValueError(
                "training.loss.class_weights must contain one weight per class"
            )
        if not torch.all(class_weights > 0):
            raise ValueError("training.loss.class_weights must be strictly positive")
    else:
        raise ValueError(
            "training.loss.class_weights must be 'balanced', 'none', or a list"
        )

    if class_weights is not None:
        class_weights = class_weights.to(device)

    return nn.CrossEntropyLoss(weight=class_weights)
