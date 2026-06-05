"""Training utilities for audio-only baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.optim import Optimizer


__all__ = [
    "EpochMetrics",
    "build_optimizer",
    "evaluate_audio_model",
    "evaluate_video_model",
    "resolve_device",
    "train_one_epoch",
    "train_video_one_epoch",
]


@dataclass(frozen=True)
class EpochMetrics:
    """Aggregate metrics produced by one training or evaluation pass.

    Args:
        loss: Mean loss over all processed samples.
        accuracy: Classification accuracy over all processed samples.
        num_samples: Number of samples processed.
        num_batches: Number of batches processed.
    """

    loss: float
    accuracy: float
    num_samples: int
    num_batches: int


def resolve_device(device_name: str = "auto") -> torch.device:
    """Resolve a configured device name into a PyTorch device.

    Args:
        device_name: Device requested by config. Supported values are
            ``"auto"``, ``"cpu"``, and ``"cuda"``.

    Returns:
        PyTorch device selected for training or evaluation.

    Raises:
        ValueError: If ``device_name`` is unsupported or CUDA is requested but
            unavailable.
    """
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")

        return torch.device("cuda")

    raise ValueError(f"Unsupported device: {device_name}")


def build_optimizer(model: nn.Module, optimizer_config: dict) -> Optimizer:
    """Build an optimizer from a training config section.

    Args:
        model: Model whose parameters should be optimized.
        optimizer_config: Optimizer configuration dictionary. Supported keys are
            ``name``, ``learning_rate``, and ``weight_decay``.

    Returns:
        Configured PyTorch optimizer.

    Raises:
        ValueError: If the optimizer name is unsupported.
    """
    optimizer_name = optimizer_config.get("name", "adam").lower()
    learning_rate = optimizer_config.get("learning_rate", 0.001)
    weight_decay = optimizer_config.get("weight_decay", 0.0)

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=optimizer_config.get("momentum", 0.0),
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_config.get('name')}")


def _validate_batch(batch: dict) -> None:
    """Validate that a batch contains the tensors needed for audio training.

    Args:
        batch: Batch dictionary produced by the project DataLoader.

    Raises:
        ValueError: If the batch does not contain ``audio`` and ``label``.
    """
    missing_keys = {"audio", "label"} - set(batch)

    if missing_keys:
        raise ValueError(f"Batch is missing required keys: {sorted(missing_keys)}")


def _update_running_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss: torch.Tensor,
    totals: dict[str, float],
) -> None:
    """Update accumulated loss and accuracy counters.

    Args:
        logits: Raw class logits produced by the model.
        labels: Ground-truth class indices.
        loss: Scalar loss for the current batch.
        totals: Mutable metric counters to update in place.
    """
    batch_size = labels.size(0)
    predictions = logits.argmax(dim=1)

    totals["loss"] += loss.item() * batch_size
    totals["correct"] += (predictions == labels).sum().item()
    totals["samples"] += batch_size
    totals["batches"] += 1


def _finalize_metrics(totals: dict[str, float]) -> EpochMetrics:
    """Convert running metric counters into an ``EpochMetrics`` object.

    Args:
        totals: Running metric counters collected during a pass.

    Returns:
        Aggregate epoch metrics.

    Raises:
        ValueError: If no samples were processed.
    """
    num_samples = int(totals["samples"])

    if num_samples == 0:
        raise ValueError("No samples were processed")

    return EpochMetrics(
        loss=totals["loss"] / num_samples,
        accuracy=totals["correct"] / num_samples,
        num_samples=num_samples,
        num_batches=int(totals["batches"]),
    )


def train_one_epoch(
    model: nn.Module,
    feature_extractor: nn.Module,
    dataloader: Iterable[dict],
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> EpochMetrics:
    """Train an audio model for one epoch.

    Args:
        model: Audio classifier receiving mel-spectrogram tensors.
        feature_extractor: Module converting waveform tensors to model inputs.
        dataloader: Iterable yielding batches with ``audio`` and ``label`` keys.
        optimizer: Optimizer used to update model parameters.
        criterion: Loss function, usually ``nn.CrossEntropyLoss``.
        device: Device on which tensors and modules should run.
        max_batches: Optional maximum number of batches to process. Useful for
            smoke tests and quick overfit checks.

    Returns:
        Mean loss, accuracy, and processed sample counts.

    Raises:
        ValueError: If a batch is missing required keys or no samples are
            processed.
    """
    model.to(device)
    feature_extractor.to(device)
    model.train()
    feature_extractor.eval()

    totals = {"loss": 0.0, "correct": 0.0, "samples": 0.0, "batches": 0.0}

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        _validate_batch(batch)

        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            features = feature_extractor(audio)

        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        _update_running_metrics(logits, labels, loss, totals)

    return _finalize_metrics(totals)


def evaluate_audio_model(
    model: nn.Module,
    feature_extractor: nn.Module,
    dataloader: Iterable[dict],
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> EpochMetrics:
    """Evaluate an audio model without updating parameters.

    Args:
        model: Audio classifier receiving mel-spectrogram tensors.
        feature_extractor: Module converting waveform tensors to model inputs.
        dataloader: Iterable yielding batches with ``audio`` and ``label`` keys.
        criterion: Loss function, usually ``nn.CrossEntropyLoss``.
        device: Device on which tensors and modules should run.
        max_batches: Optional maximum number of batches to process.

    Returns:
        Mean loss, accuracy, and processed sample counts.

    Raises:
        ValueError: If a batch is missing required keys or no samples are
            processed.
    """
    model.to(device)
    feature_extractor.to(device)
    model.eval()
    feature_extractor.eval()

    totals = {"loss": 0.0, "correct": 0.0, "samples": 0.0, "batches": 0.0}

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            _validate_batch(batch)

            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            features = feature_extractor(audio)
            logits = model(features)
            loss = criterion(logits, labels)

            _update_running_metrics(logits, labels, loss, totals)

    return _finalize_metrics(totals)


def train_video_one_epoch(
    model: nn.Module,
    dataloader: Iterable[dict],
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> EpochMetrics:
    """Train a video model for one epoch."""
    model.to(device)
    model.train()

    totals = {"loss": 0.0, "correct": 0.0, "samples": 0.0, "batches": 0.0}

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        missing_keys = {"frames", "label"} - set(batch)
        if missing_keys:
            raise ValueError(f"Batch is missing required keys: {sorted(missing_keys)}")

        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        _update_running_metrics(logits, labels, loss, totals)

    return _finalize_metrics(totals)


def evaluate_video_model(
    model: nn.Module,
    dataloader: Iterable[dict],
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> EpochMetrics:
    """Evaluate a video model without updating parameters."""
    model.to(device)
    model.eval()

    totals = {"loss": 0.0, "correct": 0.0, "samples": 0.0, "batches": 0.0}

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            missing_keys = {"frames", "label"} - set(batch)
            if missing_keys:
                raise ValueError(
                    f"Batch is missing required keys: {sorted(missing_keys)}"
                )

            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)

            logits = model(frames)
            loss = criterion(logits, labels)

            _update_running_metrics(logits, labels, loss, totals)

    return _finalize_metrics(totals)
