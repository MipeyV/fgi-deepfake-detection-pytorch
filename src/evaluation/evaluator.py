"""Evaluation loops and prediction export helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from src.evaluation.metrics import (
    BinaryClassificationMetrics,
    compute_binary_classification_metrics,
)


__all__ = [
    "AudioEvaluationResult",
    "PredictionRecord",
    "evaluate_audio_classifier",
    "evaluate_video_classifier",
    "write_evaluation_outputs",
    "write_metrics_json",
    "write_predictions_csv",
]


LABEL_NAMES = {0: "real", 1: "fake"}


@dataclass(frozen=True)
class PredictionRecord:
    """Single prediction row for a clip.

    Args:
        video_id: Source video identifier.
        clip_id: Clip identifier inside the source video.
        clip_path: Path to the evaluated clip directory.
        label: Ground-truth label name.
        label_idx: Ground-truth label index.
        pred_label: Predicted label name.
        pred_idx: Predicted label index.
        prob_real: Predicted probability for the real class.
        prob_fake: Predicted probability for the fake class.
        correct: Whether prediction matches ground truth.
    """

    video_id: str
    clip_id: str
    clip_path: str
    label: str
    label_idx: int
    pred_label: str
    pred_idx: int
    prob_real: float
    prob_fake: float
    correct: bool


@dataclass(frozen=True)
class AudioEvaluationResult:
    """Evaluation outputs for an audio classifier.

    Args:
        metrics: Aggregate binary classification metrics.
        predictions: Per-sample prediction records.
    """

    metrics: BinaryClassificationMetrics
    predictions: list[PredictionRecord]


def _label_name(label_idx: int) -> str:
    """Convert a label index into a label name.

    Args:
        label_idx: Integer label index.

    Returns:
        Label name when known, otherwise the integer index as a string.
    """
    return LABEL_NAMES.get(label_idx, str(label_idx))


def _metadata_value(batch: dict, key: str, index: int) -> str:
    """Read a metadata value from a collated batch.

    Args:
        batch: Batch dictionary produced by the project DataLoader.
        key: Metadata key to read.
        index: Batch item index.

    Returns:
        Metadata value converted to string.
    """
    values = batch.get(key)

    if values is None:
        return ""

    return str(values[index])


def evaluate_audio_classifier(
    model: nn.Module,
    feature_extractor: nn.Module,
    dataloader: Iterable[dict],
    device: torch.device,
    max_batches: int | None = None,
) -> AudioEvaluationResult:
    """Evaluate an audio classifier and collect per-sample predictions.

    Args:
        model: Audio classifier receiving mel-spectrogram tensors.
        feature_extractor: Module converting waveform tensors to model inputs.
        dataloader: Iterable yielding batches with ``audio`` and ``label`` keys.
        device: Device on which tensors and modules should run.
        max_batches: Optional maximum number of batches to process.

    Returns:
        Aggregate metrics and per-sample prediction records.

    Raises:
        ValueError: If no samples are evaluated.
    """
    model.to(device)
    feature_extractor.to(device)
    model.eval()
    feature_extractor.eval()

    all_labels: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    all_prob_fake: list[torch.Tensor] = []
    records: list[PredictionRecord] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            features = feature_extractor(audio)
            logits = model(features)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())
            all_prob_fake.append(probabilities[:, 1].cpu())

            for item_index in range(labels.size(0)):
                label_idx = int(labels[item_index].item())
                pred_idx = int(predictions[item_index].item())
                prob_real = float(probabilities[item_index, 0].item())
                prob_fake = float(probabilities[item_index, 1].item())

                records.append(
                    PredictionRecord(
                        video_id=_metadata_value(batch, "video_id", item_index),
                        clip_id=_metadata_value(batch, "clip_id", item_index),
                        clip_path=_metadata_value(batch, "clip_path", item_index),
                        label=_label_name(label_idx),
                        label_idx=label_idx,
                        pred_label=_label_name(pred_idx),
                        pred_idx=pred_idx,
                        prob_real=prob_real,
                        prob_fake=prob_fake,
                        correct=label_idx == pred_idx,
                    )
                )

    if not records:
        raise ValueError("No samples were evaluated")

    labels_tensor = torch.cat(all_labels)
    predictions_tensor = torch.cat(all_predictions)
    prob_fake_tensor = torch.cat(all_prob_fake)
    metrics = compute_binary_classification_metrics(
        labels=labels_tensor,
        predictions=predictions_tensor,
        scores=prob_fake_tensor,
    )

    return AudioEvaluationResult(metrics=metrics, predictions=records)


def evaluate_video_classifier(
    model: nn.Module,
    dataloader: Iterable[dict],
    device: torch.device,
    max_batches: int | None = None,
) -> AudioEvaluationResult:
    """Evaluate a video classifier and collect per-sample predictions."""
    model.to(device)
    model.eval()

    all_labels: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    all_prob_fake: list[torch.Tensor] = []
    records: list[PredictionRecord] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)

            logits = model(frames)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())
            all_prob_fake.append(probabilities[:, 1].cpu())

            for item_index in range(labels.size(0)):
                label_idx = int(labels[item_index].item())
                pred_idx = int(predictions[item_index].item())
                prob_real = float(probabilities[item_index, 0].item())
                prob_fake = float(probabilities[item_index, 1].item())

                records.append(
                    PredictionRecord(
                        video_id=_metadata_value(batch, "video_id", item_index),
                        clip_id=_metadata_value(batch, "clip_id", item_index),
                        clip_path=_metadata_value(batch, "clip_path", item_index),
                        label=_label_name(label_idx),
                        label_idx=label_idx,
                        pred_label=_label_name(pred_idx),
                        pred_idx=pred_idx,
                        prob_real=prob_real,
                        prob_fake=prob_fake,
                        correct=label_idx == pred_idx,
                    )
                )

    if not records:
        raise ValueError("No samples were evaluated")

    labels_tensor = torch.cat(all_labels)
    predictions_tensor = torch.cat(all_predictions)
    prob_fake_tensor = torch.cat(all_prob_fake)
    metrics = compute_binary_classification_metrics(
        labels=labels_tensor,
        predictions=predictions_tensor,
        scores=prob_fake_tensor,
    )

    return AudioEvaluationResult(metrics=metrics, predictions=records)


def write_predictions_csv(predictions: list[PredictionRecord], path: str | Path) -> None:
    """Write prediction records to a CSV file.

    Args:
        predictions: Prediction rows to write.
        path: Destination CSV path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_id",
        "clip_id",
        "clip_path",
        "label",
        "label_idx",
        "pred_label",
        "pred_idx",
        "prob_real",
        "prob_fake",
        "correct",
    ]

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for prediction in predictions:
            writer.writerow(asdict(prediction))


def write_metrics_json(
    metrics: BinaryClassificationMetrics,
    path: str | Path,
) -> None:
    """Write metrics to a JSON file.

    Args:
        metrics: Metrics object to serialize.
        path: Destination JSON path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_evaluation_outputs(
    result: AudioEvaluationResult,
    predictions_path: str | Path,
    metrics_path: str | Path,
) -> None:
    """Write predictions and metrics for an evaluation result.

    Args:
        result: Evaluation result to write.
        predictions_path: Destination predictions CSV path.
        metrics_path: Destination metrics JSON path.
    """
    write_predictions_csv(result.predictions, predictions_path)
    write_metrics_json(result.metrics, metrics_path)
