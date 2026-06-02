"""Evaluation metrics for binary deepfake detection."""

from __future__ import annotations

from dataclasses import dataclass

import torch


__all__ = [
    "BinaryClassificationMetrics",
    "compute_binary_classification_metrics",
    "compute_binary_confusion_matrix",
    "compute_binary_roc_auc",
]


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    """Aggregate binary classification metrics.

    Args:
        accuracy: Fraction of correct predictions.
        precision: Fraction of predicted fake samples that are truly fake.
        recall: Fraction of fake samples detected as fake.
        f1: Harmonic mean of precision and recall.
        auc: ROC AUC computed from fake-class probabilities, or ``None`` when
            AUC is undefined because only one class is present.
        true_negatives: Number of real samples predicted as real.
        false_positives: Number of real samples predicted as fake.
        false_negatives: Number of fake samples predicted as real.
        true_positives: Number of fake samples predicted as fake.
        num_samples: Number of evaluated samples.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float | None
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    num_samples: int


def _validate_1d_tensor(values: torch.Tensor, name: str) -> None:
    """Validate that a tensor is one-dimensional.

    Args:
        values: Tensor to validate.
        name: Human-readable tensor name used in error messages.

    Raises:
        ValueError: If ``values`` is not one-dimensional.
    """
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D tensor")


def compute_binary_confusion_matrix(
    labels: torch.Tensor,
    predictions: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Compute a binary confusion matrix with fake as positive class.

    Args:
        labels: Ground-truth labels where ``0`` is real and ``1`` is fake.
        predictions: Predicted labels where ``0`` is real and ``1`` is fake.

    Returns:
        Tuple ``(tn, fp, fn, tp)``.

    Raises:
        ValueError: If tensors are not one-dimensional or do not share shape.
    """
    _validate_1d_tensor(labels, "labels")
    _validate_1d_tensor(predictions, "predictions")

    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")

    labels = labels.long()
    predictions = predictions.long()

    true_negatives = ((labels == 0) & (predictions == 0)).sum().item()
    false_positives = ((labels == 0) & (predictions == 1)).sum().item()
    false_negatives = ((labels == 1) & (predictions == 0)).sum().item()
    true_positives = ((labels == 1) & (predictions == 1)).sum().item()

    return (
        int(true_negatives),
        int(false_positives),
        int(false_negatives),
        int(true_positives),
    )


def compute_binary_roc_auc(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    """Compute ROC AUC for binary labels using rank statistics.

    Args:
        labels: Ground-truth labels where ``0`` is real and ``1`` is fake.
        scores: Fake-class scores or probabilities.

    Returns:
        ROC AUC value, or ``None`` if either class is absent.

    Raises:
        ValueError: If tensors are not one-dimensional or do not share shape.
    """
    _validate_1d_tensor(labels, "labels")
    _validate_1d_tensor(scores, "scores")

    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")

    labels = labels.long()
    scores = scores.float()

    num_positive = int((labels == 1).sum().item())
    num_negative = int((labels == 0).sum().item())

    if num_positive == 0 or num_negative == 0:
        return None

    sorted_indices = torch.argsort(scores)
    sorted_scores = scores[sorted_indices]
    ranks = torch.empty_like(scores, dtype=torch.float)

    start = 0
    rank = 1.0

    while start < scores.numel():
        end = start + 1

        while end < scores.numel() and sorted_scores[end] == sorted_scores[start]:
            end += 1

        average_rank = (rank + rank + (end - start) - 1) / 2.0
        ranks[sorted_indices[start:end]] = average_rank
        rank += end - start
        start = end

    positive_rank_sum = ranks[labels == 1].sum().item()
    auc = (
        positive_rank_sum - num_positive * (num_positive + 1) / 2.0
    ) / (num_positive * num_negative)

    return float(auc)


def compute_binary_classification_metrics(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    scores: torch.Tensor | None = None,
) -> BinaryClassificationMetrics:
    """Compute binary classification metrics for real/fake prediction.

    Args:
        labels: Ground-truth labels where ``0`` is real and ``1`` is fake.
        predictions: Predicted labels where ``0`` is real and ``1`` is fake.
        scores: Optional fake-class scores or probabilities used for ROC AUC.

    Returns:
        Aggregate binary classification metrics.

    Raises:
        ValueError: If labels and predictions are empty or shape-incompatible.
    """
    _validate_1d_tensor(labels, "labels")
    _validate_1d_tensor(predictions, "predictions")

    if labels.numel() == 0:
        raise ValueError("At least one sample is required to compute metrics")

    tn, fp, fn, tp = compute_binary_confusion_matrix(labels, predictions)
    num_samples = int(labels.numel())
    correct = tn + tp
    accuracy = correct / num_samples
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    auc = compute_binary_roc_auc(labels, scores) if scores is not None else None

    return BinaryClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
        num_samples=num_samples,
    )
