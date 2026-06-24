"""Evaluation metrics for binary deepfake detection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "BinaryClassificationMetrics",
    "ThresholdCalibrationResult",
    "calibrate_binary_threshold",
    "compute_binary_average_precision",
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
        specificity: Fraction of real samples detected as real.
        balanced_accuracy: Mean of recall and specificity.
        f1: Harmonic mean of precision and recall.
        f1_macro: Mean F1 score across the real and fake classes.
        auc: ROC AUC computed from fake-class probabilities, or ``None`` when
            AUC is undefined because only one class is present.
        average_precision: Area under the precision-recall step curve, or
            ``None`` when no fake sample is present.
        true_negatives: Number of real samples predicted as real.
        false_positives: Number of real samples predicted as fake.
        false_negatives: Number of fake samples predicted as real.
        true_positives: Number of fake samples predicted as fake.
        num_samples: Number of evaluated samples.
    """

    accuracy: float
    precision: float
    recall: float
    specificity: float
    balanced_accuracy: float
    f1: float
    f1_macro: float
    auc: float | None
    average_precision: float | None
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    num_samples: int
    decision_threshold: float | None = None


@dataclass(frozen=True)
class ThresholdCalibrationResult:
    """Decision threshold selected on a labelled calibration split."""

    threshold: float
    metric_name: str
    metric_value: float
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
    auc = (positive_rank_sum - num_positive * (num_positive + 1) / 2.0) / (
        num_positive * num_negative
    )

    return float(auc)


def compute_binary_average_precision(
    labels: torch.Tensor,
    scores: torch.Tensor,
) -> float | None:
    """Compute average precision for fake as the positive class."""
    _validate_1d_tensor(labels, "labels")
    _validate_1d_tensor(scores, "scores")

    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")

    labels = labels.long()
    scores = scores.float()
    num_positive = int((labels == 1).sum().item())
    if num_positive == 0:
        return None

    sorted_indices = torch.argsort(scores, descending=True, stable=True)
    sorted_labels = labels[sorted_indices]
    sorted_scores = scores[sorted_indices]
    true_positives = torch.cumsum(sorted_labels == 1, dim=0).float()
    false_positives = torch.cumsum(sorted_labels == 0, dim=0).float()
    group_ends = torch.ones_like(sorted_scores, dtype=torch.bool)
    group_ends[:-1] = sorted_scores[:-1] != sorted_scores[1:]

    recalls = true_positives[group_ends] / num_positive
    precisions = true_positives[group_ends] / (
        true_positives[group_ends] + false_positives[group_ends]
    )
    previous_recalls = torch.cat([torch.zeros(1), recalls[:-1]])
    average_precision = ((recalls - previous_recalls) * precisions).sum()
    return float(average_precision.item())


def compute_binary_classification_metrics(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    scores: torch.Tensor | None = None,
    decision_threshold: float | None = None,
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
    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    real_precision = tn / (tn + fn) if tn + fn > 0 else 0.0
    real_recall = specificity
    real_f1 = (
        2 * real_precision * real_recall / (real_precision + real_recall)
        if real_precision + real_recall > 0
        else 0.0
    )
    f1_macro = (real_f1 + f1) / 2
    auc = compute_binary_roc_auc(labels, scores) if scores is not None else None
    average_precision = (
        compute_binary_average_precision(labels, scores) if scores is not None else None
    )

    return BinaryClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        balanced_accuracy=balanced_accuracy,
        f1=f1,
        f1_macro=f1_macro,
        auc=auc,
        average_precision=average_precision,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
        num_samples=num_samples,
        decision_threshold=decision_threshold,
    )


def calibrate_binary_threshold(
    labels: torch.Tensor,
    scores: torch.Tensor,
    metric_name: str = "balanced_accuracy",
) -> ThresholdCalibrationResult:
    """Select a decision threshold using labelled validation scores.

    Ties are resolved by choosing the threshold closest to ``0.5`` and then
    the lower threshold.
    """
    _validate_1d_tensor(labels, "labels")
    _validate_1d_tensor(scores, "scores")
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
    if labels.numel() == 0:
        raise ValueError("At least one sample is required for calibration")
    if metric_name not in {"balanced_accuracy", "f1_macro"}:
        raise ValueError(f"Unsupported calibration metric: {metric_name}")

    labels = labels.long()
    scores = scores.float()
    if int((labels == 0).sum()) == 0 or int((labels == 1).sum()) == 0:
        raise ValueError("Threshold calibration requires both classes")

    unique_scores = torch.unique(scores).sort().values.tolist()
    candidates = [0.0, *[float(score) for score in unique_scores], 1.0]
    best_threshold = 0.5
    best_value = -1.0

    for threshold in candidates:
        predictions = (scores > threshold).long()
        metrics = compute_binary_classification_metrics(labels, predictions)
        metric_value = float(getattr(metrics, metric_name))
        candidate_key = (
            metric_value,
            -abs(threshold - 0.5),
            -threshold,
        )
        best_key = (
            best_value,
            -abs(best_threshold - 0.5),
            -best_threshold,
        )
        if candidate_key > best_key:
            best_threshold = threshold
            best_value = metric_value

    return ThresholdCalibrationResult(
        threshold=best_threshold,
        metric_name=metric_name,
        metric_value=best_value,
        num_samples=int(labels.numel()),
    )
