import pytest
import torch

from src.evaluation.metrics import (
    compute_binary_classification_metrics,
    compute_binary_confusion_matrix,
    compute_binary_roc_auc,
)


def test_compute_binary_confusion_matrix_returns_tn_fp_fn_tp() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    predictions = torch.tensor([0, 1, 0, 1])

    assert compute_binary_confusion_matrix(labels, predictions) == (1, 1, 1, 1)


def test_compute_binary_roc_auc_returns_expected_value() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    scores = torch.tensor([0.1, 0.4, 0.35, 0.8])

    auc = compute_binary_roc_auc(labels, scores)

    assert auc == pytest.approx(0.75)


def test_compute_binary_roc_auc_returns_none_for_single_class() -> None:
    labels = torch.tensor([0, 0, 0])
    scores = torch.tensor([0.1, 0.2, 0.3])

    assert compute_binary_roc_auc(labels, scores) is None


def test_compute_binary_classification_metrics() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    predictions = torch.tensor([0, 1, 0, 1])
    scores = torch.tensor([0.1, 0.8, 0.4, 0.9])

    metrics = compute_binary_classification_metrics(labels, predictions, scores)

    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.auc == pytest.approx(0.75)
    assert metrics.true_negatives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.true_positives == 1
    assert metrics.num_samples == 4


def test_compute_binary_classification_metrics_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match="At least one sample"):
        compute_binary_classification_metrics(
            torch.tensor([]),
            torch.tensor([]),
        )


def test_compute_binary_classification_metrics_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_binary_classification_metrics(
            torch.tensor([0, 1]),
            torch.tensor([0]),
        )
