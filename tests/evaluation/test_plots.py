import json
from pathlib import Path

import pytest

from src.evaluation.plots import (
    load_training_history,
    plot_confusion_matrix_svg,
    plot_metric_history_svg,
    plot_roc_comparison_svg,
    plot_roc_curve_svg,
    plot_training_history_svg,
)


def test_load_training_history_reads_non_empty_list(tmp_path: Path) -> None:
    metrics_path = tmp_path / "train_metrics.json"
    metrics_path.write_text(
        json.dumps([{"epoch": 1, "loss": 0.5, "accuracy": 0.8}]),
        encoding="utf-8",
    )

    history = load_training_history(metrics_path)

    assert history == [{"epoch": 1, "loss": 0.5, "accuracy": 0.8}]


def test_load_training_history_rejects_empty_history(tmp_path: Path) -> None:
    metrics_path = tmp_path / "train_metrics.json"
    metrics_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty list"):
        load_training_history(metrics_path)


def test_plot_training_history_svg_writes_svg(tmp_path: Path) -> None:
    metrics_path = tmp_path / "train_metrics.json"
    output_path = tmp_path / "plots" / "training_history.svg"
    metrics_path.write_text(
        json.dumps(
            [
                {"epoch": 1, "loss": 0.7, "accuracy": 0.5},
                {"epoch": 2, "loss": 0.4, "accuracy": 0.8},
            ]
        ),
        encoding="utf-8",
    )

    plot_training_history_svg(metrics_path, output_path)

    svg = output_path.read_text(encoding="utf-8")

    assert svg.startswith("<svg")
    assert "Training History" in svg
    assert "loss 0.4000" in svg
    assert "accuracy 0.8000" in svg
    assert 'stroke="#e5e7eb"' in svg
    assert 'font-size="11"' in svg


def test_plot_training_history_svg_rejects_missing_metric_keys(tmp_path: Path) -> None:
    metrics_path = tmp_path / "train_metrics.json"
    output_path = tmp_path / "plot.svg"
    metrics_path.write_text(json.dumps([{"epoch": 1, "loss": 0.7}]), encoding="utf-8")

    with pytest.raises(ValueError, match="missing keys"):
        plot_training_history_svg(metrics_path, output_path)


def test_plot_metric_history_svg_writes_single_metric_svg(tmp_path: Path) -> None:
    metrics_path = tmp_path / "train_metrics.json"
    output_path = tmp_path / "plots" / "train_loss.svg"
    metrics_path.write_text(
        json.dumps(
            [
                {"epoch": 1, "loss": 0.7, "accuracy": 0.5},
                {"epoch": 2, "loss": 0.4, "accuracy": 0.8},
            ]
        ),
        encoding="utf-8",
    )

    plot_metric_history_svg(
        metrics_path=metrics_path,
        output_path=output_path,
        metric_keys={"train loss": "loss"},
        title="Train Loss",
        y_label="Loss",
    )

    svg = output_path.read_text(encoding="utf-8")

    assert svg.startswith("<svg")
    assert "Train Loss" in svg
    assert "train loss 0.4000" in svg
    assert 'stroke="#e5e7eb"' in svg


def test_plot_metric_history_svg_writes_train_vs_val_svg(tmp_path: Path) -> None:
    metrics_path = tmp_path / "train_metrics.json"
    output_path = tmp_path / "plots" / "loss_train_vs_val.svg"
    metrics_path.write_text(
        json.dumps(
            [
                {"epoch": 1, "train_loss": 0.7, "val_loss": 0.8},
                {"epoch": 2, "train_loss": 0.4, "val_loss": 0.5},
            ]
        ),
        encoding="utf-8",
    )

    plot_metric_history_svg(
        metrics_path=metrics_path,
        output_path=output_path,
        metric_keys={"train": "train_loss", "val": "val_loss"},
        title="Loss Train vs Val",
        y_label="Loss",
    )

    svg = output_path.read_text(encoding="utf-8")

    assert "Loss Train vs Val" in svg
    assert "train 0.4000" in svg
    assert "val 0.5000" in svg


def test_plot_metric_history_svg_rejects_empty_series(tmp_path: Path) -> None:
    metrics_path = tmp_path / "train_metrics.json"
    output_path = tmp_path / "plot.svg"
    metrics_path.write_text(
        json.dumps([{"epoch": 1, "loss": 0.7}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metric_keys"):
        plot_metric_history_svg(
            metrics_path=metrics_path,
            output_path=output_path,
            metric_keys={},
            title="Empty",
            y_label="Metric",
        )


def test_plot_confusion_matrix_svg_writes_svg(tmp_path: Path) -> None:
    metrics_path = tmp_path / "test_metrics.json"
    output_path = tmp_path / "plots" / "confusion_matrix.svg"
    metrics_path.write_text(
        json.dumps(
            {
                "true_negatives": 8,
                "false_positives": 2,
                "false_negatives": 1,
                "true_positives": 9,
            }
        ),
        encoding="utf-8",
    )

    plot_confusion_matrix_svg(metrics_path, output_path)

    svg = output_path.read_text(encoding="utf-8")

    assert svg.startswith("<svg")
    assert "Confusion Matrix" in svg
    assert "TN" in svg
    assert "FP" in svg
    assert "FN" in svg
    assert "TP" in svg


def test_plot_confusion_matrix_svg_rejects_missing_keys(tmp_path: Path) -> None:
    metrics_path = tmp_path / "test_metrics.json"
    output_path = tmp_path / "plots" / "confusion_matrix.svg"
    metrics_path.write_text(json.dumps({"true_negatives": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing keys"):
        plot_confusion_matrix_svg(metrics_path, output_path)


def test_plot_roc_curve_svg_writes_auc(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.csv"
    output_path = tmp_path / "plots" / "roc_curve.svg"
    predictions_path.write_text(
        "\n".join(
            [
                "label_idx,prob_fake",
                "0,0.1",
                "0,0.4",
                "1,0.35",
                "1,0.8",
            ]
        ),
        encoding="utf-8",
    )

    plot_roc_curve_svg(predictions_path, output_path)

    svg = output_path.read_text(encoding="utf-8")
    assert svg.startswith("<svg")
    assert "ROC Curve" in svg
    assert "AUC = 0.7500" in svg
    assert "False positive rate" in svg


def test_plot_roc_curve_svg_requires_both_classes(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.csv"
    predictions_path.write_text(
        "label_idx,prob_fake\n1,0.8\n1,0.9\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires both classes"):
        plot_roc_curve_svg(predictions_path, tmp_path / "roc.svg")


def test_plot_roc_comparison_svg_writes_all_series(tmp_path: Path) -> None:
    predictions_path = tmp_path / "ensemble.csv"
    output_path = tmp_path / "plots" / "comparison.svg"
    predictions_path.write_text(
        "\n".join(
            [
                "label_idx,audio_prob_fake,video_prob_fake,ensemble_prob_fake",
                "0,0.1,0.2,0.15",
                "0,0.4,0.3,0.35",
                "1,0.35,0.7,0.525",
                "1,0.8,0.9,0.85",
            ]
        ),
        encoding="utf-8",
    )

    plot_roc_comparison_svg(
        {
            "Audio": (predictions_path, "audio_prob_fake"),
            "Video": (predictions_path, "video_prob_fake"),
            "Ensemble": (predictions_path, "ensemble_prob_fake"),
        },
        output_path,
    )

    svg = output_path.read_text(encoding="utf-8")
    assert "Audio: AUC = 0.7500" in svg
    assert "Video: AUC = 1.0000" in svg
    assert "Ensemble: AUC = 1.0000" in svg
