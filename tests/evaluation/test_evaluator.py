import csv
import json
from pathlib import Path

import torch
from torch import nn

from src.evaluation.evaluator import (
    evaluate_audio_classifier,
    evaluate_audio_video_ensemble,
    evaluate_video_classifier,
    write_ensemble_evaluation_outputs,
    write_evaluation_outputs,
)


class MeanScoreFeatureExtractor(nn.Module):
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return audio


class MeanScoreModel(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scores = features.mean(dim=(1, 2))
        return torch.stack([-scores, scores], dim=1)


class MeanFrameScoreModel(nn.Module):
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        scores = frames.mean(dim=(1, 2, 3, 4))
        return torch.stack([-scores, scores], dim=1)


def make_eval_batches() -> list[dict]:
    return [
        {
            "audio": torch.tensor(
                [
                    [[-1.0, -1.0, -1.0]],
                    [[1.0, 1.0, 1.0]],
                ]
            ),
            "label": torch.tensor([0, 1], dtype=torch.long),
            "clip_path": ["clip_real", "clip_fake"],
            "video_id": ["video_real", "video_fake"],
            "clip_id": ["000000", "000001"],
        }
    ]


def make_video_eval_batches() -> list[dict]:
    return [
        {
            "frames": torch.tensor(
                [
                    [[[[0.0, 0.0], [0.0, 0.0]]]],
                    [[[[1.0, 1.0], [1.0, 1.0]]]],
                ]
            ),
            "label": torch.tensor([0, 1], dtype=torch.long),
            "clip_path": ["clip_real", "clip_fake"],
            "video_id": ["video_real", "video_fake"],
            "clip_id": ["000000", "000001"],
        }
    ]


def make_ensemble_eval_batches() -> list[dict]:
    batch = make_eval_batches()[0]
    batch["frames"] = torch.tensor(
        [
            [[[[1.0, 1.0], [1.0, 1.0]]]],
            [[[[0.0, 0.0], [0.0, 0.0]]]],
        ]
    )
    return [batch]


def test_evaluate_audio_classifier_returns_metrics_and_predictions() -> None:
    result = evaluate_audio_classifier(
        model=MeanScoreModel(),
        feature_extractor=MeanScoreFeatureExtractor(),
        dataloader=make_eval_batches(),
        device=torch.device("cpu"),
    )

    assert result.metrics.accuracy == 1.0
    assert result.metrics.precision == 1.0
    assert result.metrics.recall == 1.0
    assert result.metrics.f1 == 1.0
    assert result.metrics.auc == 1.0
    assert len(result.predictions) == 2
    assert result.predictions[0].label == "real"
    assert result.predictions[0].pred_label == "real"
    assert result.predictions[1].label == "fake"
    assert result.predictions[1].pred_label == "fake"


def test_evaluate_audio_classifier_can_limit_batches() -> None:
    batches = make_eval_batches() + make_eval_batches()

    result = evaluate_audio_classifier(
        model=MeanScoreModel(),
        feature_extractor=MeanScoreFeatureExtractor(),
        dataloader=batches,
        device=torch.device("cpu"),
        max_batches=1,
    )

    assert result.metrics.num_samples == 2
    assert len(result.predictions) == 2


def test_evaluate_video_classifier_returns_metrics_and_predictions() -> None:
    result = evaluate_video_classifier(
        model=MeanFrameScoreModel(),
        dataloader=make_video_eval_batches(),
        device=torch.device("cpu"),
    )

    assert result.metrics.accuracy == 1.0
    assert result.metrics.f1 == 1.0
    assert len(result.predictions) == 2
    assert result.predictions[0].label == "real"
    assert result.predictions[1].pred_label == "fake"


def test_evaluate_audio_video_ensemble_reports_disagreements() -> None:
    result = evaluate_audio_video_ensemble(
        audio_model=MeanScoreModel(),
        video_model=MeanFrameScoreModel(),
        feature_extractor=MeanScoreFeatureExtractor(),
        dataloader=make_ensemble_eval_batches(),
        device=torch.device("cpu"),
    )

    assert result.audio_metrics.accuracy == 1.0
    assert result.video_metrics.accuracy == 0.0
    assert result.agreement_count == 0
    assert result.disagreement_count == 2
    assert result.audio_fake_video_real_count == 1
    assert result.audio_real_video_fake_count == 1
    assert not result.predictions[0].models_agree


def test_write_evaluation_outputs_writes_csv_and_json(tmp_path: Path) -> None:
    result = evaluate_audio_classifier(
        model=MeanScoreModel(),
        feature_extractor=MeanScoreFeatureExtractor(),
        dataloader=make_eval_batches(),
        device=torch.device("cpu"),
    )
    predictions_path = tmp_path / "predictions.csv"
    metrics_path = tmp_path / "metrics.json"

    write_evaluation_outputs(result, predictions_path, metrics_path)

    with predictions_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert len(rows) == 2
    assert rows[0]["video_id"] == "video_real"
    assert rows[1]["pred_label"] == "fake"
    assert metrics["accuracy"] == 1.0
    assert metrics["num_samples"] == 2


def test_write_ensemble_evaluation_outputs_writes_comparison(tmp_path: Path) -> None:
    result = evaluate_audio_video_ensemble(
        audio_model=MeanScoreModel(),
        video_model=MeanFrameScoreModel(),
        feature_extractor=MeanScoreFeatureExtractor(),
        dataloader=make_ensemble_eval_batches(),
        device=torch.device("cpu"),
    )
    predictions_path = tmp_path / "ensemble_predictions.csv"
    metrics_path = tmp_path / "ensemble_metrics.json"
    comparison_path = tmp_path / "comparison.json"

    write_ensemble_evaluation_outputs(
        result,
        predictions_path,
        metrics_path,
        comparison_path,
    )

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    with predictions_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 2
    assert rows[0]["models_agree"] == "False"
    assert comparison["agreement_rate"] == 0.0
    assert comparison["audio_metrics"]["accuracy"] == 1.0
