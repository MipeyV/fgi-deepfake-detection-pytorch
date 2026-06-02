import csv
import json
from pathlib import Path

import torch
from torch import nn

from src.evaluation.evaluator import (
    evaluate_audio_classifier,
    write_evaluation_outputs,
)


class MeanScoreFeatureExtractor(nn.Module):
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return audio


class MeanScoreModel(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scores = features.mean(dim=(1, 2))
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
