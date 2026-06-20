import pytest
import torch
from torch import nn

from src.models.baselines.audio.audio_models import AudioCNNBaseline
from src.training.trainer import (
    build_optimizer,
    evaluate_audio_model,
    resolve_device,
    train_one_epoch,
)


class ToyFeatureExtractor(nn.Module):
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return audio.unsqueeze(2).repeat(1, 1, 8, 1)


def make_batches() -> list[dict]:
    return [
        {
            "audio": torch.randn(4, 1, 16),
            "label": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        },
        {
            "audio": torch.randn(4, 1, 16),
            "label": torch.tensor([1, 1, 0, 0], dtype=torch.long),
        },
    ]


def test_resolve_device_returns_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_build_optimizer_builds_adam() -> None:
    model = nn.Linear(2, 2)

    optimizer = build_optimizer(
        model,
        {"name": "adam", "learning_rate": 0.001, "weight_decay": 0.0},
    )

    assert isinstance(optimizer, torch.optim.Adam)


def test_build_optimizer_rejects_unknown_optimizer() -> None:
    model = nn.Linear(2, 2)

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        build_optimizer(model, {"name": "rmsprop"})


def test_train_one_epoch_updates_model_and_returns_metrics() -> None:
    model = AudioCNNBaseline(
        input_channels=1,
        num_classes=2,
        conv_channels=(4,),
        dense_channels=(8,),
        dropout=0.0,
    )
    feature_extractor = ToyFeatureExtractor()
    optimizer = build_optimizer(model, {"name": "adam", "learning_rate": 0.01})
    criterion = nn.CrossEntropyLoss()
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = train_one_epoch(
        model=model,
        feature_extractor=feature_extractor,
        dataloader=make_batches(),
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    after = list(model.parameters())

    assert metrics.num_samples == 8
    assert metrics.num_batches == 2
    assert metrics.loss > 0
    assert 0 <= metrics.accuracy <= 1
    assert any(not torch.equal(old, new) for old, new in zip(before, after))


def test_train_one_epoch_can_limit_number_of_batches() -> None:
    model = AudioCNNBaseline(
        input_channels=1,
        num_classes=2,
        conv_channels=(4,),
        dense_channels=(8,),
        dropout=0.0,
    )
    feature_extractor = ToyFeatureExtractor()
    optimizer = build_optimizer(model, {"name": "adam", "learning_rate": 0.01})
    criterion = nn.CrossEntropyLoss()

    metrics = train_one_epoch(
        model=model,
        feature_extractor=feature_extractor,
        dataloader=make_batches(),
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        max_batches=1,
    )

    assert metrics.num_samples == 4
    assert metrics.num_batches == 1


def test_evaluate_audio_model_does_not_update_model() -> None:
    model = AudioCNNBaseline(
        input_channels=1,
        num_classes=2,
        conv_channels=(4,),
        dense_channels=(8,),
        dropout=0.0,
    )
    feature_extractor = ToyFeatureExtractor()
    criterion = nn.CrossEntropyLoss()
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = evaluate_audio_model(
        model=model,
        feature_extractor=feature_extractor,
        dataloader=make_batches(),
        criterion=criterion,
        device=torch.device("cpu"),
    )

    after = list(model.parameters())

    assert metrics.num_samples == 8
    assert metrics.num_batches == 2
    assert all(torch.equal(old, new) for old, new in zip(before, after))


def test_train_one_epoch_rejects_missing_batch_keys() -> None:
    model = AudioCNNBaseline(
        input_channels=1,
        num_classes=2,
        conv_channels=(4,),
        dense_channels=(8,),
        dropout=0.0,
    )
    feature_extractor = ToyFeatureExtractor()
    optimizer = build_optimizer(model, {"name": "adam", "learning_rate": 0.01})
    criterion = nn.CrossEntropyLoss()

    with pytest.raises(ValueError, match="missing required keys"):
        train_one_epoch(
            model=model,
            feature_extractor=feature_extractor,
            dataloader=[{"audio": torch.randn(4, 1, 16)}],
            optimizer=optimizer,
            criterion=criterion,
            device=torch.device("cpu"),
        )
