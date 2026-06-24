import pytest
import torch
from torch import nn

from src.models.baselines.video import VideoCNNBaseline
from src.training.trainer import (
    build_optimizer,
    evaluate_video_model,
    train_video_one_epoch,
)


def make_video_batches() -> list[dict]:
    return [
        {
            "frames": torch.randn(4, 3, 3, 8, 8),
            "label": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        },
        {
            "frames": torch.randn(4, 3, 3, 8, 8),
            "label": torch.tensor([1, 1, 0, 0], dtype=torch.long),
        },
    ]


def make_video_model() -> VideoCNNBaseline:
    return VideoCNNBaseline(
        input_channels=3,
        num_classes=2,
        conv_channels=(4,),
        dense_channels=(8,),
        dropout=0.0,
    )


def test_train_video_one_epoch_updates_model_and_returns_metrics() -> None:
    model = make_video_model()
    optimizer = build_optimizer(model, {"name": "adam", "learning_rate": 0.01})
    criterion = nn.CrossEntropyLoss()
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = train_video_one_epoch(
        model=model,
        dataloader=make_video_batches(),
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    after = list(model.parameters())

    assert metrics.num_samples == 8
    assert metrics.num_batches == 2
    assert metrics.loss > 0
    assert 0 <= metrics.accuracy <= 1
    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))


def test_evaluate_video_model_does_not_update_model() -> None:
    model = make_video_model()
    criterion = nn.CrossEntropyLoss()
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = evaluate_video_model(
        model=model,
        dataloader=make_video_batches(),
        criterion=criterion,
        device=torch.device("cpu"),
    )

    after = list(model.parameters())

    assert metrics.num_samples == 8
    assert metrics.num_batches == 2
    assert all(torch.equal(old, new) for old, new in zip(before, after, strict=True))


def test_train_video_one_epoch_rejects_missing_batch_keys() -> None:
    model = make_video_model()
    optimizer = build_optimizer(model, {"name": "adam", "learning_rate": 0.01})
    criterion = nn.CrossEntropyLoss()

    with pytest.raises(ValueError, match="missing required keys"):
        train_video_one_epoch(
            model=model,
            dataloader=[{"label": torch.tensor([0], dtype=torch.long)}],
            optimizer=optimizer,
            criterion=criterion,
            device=torch.device("cpu"),
        )
