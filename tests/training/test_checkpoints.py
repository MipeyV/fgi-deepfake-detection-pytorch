from pathlib import Path

import torch
from torch import nn

from src.training.checkpoints import (
    checkpoint_metric_is_better,
    load_model_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)


def test_checkpoint_metric_is_better_minimizes_loss() -> None:
    assert checkpoint_metric_is_better(0.4, None, "val_loss")
    assert checkpoint_metric_is_better(0.4, 0.5, "val_loss")
    assert not checkpoint_metric_is_better(0.6, 0.5, "val_loss")


def test_checkpoint_metric_is_better_maximizes_other_metrics() -> None:
    assert checkpoint_metric_is_better(0.8, None, "val_accuracy")
    assert checkpoint_metric_is_better(0.8, 0.7, "val_accuracy")
    assert not checkpoint_metric_is_better(0.6, 0.7, "val_accuracy")


def test_save_training_checkpoint_can_be_loaded_into_model(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    checkpoint_path = tmp_path / "best.pt"

    save_training_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        metrics={"val_loss": 0.25},
        config={"experiment": {"name": "test"}},
        best_metric_name="val_loss",
        best_metric_value=0.25,
    )

    restored_model = nn.Linear(2, 2)
    checkpoint = load_model_checkpoint(restored_model, checkpoint_path)

    assert checkpoint["epoch"] == 3
    assert checkpoint["metrics"] == {"val_loss": 0.25}
    assert checkpoint["best_metric_name"] == "val_loss"
    assert checkpoint["best_metric_value"] == 0.25
    for original, restored in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(original, restored)


def test_load_training_checkpoint_restores_optimizer(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    checkpoint_path = tmp_path / "last.pt"
    save_training_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=2,
        metrics={"loss": 0.5},
    )

    restored_model = nn.Linear(2, 2)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.001)
    checkpoint = load_training_checkpoint(
        restored_model,
        restored_optimizer,
        checkpoint_path,
    )

    assert checkpoint["epoch"] == 2
    assert restored_optimizer.state_dict()["state"]
