from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.training.trainer import (
    build_optimizer,
    evaluate_fgi_model,
    train_fgi_one_epoch,
)


class TinyFGIModel(nn.Module):
    """Small multimodal classifier exposing the FGI output contract."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 2)

    def forward(
        self,
        frames: torch.Tensor,
        audio: torch.Tensor,
    ) -> SimpleNamespace:
        features = torch.stack(
            [
                frames.mean(dim=(1, 2, 3, 4)),
                audio.mean(dim=1),
            ],
            dim=1,
        )
        return SimpleNamespace(logits=self.classifier(features))


def make_fgi_batches() -> list[dict]:
    """Return two lightweight synchronized batches."""
    return [
        {
            "frames": torch.randn(2, 3, 3, 4, 4),
            "audio": torch.randn(2, 32),
            "label": torch.tensor([0, 1], dtype=torch.long),
        },
        {
            "frames": torch.randn(2, 3, 3, 4, 4),
            "audio": torch.randn(2, 32),
            "label": torch.tensor([1, 0], dtype=torch.long),
        },
    ]


def test_train_fgi_one_epoch_updates_model() -> None:
    model = TinyFGIModel()
    optimizer = build_optimizer(model, {"name": "adam", "learning_rate": 0.01})
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = train_fgi_one_epoch(
        model=model,
        dataloader=make_fgi_batches(),
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )

    assert metrics.num_samples == 4
    assert metrics.num_batches == 2
    assert metrics.loss > 0
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters())
    )


def test_evaluate_fgi_model_does_not_update_model() -> None:
    model = TinyFGIModel()
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = evaluate_fgi_model(
        model=model,
        dataloader=make_fgi_batches(),
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )

    assert metrics.num_samples == 4
    assert all(
        torch.equal(old, new)
        for old, new in zip(before, model.parameters())
    )


def test_train_fgi_one_epoch_rejects_missing_audio() -> None:
    model = TinyFGIModel()
    optimizer = build_optimizer(model, {"name": "adam"})

    with pytest.raises(ValueError, match="missing required keys"):
        train_fgi_one_epoch(
            model=model,
            dataloader=[
                {
                    "frames": torch.randn(1, 3, 3, 4, 4),
                    "label": torch.tensor([0]),
                }
            ],
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
        )
