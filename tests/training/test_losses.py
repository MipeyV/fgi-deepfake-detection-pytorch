from pathlib import Path

import pytest
import torch

from src.training.losses import (
    build_classification_criterion,
    compute_balanced_class_weights,
)


def write_manifest(path: Path, labels: list[str]) -> None:
    rows = ["clip_path,label,video_id,clip_id"]
    rows.extend(
        f"/tmp/clip-{index},{label},video-{index},{index:06d}"
        for index, label in enumerate(labels)
    )
    path.write_text("\n".join(rows), encoding="utf-8")


def test_compute_balanced_class_weights_uses_inverse_frequency(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "train.csv"
    write_manifest(manifest_path, ["real", "fake", "fake", "fake"])

    weights = compute_balanced_class_weights(
        manifest_path,
        {"real": 0, "fake": 1},
    )

    assert torch.allclose(weights, torch.tensor([2.0, 2.0 / 3.0]))


def test_build_classification_criterion_accepts_manual_weights(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "train.csv"
    write_manifest(manifest_path, ["real", "fake"])

    criterion = build_classification_criterion(
        {"name": "cross_entropy", "class_weights": [3.0, 1.0]},
        manifest_path,
        {"real": 0, "fake": 1},
        torch.device("cpu"),
    )

    assert torch.equal(criterion.weight, torch.tensor([3.0, 1.0]))


def test_compute_balanced_class_weights_rejects_missing_class(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "train.csv"
    write_manifest(manifest_path, ["fake", "fake"])

    with pytest.raises(ValueError, match="class 'real' is absent"):
        compute_balanced_class_weights(
            manifest_path,
            {"real": 0, "fake": 1},
        )
