from pathlib import Path

import pandas as pd
import pytest
from tests.data.helpers import write_split_test_manifest

from src.data.split_manifest import (
    assign_split_from_hashmod,
    hash_video_id,
    split_manifest,
)


def test_hash_video_id_is_stable() -> None:
    assert hash_video_id("video_a") == hash_video_id("video_a")
    assert hash_video_id("video_b") == hash_video_id("video_b")


def test_assign_split_from_hashmod() -> None:
    assert assign_split_from_hashmod(0) == "train"
    assert assign_split_from_hashmod(6) == "train"
    assert assign_split_from_hashmod(7) == "val"
    assert assign_split_from_hashmod(8) == "val"
    assert assign_split_from_hashmod(9) == "test"


def test_split_manifest_creates_train_val_test(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    output_dir = tmp_path / "splits"
    write_split_test_manifest(manifest_path)

    split_paths = split_manifest(manifest_path, output_dir)

    assert set(split_paths) == {"train", "val", "test"}
    assert split_paths["train"].is_file()
    assert split_paths["val"].is_file()
    assert split_paths["test"].is_file()


def test_split_manifest_keeps_same_video_in_one_split(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    output_dir = tmp_path / "splits"
    write_split_test_manifest(manifest_path)

    split_manifest(manifest_path, output_dir)

    split_dataframes = [
        pd.read_csv(output_dir / "train_manifest.csv"),
        pd.read_csv(output_dir / "val_manifest.csv"),
        pd.read_csv(output_dir / "test_manifest.csv"),
    ]
    samples = pd.concat(split_dataframes, ignore_index=True)

    video_splits = samples.groupby("video_id")["split"].nunique()

    assert video_splits.max() == 1


def test_split_manifest_adds_split_metadata_columns(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    output_dir = tmp_path / "splits"
    write_split_test_manifest(manifest_path)

    split_manifest(manifest_path, output_dir)

    samples = pd.concat(
        [
            pd.read_csv(output_dir / "train_manifest.csv"),
            pd.read_csv(output_dir / "val_manifest.csv"),
            pd.read_csv(output_dir / "test_manifest.csv"),
        ],
        ignore_index=True,
    )

    assert "video_id_hashmod10" in samples.columns
    assert "split" in samples.columns


def test_split_manifest_raises_for_missing_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "missing_manifest.csv"

    with pytest.raises(FileNotFoundError, match="Manifest file does not exist"):
        split_manifest(manifest_path, tmp_path / "splits")


def test_split_manifest_raises_for_missing_columns(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "clip_path,label,video_id",
                "clips/video_a/000000,real,video_a",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Manifest is missing required columns"):
        split_manifest(manifest_path, tmp_path / "splits")
