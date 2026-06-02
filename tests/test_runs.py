import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.runs import create_run_context, generate_run_id, slugify


def write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "experiment:\n"
        "  name: Baseline Audio\n"
        "  runs_root: runs\n",
        encoding="utf-8",
    )
    return config_path


def test_slugify_normalizes_text() -> None:
    assert slugify("Baseline Audio v1") == "baseline-audio-v1"


def test_slugify_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="slug"):
        slugify("!!!")


def test_generate_run_id_uses_timestamp_experiment_and_commit() -> None:
    run_id = generate_run_id(
        experiment_name="Baseline Audio",
        created_at=datetime(2026, 6, 2, 12, 30, 4, tzinfo=timezone.utc),
        git_commit="abcdef123456",
    )

    assert run_id == "20260602-123004_baseline-audio_abcdef1"


def test_create_run_context_creates_expected_run_files(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    config = {
        "experiment": {
            "name": "Baseline Audio",
            "runs_root": str(tmp_path / "runs"),
        }
    }

    context = create_run_context(
        config=config,
        config_path=config_path,
        command="python main.py train --config config.yaml",
        run_id="20260602-123004_baseline-audio_testsha",
    )

    assert context.run_dir.is_dir()
    assert context.checkpoints_dir.is_dir()
    assert context.logs_dir.is_dir()
    assert context.metrics_dir.is_dir()
    assert context.predictions_dir.is_dir()
    assert context.plots_dir.is_dir()
    assert context.config_path.read_text(encoding="utf-8") == config_path.read_text(
        encoding="utf-8"
    )
    assert context.command_path.read_text(encoding="utf-8").strip() == (
        "python main.py train --config config.yaml"
    )

    git_snapshot = json.loads(context.git_path.read_text(encoding="utf-8"))
    metadata = json.loads(context.metadata_path.read_text(encoding="utf-8"))

    assert {"branch", "commit", "is_dirty"} <= set(git_snapshot)
    assert metadata["run_id"] == context.run_id
    assert metadata["experiment_name"] == "Baseline Audio"


def test_create_run_context_rejects_existing_run_dir(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    config = {
        "experiment": {
            "name": "Baseline Audio",
            "runs_root": str(tmp_path / "runs"),
        }
    }

    create_run_context(
        config=config,
        config_path=config_path,
        run_id="same-run",
    )

    with pytest.raises(FileExistsError):
        create_run_context(
            config=config,
            config_path=config_path,
            run_id="same-run",
        )


def test_create_run_context_requires_experiment_name(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)

    with pytest.raises(ValueError, match="experiment.name"):
        create_run_context(config={"experiment": {}}, config_path=config_path)
