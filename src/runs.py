"""Run directory management for reproducible experiments."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


__all__ = [
    "GitSnapshot",
    "RunContext",
    "create_run_context",
    "generate_run_id",
    "get_git_snapshot",
    "slugify",
]


@dataclass(frozen=True)
class GitSnapshot:
    """Git state captured when a run is created.

    Args:
        branch: Current Git branch name.
        commit: Current Git commit SHA.
        is_dirty: Whether the working tree has uncommitted changes.
    """

    branch: str
    commit: str
    is_dirty: bool


@dataclass(frozen=True)
class RunContext:
    """Paths and metadata for one experiment run.

    Args:
        run_id: Unique run identifier.
        experiment_name: Experiment name from the config.
        run_dir: Root directory for this run.
        config_path: Copied config path inside the run directory.
        command_path: Path storing the command used to launch the run.
        git_path: Path storing Git branch, commit, and dirty state.
        metadata_path: Path storing run metadata as JSON.
        checkpoints_dir: Directory reserved for model checkpoints.
        logs_dir: Directory reserved for logs.
        metrics_dir: Directory reserved for metrics files.
        predictions_dir: Directory reserved for prediction CSV files.
    """

    run_id: str
    experiment_name: str
    run_dir: Path
    config_path: Path
    command_path: Path
    git_path: Path
    metadata_path: Path
    checkpoints_dir: Path
    logs_dir: Path
    metrics_dir: Path
    predictions_dir: Path


def slugify(value: str) -> str:
    """Convert a string into a filesystem-friendly slug.

    Args:
        value: String to normalize.

    Returns:
        Lowercase slug containing only letters, digits, and single dashes.

    Raises:
        ValueError: If ``value`` does not contain any slug-compatible character.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)

    if not slug:
        raise ValueError("Cannot create a slug from an empty value")

    return slug


def get_git_snapshot(repo_dir: str | Path = ".") -> GitSnapshot:
    """Capture the current Git branch, commit, and dirty state.

    Args:
        repo_dir: Repository directory where Git commands should run.

    Returns:
        Current Git snapshot. Unknown values are set to ``"unknown"`` when Git
        metadata cannot be read.
    """
    repo_dir = Path(repo_dir)

    def run_git(args: list[str]) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        branch = run_git(["branch", "--show-current"]) or "unknown"
        commit = run_git(["rev-parse", "HEAD"]) or "unknown"
        status = run_git(["status", "--short"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return GitSnapshot(branch="unknown", commit="unknown", is_dirty=True)

    return GitSnapshot(branch=branch, commit=commit, is_dirty=bool(status))


def generate_run_id(
    experiment_name: str,
    created_at: datetime | None = None,
    git_commit: str | None = None,
) -> str:
    """Generate a reproducible-looking run identifier.

    Args:
        experiment_name: Human-readable experiment name.
        created_at: Optional timestamp used for deterministic tests.
        git_commit: Optional Git commit SHA included as a short suffix.

    Returns:
        Run ID formatted as ``YYYYMMDD-HHMMSS_experiment_shortsha``.
    """
    timestamp = created_at or datetime.now(timezone.utc)
    experiment_slug = slugify(experiment_name)
    short_commit = (git_commit or "unknown")[:7]

    return f"{timestamp:%Y%m%d-%H%M%S}_{experiment_slug}_{short_commit}"


def _default_command() -> str:
    """Return the current Python command as a shell-like string.

    Returns:
        Command string built from ``sys.argv``.
    """
    return " ".join(sys.argv)


def _write_json(path: Path, payload: dict) -> None:
    """Write a JSON file with stable indentation.

    Args:
        path: Destination path.
        payload: JSON-serializable dictionary.
    """
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def create_run_context(
    config: dict,
    config_path: str | Path,
    runs_root: str | Path | None = None,
    command: str | None = None,
    run_id: str | None = None,
    repo_dir: str | Path = ".",
) -> RunContext:
    """Create a run directory and persist launch metadata.

    Args:
        config: Loaded experiment configuration.
        config_path: Original YAML config path to copy into the run directory.
        runs_root: Optional root directory for all runs. Defaults to
            ``config["experiment"]["runs_root"]`` when present, otherwise
            ``"runs"``.
        command: Optional command string to write into ``command.txt``.
        run_id: Optional explicit run ID. If omitted, one is generated.
        repo_dir: Repository directory used to capture Git metadata.

    Returns:
        Run context with all important output paths.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        ValueError: If the config does not contain an experiment name.
        FileExistsError: If the target run directory already exists.
    """
    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    experiment = config.get("experiment", {})
    experiment_name = experiment.get("name")

    if not experiment_name:
        raise ValueError("Config must contain experiment.name")

    git_snapshot = get_git_snapshot(repo_dir=repo_dir)
    run_id = run_id or generate_run_id(
        experiment_name=experiment_name,
        git_commit=git_snapshot.commit,
    )
    runs_root = Path(runs_root or experiment.get("runs_root", "runs"))
    run_dir = runs_root / slugify(experiment_name) / run_id

    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"

    for directory in (checkpoints_dir, logs_dir, metrics_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=False)

    copied_config_path = run_dir / "config.yaml"
    command_path = run_dir / "command.txt"
    git_path = run_dir / "git.json"
    metadata_path = run_dir / "metadata.json"

    shutil.copy2(config_path, copied_config_path)
    command_path.write_text((command or _default_command()) + "\n", encoding="utf-8")
    _write_json(git_path, asdict(git_snapshot))
    _write_json(
        metadata_path,
        {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "original_config_path": str(config_path),
        },
    )

    return RunContext(
        run_id=run_id,
        experiment_name=experiment_name,
        run_dir=run_dir,
        config_path=copied_config_path,
        command_path=command_path,
        git_path=git_path,
        metadata_path=metadata_path,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        metrics_dir=metrics_dir,
        predictions_dir=predictions_dir,
    )
