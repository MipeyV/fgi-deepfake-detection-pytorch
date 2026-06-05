from argparse import Namespace
from pathlib import Path

import yaml

from main import evaluate_audio_baseline, train_audio_baseline
from tests.data.helpers import create_clip


def write_train_config(tmp_path: Path, manifest_path: Path) -> Path:
    config = {
        "experiment": {
            "name": "baseline_audio",
            "version": 1,
            "seed": 42,
            "runs_root": str(tmp_path / "runs"),
            "output_dir": str(tmp_path / "runs" / "baseline_audio"),
        },
        "data": {
            "manifest_dir": str(tmp_path),
            "train_manifest": str(manifest_path),
            "val_manifest": str(manifest_path),
            "test_manifest": str(manifest_path),
            "label_mapping": {"real": 0, "fake": 1},
        },
        "audio": {
            "source_file": "audio.wav",
            "sample_rate": 48000,
            "duration_seconds": 1.0,
            "mono": True,
            "normalize": True,
        },
        "features": {
            "type": "mel_spectrogram",
            "n_mels": 16,
            "n_fft": 512,
            "hop_length": 256,
            "win_length": 512,
            "f_min": 20,
            "f_max": 24000,
            "power": 2.0,
            "log_scale": True,
        },
        "model": {
            "name": "audio_cnn_baseline",
            "input_channels": 1,
            "num_classes": 2,
            "conv_channels": [4],
            "dense_channels": [8],
            "dropout": 0.0,
        },
        "training": {
            "device": "cpu",
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "optimizer": {
                "name": "adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
            },
            "loss": {"name": "cross_entropy"},
        },
        "validation": {
            "batch_size": 2,
            "interval_epochs": 1,
            "metric_for_best_checkpoint": "val_loss",
        },
        "evaluation": {
            "batch_size": 2,
            "metrics": ["accuracy", "f1"],
        },
        "checkpointing": {
            "save_dir": "checkpoints",
            "save_best": True,
            "save_last": True,
        },
        "logging": {
            "log_dir": "logs",
            "level": "info",
            "tensorboard": False,
        },
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_train_audio_baseline_writes_run_metrics(tmp_path: Path) -> None:
    real_clip = tmp_path / "clips" / "real" / "000000"
    fake_clip = tmp_path / "clips" / "fake" / "000000"
    create_clip(real_clip)
    create_clip(fake_clip)

    manifest_path = tmp_path / "train_manifest.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "clip_path,label,video_id,clip_id",
                f"{real_clip},real,video_real,000000",
                f"{fake_clip},fake,video_fake,000000",
            ]
        ),
        encoding="utf-8",
    )
    config_path = write_train_config(tmp_path, manifest_path)

    train_audio_baseline(
        Namespace(
            config=config_path,
            epochs=1,
            max_batches=1,
            batch_size=1,
            run_id="test-run",
            runs_root=tmp_path / "runs",
            device="cpu",
        )
    )

    run_dir = tmp_path / "runs" / "baseline-audio" / "test-run"

    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "metrics" / "train_metrics.json").is_file()
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    assert (run_dir / "plots" / "training_history.svg").is_file()
    assert (run_dir / "plots" / "train_loss.svg").is_file()
    assert (run_dir / "plots" / "train_accuracy.svg").is_file()

    evaluate_audio_baseline(
        Namespace(
            config=config_path,
            split="test",
            checkpoint=run_dir / "checkpoints" / "best.pt",
            max_batches=1,
            batch_size=1,
            run_id="checkpoint-eval-run",
            runs_root=tmp_path / "runs",
            device="cpu",
        )
    )

    eval_run_dir = tmp_path / "runs" / "baseline-audio" / "checkpoint-eval-run"

    assert (eval_run_dir / "metrics" / "test_metrics.json").is_file()
    assert (eval_run_dir / "predictions" / "test_predictions.csv").is_file()


def test_evaluate_audio_baseline_writes_predictions_and_metrics(tmp_path: Path) -> None:
    real_clip = tmp_path / "clips" / "real" / "000000"
    fake_clip = tmp_path / "clips" / "fake" / "000000"
    create_clip(real_clip)
    create_clip(fake_clip)

    manifest_path = tmp_path / "test_manifest.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "clip_path,label,video_id,clip_id",
                f"{real_clip},real,video_real,000000",
                f"{fake_clip},fake,video_fake,000000",
            ]
        ),
        encoding="utf-8",
    )
    config_path = write_train_config(tmp_path, manifest_path)

    evaluate_audio_baseline(
        Namespace(
            config=config_path,
            split="test",
            checkpoint=None,
            max_batches=1,
            batch_size=1,
            run_id="eval-test-run",
            runs_root=tmp_path / "runs",
            device="cpu",
        )
    )

    run_dir = tmp_path / "runs" / "baseline-audio" / "eval-test-run"

    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "metrics" / "test_metrics.json").is_file()
    assert (run_dir / "predictions" / "test_predictions.csv").is_file()
    assert (run_dir / "plots" / "test_confusion_matrix.svg").is_file()
