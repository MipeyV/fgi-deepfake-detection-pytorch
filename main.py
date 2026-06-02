import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from src.config import load_config, validate_audio_baseline_config
from src.data.audio_feature import build_audio_feature_extractor
from src.data.dataloader import create_dataloader
from src.data.preprocessing_pipeline import preprocess_dataset, write_manifest
from src.models.audio_models import build_audio_model
from src.runs import create_run_context
from src.training.trainer import build_optimizer, resolve_device, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FGI deepfake detection pipeline")

    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess real and fake videos into synchronized audio-visual clips.",
    )

    preprocess_parser.add_argument("--real-dir", type=Path, required=True)
    preprocess_parser.add_argument("--fake-dir", type=Path, required=True)
    preprocess_parser.add_argument("--output-dir", type=Path, required=True)
    preprocess_parser.add_argument("--fps", type=int, default=30)
    preprocess_parser.add_argument("--clip-size", type=int, default=30)
    preprocess_parser.add_argument("--sample-rate", type=int, default=48000)

    train_parser = subparsers.add_parser(
        "train",
        help="Train an audio-only baseline from a YAML config.",
    )
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--max-batches", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--run-id", type=str, default=None)
    train_parser.add_argument("--runs-root", type=Path, default=None)
    train_parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def write_json(path: Path, payload: dict | list) -> None:
    """Write JSON data with stable formatting.

    Args:
        path: Destination path.
        payload: JSON-serializable payload to write.
    """
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def train_audio_baseline(args: argparse.Namespace) -> None:
    """Train the baseline audio model from command-line arguments.

    Args:
        args: Parsed command-line arguments for the ``train`` subcommand.

    Raises:
        FileNotFoundError: If the config or train manifest does not exist.
        ValueError: If the config is invalid or no training samples are
            processed.
    """
    config = load_config(args.config)
    validate_audio_baseline_config(config)

    run_context = create_run_context(
        config=config,
        config_path=args.config,
        runs_root=args.runs_root,
        run_id=args.run_id,
    )

    training_config = config["training"]
    audio_config = config["audio"]
    device_name = args.device or training_config["device"]
    epochs = args.epochs or training_config["epochs"]
    batch_size = args.batch_size or training_config["batch_size"]
    device = resolve_device(device_name)

    train_loader = create_dataloader(
        manifest_path=config["data"]["train_manifest"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=training_config["num_workers"],
        include_frames=False,
    )
    feature_extractor = build_audio_feature_extractor(
        features_config=config["features"],
        sample_rate=audio_config["sample_rate"],
    )
    model = build_audio_model(config["model"])
    optimizer = build_optimizer(model, training_config["optimizer"])
    criterion = torch.nn.CrossEntropyLoss()

    history = []

    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(
            model=model,
            feature_extractor=feature_extractor,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_batches=args.max_batches,
        )
        epoch_metrics = {"epoch": epoch, **asdict(metrics)}
        history.append(epoch_metrics)

        print(
            "epoch "
            f"{epoch}/{epochs} "
            f"loss={metrics.loss:.4f} "
            f"accuracy={metrics.accuracy:.4f} "
            f"samples={metrics.num_samples}"
        )

    metrics_path = run_context.metrics_dir / "train_metrics.json"
    write_json(metrics_path, history)

    print(f"Run directory: {run_context.run_dir}")
    print(f"Training metrics: {metrics_path}")


def main() -> None:
    args = parse_args()

    if args.command == "preprocess":
        clip_paths = preprocess_dataset(
            real_dir=args.real_dir,
            fake_dir=args.fake_dir,
            output_dir=args.output_dir,
            fps=args.fps,
            clip_size=args.clip_size,
            sample_rate=args.sample_rate,
        )

        write_manifest(
            clip_paths=clip_paths,
            manifest_path=args.output_dir / "manifest.csv",
        )

        print(f"Preprocessing complete. Created {len(clip_paths)} clips.")
        print(f"Manifest written to {args.output_dir / 'manifest.csv'}")

    if args.command == "train":
        train_audio_baseline(args)


if __name__ == "__main__":
    main()
