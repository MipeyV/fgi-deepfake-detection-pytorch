import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from src.config import (
    load_config,
    validate_audio_baseline_config,
    validate_video_baseline_config,
)
from src.data.audio_feature import build_audio_feature_extractor
from src.data.dataloader import create_dataloader
from src.data.dataset import build_frame_resize_transform
from src.data.preprocessing_pipeline import preprocess_dataset, write_manifest
from src.evaluation.evaluator import (
    evaluate_audio_classifier,
    evaluate_video_classifier,
    write_evaluation_outputs,
)
from src.evaluation.plots import (
    plot_confusion_matrix_svg,
    plot_metric_history_svg,
    plot_training_history_svg,
)
from src.models.audio_models import build_audio_model
from src.models.video_models import build_video_model
from src.runs import create_run_context
from src.training.checkpoints import (
    checkpoint_metric_is_better,
    load_model_checkpoint,
    save_training_checkpoint,
)
from src.training.trainer import (
    build_optimizer,
    evaluate_audio_model,
    evaluate_video_model,
    resolve_device,
    train_one_epoch,
    train_video_one_epoch,
)


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

    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate an audio-only baseline from a YAML config.",
    )
    eval_parser.add_argument("--config", type=Path, required=True)
    eval_parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    eval_parser.add_argument("--checkpoint", type=Path, default=None)
    eval_parser.add_argument("--max-batches", type=int, default=None)
    eval_parser.add_argument("--batch-size", type=int, default=None)
    eval_parser.add_argument("--run-id", type=str, default=None)
    eval_parser.add_argument("--runs-root", type=Path, default=None)
    eval_parser.add_argument("--device", type=str, default=None)

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


def _model_name_from_config(config: dict) -> str:
    return str(config.get("model", {}).get("name", ""))


def _write_training_plots(metrics_path: Path, plots_dir: Path) -> None:
    plot_path = plots_dir / "training_history.svg"
    loss_plot_path = plots_dir / "train_loss.svg"
    accuracy_plot_path = plots_dir / "train_accuracy.svg"
    loss_train_vs_val_plot_path = plots_dir / "loss_train_vs_val.svg"
    accuracy_train_vs_val_plot_path = plots_dir / "accuracy_train_vs_val.svg"

    plot_training_history_svg(metrics_path, plot_path)
    plot_metric_history_svg(
        metrics_path=metrics_path,
        output_path=loss_plot_path,
        metric_keys={"train loss": "loss"},
        title="Train Loss",
        y_label="Loss",
    )
    plot_metric_history_svg(
        metrics_path=metrics_path,
        output_path=accuracy_plot_path,
        metric_keys={"train accuracy": "accuracy"},
        title="Train Accuracy",
        y_label="Accuracy",
    )

    history = json.loads(metrics_path.read_text(encoding="utf-8"))
    if history and {"val_loss", "val_accuracy"} <= set(history[-1]):
        plot_metric_history_svg(
            metrics_path=metrics_path,
            output_path=loss_train_vs_val_plot_path,
            metric_keys={"train": "loss", "val": "val_loss"},
            title="Loss Train vs Val",
            y_label="Loss",
        )
        plot_metric_history_svg(
            metrics_path=metrics_path,
            output_path=accuracy_train_vs_val_plot_path,
            metric_keys={"train": "accuracy", "val": "val_accuracy"},
            title="Accuracy Train vs Val",
            y_label="Accuracy",
        )


def _print_training_outputs(run_dir: Path, metrics_path: Path, plots_dir: Path) -> None:
    print(f"Run directory: {run_dir}")
    print(f"Training metrics: {metrics_path}")
    print(f"Training plot: {plots_dir / 'training_history.svg'}")
    print(f"Train loss plot: {plots_dir / 'train_loss.svg'}")
    print(f"Train accuracy plot: {plots_dir / 'train_accuracy.svg'}")

    loss_train_vs_val_plot_path = plots_dir / "loss_train_vs_val.svg"
    accuracy_train_vs_val_plot_path = plots_dir / "accuracy_train_vs_val.svg"
    if loss_train_vs_val_plot_path.is_file():
        print(f"Loss train vs val plot: {loss_train_vs_val_plot_path}")
        print(f"Accuracy train vs val plot: {accuracy_train_vs_val_plot_path}")


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
    val_manifest_path = Path(config["data"]["val_manifest"])
    val_loader = None
    if val_manifest_path.is_file():
        val_loader = create_dataloader(
            manifest_path=val_manifest_path,
            batch_size=config["validation"]["batch_size"],
            shuffle=False,
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
    checkpointing_config = config["checkpointing"]
    save_last = checkpointing_config.get("save_last", True)
    save_best = checkpointing_config.get("save_best", True)
    best_metric_name = config["validation"].get(
        "metric_for_best_checkpoint",
        "val_loss" if val_loader is not None else "loss",
    )
    best_metric_value = None

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

        if val_loader is not None:
            val_metrics = evaluate_audio_model(
                model=model,
                feature_extractor=feature_extractor,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                max_batches=args.max_batches,
            )
            epoch_metrics.update(
                {
                    "val_loss": val_metrics.loss,
                    "val_accuracy": val_metrics.accuracy,
                    "val_num_samples": val_metrics.num_samples,
                    "val_num_batches": val_metrics.num_batches,
                }
            )

        history.append(epoch_metrics)

        print(
            "epoch "
            f"{epoch}/{epochs} "
            f"loss={metrics.loss:.4f} "
            f"accuracy={metrics.accuracy:.4f} "
            f"samples={metrics.num_samples}"
        )

        metric_name = best_metric_name
        if metric_name not in epoch_metrics:
            metric_name = "val_loss" if "val_loss" in epoch_metrics else "loss"

        current_metric_value = float(epoch_metrics[metric_name])
        if save_best and checkpoint_metric_is_better(
            current_metric_value,
            best_metric_value,
            metric_name,
        ):
            best_metric_name = metric_name
            best_metric_value = current_metric_value
            save_training_checkpoint(
                checkpoint_path=run_context.checkpoints_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_metrics,
                config=config,
                best_metric_name=best_metric_name,
                best_metric_value=best_metric_value,
            )

        if save_last:
            save_training_checkpoint(
                checkpoint_path=run_context.checkpoints_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_metrics,
                config=config,
                best_metric_name=best_metric_name,
                best_metric_value=best_metric_value,
            )

    metrics_path = run_context.metrics_dir / "train_metrics.json"
    write_json(metrics_path, history)
    _write_training_plots(metrics_path, run_context.plots_dir)
    _print_training_outputs(run_context.run_dir, metrics_path, run_context.plots_dir)


def train_video_baseline(args: argparse.Namespace) -> None:
    """Train the baseline video model from command-line arguments."""
    config = load_config(args.config)
    validate_video_baseline_config(config)

    run_context = create_run_context(
        config=config,
        config_path=args.config,
        runs_root=args.runs_root,
        run_id=args.run_id,
    )

    training_config = config["training"]
    device_name = args.device or training_config["device"]
    epochs = args.epochs or training_config["epochs"]
    batch_size = args.batch_size or training_config["batch_size"]
    device = resolve_device(device_name)
    frame_transform = build_frame_resize_transform(config["video"]["frame_size"])

    train_loader = create_dataloader(
        manifest_path=config["data"]["train_manifest"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=training_config["num_workers"],
        frame_transform=frame_transform,
        include_frames=True,
    )
    val_manifest_path = Path(config["data"]["val_manifest"])
    val_loader = None
    if val_manifest_path.is_file():
        val_loader = create_dataloader(
            manifest_path=val_manifest_path,
            batch_size=config["validation"]["batch_size"],
            shuffle=False,
            num_workers=training_config["num_workers"],
            frame_transform=frame_transform,
            include_frames=True,
        )

    model = build_video_model(config["model"])
    optimizer = build_optimizer(model, training_config["optimizer"])
    criterion = torch.nn.CrossEntropyLoss()

    history = []
    checkpointing_config = config["checkpointing"]
    save_last = checkpointing_config.get("save_last", True)
    save_best = checkpointing_config.get("save_best", True)
    best_metric_name = config["validation"].get(
        "metric_for_best_checkpoint",
        "val_loss" if val_loader is not None else "loss",
    )
    best_metric_value = None

    for epoch in range(1, epochs + 1):
        metrics = train_video_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_batches=args.max_batches,
        )
        epoch_metrics = {"epoch": epoch, **asdict(metrics)}

        if val_loader is not None:
            val_metrics = evaluate_video_model(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                max_batches=args.max_batches,
            )
            epoch_metrics.update(
                {
                    "val_loss": val_metrics.loss,
                    "val_accuracy": val_metrics.accuracy,
                    "val_num_samples": val_metrics.num_samples,
                    "val_num_batches": val_metrics.num_batches,
                }
            )

        history.append(epoch_metrics)

        print(
            "epoch "
            f"{epoch}/{epochs} "
            f"loss={metrics.loss:.4f} "
            f"accuracy={metrics.accuracy:.4f} "
            f"samples={metrics.num_samples}"
        )

        metric_name = best_metric_name
        if metric_name not in epoch_metrics:
            metric_name = "val_loss" if "val_loss" in epoch_metrics else "loss"

        current_metric_value = float(epoch_metrics[metric_name])
        if save_best and checkpoint_metric_is_better(
            current_metric_value,
            best_metric_value,
            metric_name,
        ):
            best_metric_name = metric_name
            best_metric_value = current_metric_value
            save_training_checkpoint(
                checkpoint_path=run_context.checkpoints_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_metrics,
                config=config,
                best_metric_name=best_metric_name,
                best_metric_value=best_metric_value,
            )

        if save_last:
            save_training_checkpoint(
                checkpoint_path=run_context.checkpoints_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_metrics,
                config=config,
                best_metric_name=best_metric_name,
                best_metric_value=best_metric_value,
            )

    metrics_path = run_context.metrics_dir / "train_metrics.json"
    write_json(metrics_path, history)
    _write_training_plots(metrics_path, run_context.plots_dir)
    _print_training_outputs(run_context.run_dir, metrics_path, run_context.plots_dir)


def evaluate_audio_baseline(args: argparse.Namespace) -> None:
    """Evaluate the baseline audio model from command-line arguments.

    Args:
        args: Parsed command-line arguments for the ``eval`` subcommand.

    Raises:
        FileNotFoundError: If the config, manifest, or checkpoint does not
            exist.
        ValueError: If the config is invalid or no samples are evaluated.
    """
    config = load_config(args.config)
    validate_audio_baseline_config(config)

    run_context = create_run_context(
        config=config,
        config_path=args.config,
        runs_root=args.runs_root,
        run_id=args.run_id,
    )

    data_key = f"{args.split}_manifest"
    manifest_path = config["data"][data_key]
    device_name = args.device or config["training"]["device"]
    batch_size = args.batch_size or config["evaluation"]["batch_size"]
    device = resolve_device(device_name)

    dataloader = create_dataloader(
        manifest_path=manifest_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        include_frames=False,
    )
    feature_extractor = build_audio_feature_extractor(
        features_config=config["features"],
        sample_rate=config["audio"]["sample_rate"],
    )
    model = build_audio_model(config["model"])

    if args.checkpoint is not None:
        load_model_checkpoint(model, args.checkpoint)

    result = evaluate_audio_classifier(
        model=model,
        feature_extractor=feature_extractor,
        dataloader=dataloader,
        device=device,
        max_batches=args.max_batches,
    )
    predictions_path = run_context.predictions_dir / f"{args.split}_predictions.csv"
    metrics_path = run_context.metrics_dir / f"{args.split}_metrics.json"
    confusion_matrix_path = (
        run_context.plots_dir / f"{args.split}_confusion_matrix.svg"
    )

    write_evaluation_outputs(result, predictions_path, metrics_path)
    plot_confusion_matrix_svg(metrics_path, confusion_matrix_path)

    print(
        f"{args.split} "
        f"accuracy={result.metrics.accuracy:.4f} "
        f"f1={result.metrics.f1:.4f} "
        f"samples={result.metrics.num_samples}"
    )
    print(f"Run directory: {run_context.run_dir}")
    print(f"Evaluation metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Confusion matrix: {confusion_matrix_path}")


def evaluate_video_baseline(args: argparse.Namespace) -> None:
    """Evaluate the baseline video model from command-line arguments."""
    config = load_config(args.config)
    validate_video_baseline_config(config)

    run_context = create_run_context(
        config=config,
        config_path=args.config,
        runs_root=args.runs_root,
        run_id=args.run_id,
    )

    data_key = f"{args.split}_manifest"
    manifest_path = config["data"][data_key]
    device_name = args.device or config["training"]["device"]
    batch_size = args.batch_size or config["evaluation"]["batch_size"]
    device = resolve_device(device_name)
    frame_transform = build_frame_resize_transform(config["video"]["frame_size"])

    dataloader = create_dataloader(
        manifest_path=manifest_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        frame_transform=frame_transform,
        include_frames=True,
    )
    model = build_video_model(config["model"])

    if args.checkpoint is not None:
        load_model_checkpoint(model, args.checkpoint)

    result = evaluate_video_classifier(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=args.max_batches,
    )
    predictions_path = run_context.predictions_dir / f"{args.split}_predictions.csv"
    metrics_path = run_context.metrics_dir / f"{args.split}_metrics.json"
    confusion_matrix_path = (
        run_context.plots_dir / f"{args.split}_confusion_matrix.svg"
    )

    write_evaluation_outputs(result, predictions_path, metrics_path)
    plot_confusion_matrix_svg(metrics_path, confusion_matrix_path)

    print(
        f"{args.split} "
        f"accuracy={result.metrics.accuracy:.4f} "
        f"f1={result.metrics.f1:.4f} "
        f"samples={result.metrics.num_samples}"
    )
    print(f"Run directory: {run_context.run_dir}")
    print(f"Evaluation metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Confusion matrix: {confusion_matrix_path}")


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
        config = load_config(args.config)
        model_name = _model_name_from_config(config)
        if model_name == "audio_cnn_baseline":
            train_audio_baseline(args)
        elif model_name == "video_cnn_baseline":
            train_video_baseline(args)
        else:
            raise ValueError(f"Unsupported model for training: {model_name}")

    if args.command == "eval":
        config = load_config(args.config)
        model_name = _model_name_from_config(config)
        if model_name == "audio_cnn_baseline":
            evaluate_audio_baseline(args)
        elif model_name == "video_cnn_baseline":
            evaluate_video_baseline(args)
        else:
            raise ValueError(f"Unsupported model for evaluation: {model_name}")


if __name__ == "__main__":
    main()
