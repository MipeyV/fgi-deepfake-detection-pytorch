import argparse
import json
import shutil
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
    evaluate_audio_video_ensemble,
    evaluate_video_classifier,
    write_ensemble_evaluation_outputs,
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
    load_training_checkpoint,
    save_training_checkpoint,
)
from src.training.early_stopping import build_early_stopping_state
from src.training.losses import build_classification_criterion
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
    train_parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Continue training from a checkpoint; --epochs remains the total target.",
    )

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

    ensemble_parser = subparsers.add_parser(
        "ensemble-eval",
        help="Compare audio and video checkpoints and evaluate their ensemble.",
    )
    ensemble_parser.add_argument("--config", type=Path, required=True)
    ensemble_parser.add_argument("--audio-checkpoint", type=Path, required=True)
    ensemble_parser.add_argument("--video-checkpoint", type=Path, required=True)
    ensemble_parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
    )
    ensemble_parser.add_argument("--max-batches", type=int, default=None)
    ensemble_parser.add_argument("--batch-size", type=int, default=None)
    ensemble_parser.add_argument("--run-id", type=str, default=None)
    ensemble_parser.add_argument("--runs-root", type=Path, default=None)
    ensemble_parser.add_argument("--device", type=str, default=None)

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


def _write_test_evaluation(
    result,
    run_context,
) -> None:
    """Write test metrics, predictions, and confusion matrix into a training run."""
    predictions_path = run_context.predictions_dir / "test_predictions.csv"
    metrics_path = run_context.metrics_dir / "test_metrics.json"
    confusion_matrix_path = run_context.plots_dir / "test_confusion_matrix.svg"
    write_evaluation_outputs(result, predictions_path, metrics_path)
    plot_confusion_matrix_svg(metrics_path, confusion_matrix_path)
    print(
        "test "
        f"accuracy={result.metrics.accuracy:.4f} "
        f"f1={result.metrics.f1:.4f} "
        f"samples={result.metrics.num_samples}"
    )
    print(f"Test metrics: {metrics_path}")
    print(f"Test predictions: {predictions_path}")
    print(f"Test confusion matrix: {confusion_matrix_path}")


def _restore_training_state(
    model,
    optimizer,
    resume_path: Path | None,
    run_context,
) -> tuple[int, list[dict], str | None, float | None]:
    """Restore a checkpoint and any available metric history."""
    if resume_path is None:
        return 0, [], None, None

    checkpoint = load_training_checkpoint(model, optimizer, resume_path)
    start_epoch = int(checkpoint.get("epoch", 0))
    history_path = resume_path.parent.parent / "metrics" / "train_metrics.json"

    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        history = [
            row for row in history if int(row.get("epoch", 0)) <= start_epoch
        ]
    else:
        metrics = checkpoint.get("metrics")
        history = [metrics] if isinstance(metrics, dict) else []

    source_best_path = resume_path.parent / "best.pt"
    if not source_best_path.is_file() and resume_path.name == "best.pt":
        source_best_path = resume_path
    if source_best_path.is_file():
        shutil.copy2(source_best_path, run_context.checkpoints_dir / "best.pt")

    print(f"Resuming training from epoch {start_epoch}: {resume_path}")
    return (
        start_epoch,
        history,
        checkpoint.get("best_metric_name"),
        checkpoint.get("best_metric_value"),
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
    criterion = build_classification_criterion(
        loss_config=training_config["loss"],
        train_manifest_path=config["data"]["train_manifest"],
        label_mapping=config["data"]["label_mapping"],
        device=device,
    )
    if criterion.weight is not None:
        print(f"Class weights: {criterion.weight.detach().cpu().tolist()}")

    resume_path = getattr(args, "resume", None)
    start_epoch, history, resumed_metric_name, best_metric_value = (
        _restore_training_state(model, optimizer, resume_path, run_context)
    )
    metrics_path = run_context.metrics_dir / "train_metrics.json"
    if history:
        write_json(metrics_path, history)
    checkpointing_config = config["checkpointing"]
    save_last = checkpointing_config.get("save_last", True)
    save_best = checkpointing_config.get("save_best", True)
    best_metric_name = resumed_metric_name or config["validation"].get(
        "metric_for_best_checkpoint",
        "val_loss" if val_loader is not None else "loss",
    )
    early_stopping = build_early_stopping_state(training_config, best_metric_name)

    for epoch in range(start_epoch + 1, epochs + 1):
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
        write_json(metrics_path, history)

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

        should_stop = False
        if early_stopping is not None and early_stopping.update(epoch_metrics):
            epoch_metrics["early_stopped"] = True
            print(
                "early stopping triggered "
                f"after {early_stopping.bad_epochs} non-improving epochs "
                f"on {early_stopping.metric_name}"
            )
            should_stop = True

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

        if should_stop:
            break

    if not history:
        raise ValueError(
            f"No epochs to train: checkpoint epoch {start_epoch}, target {epochs}"
        )
    write_json(metrics_path, history)
    _write_training_plots(metrics_path, run_context.plots_dir)
    _print_training_outputs(run_context.run_dir, metrics_path, run_context.plots_dir)

    if config["evaluation"].get("auto_after_training", False):
        test_loader = create_dataloader(
            manifest_path=config["data"]["test_manifest"],
            batch_size=config["evaluation"]["batch_size"],
            shuffle=False,
            num_workers=training_config["num_workers"],
            include_frames=False,
        )
        checkpoint_path = run_context.checkpoints_dir / "best.pt"
        if checkpoint_path.is_file():
            load_model_checkpoint(model, checkpoint_path)
        result = evaluate_audio_classifier(
            model=model,
            feature_extractor=feature_extractor,
            dataloader=test_loader,
            device=device,
            max_batches=args.max_batches,
        )
        _write_test_evaluation(result, run_context)


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
    criterion = build_classification_criterion(
        loss_config=training_config["loss"],
        train_manifest_path=config["data"]["train_manifest"],
        label_mapping=config["data"]["label_mapping"],
        device=device,
    )
    if criterion.weight is not None:
        print(f"Class weights: {criterion.weight.detach().cpu().tolist()}")

    resume_path = getattr(args, "resume", None)
    start_epoch, history, resumed_metric_name, best_metric_value = (
        _restore_training_state(model, optimizer, resume_path, run_context)
    )
    metrics_path = run_context.metrics_dir / "train_metrics.json"
    if history:
        write_json(metrics_path, history)
    checkpointing_config = config["checkpointing"]
    save_last = checkpointing_config.get("save_last", True)
    save_best = checkpointing_config.get("save_best", True)
    best_metric_name = resumed_metric_name or config["validation"].get(
        "metric_for_best_checkpoint",
        "val_loss" if val_loader is not None else "loss",
    )
    early_stopping = build_early_stopping_state(training_config, best_metric_name)

    for epoch in range(start_epoch + 1, epochs + 1):
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
        write_json(metrics_path, history)

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

        should_stop = False
        if early_stopping is not None and early_stopping.update(epoch_metrics):
            epoch_metrics["early_stopped"] = True
            print(
                "early stopping triggered "
                f"after {early_stopping.bad_epochs} non-improving epochs "
                f"on {early_stopping.metric_name}"
            )
            should_stop = True

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

        if should_stop:
            break

    if not history:
        raise ValueError(
            f"No epochs to train: checkpoint epoch {start_epoch}, target {epochs}"
        )
    write_json(metrics_path, history)
    _write_training_plots(metrics_path, run_context.plots_dir)
    _print_training_outputs(run_context.run_dir, metrics_path, run_context.plots_dir)

    if config["evaluation"].get("auto_after_training", False):
        test_loader = create_dataloader(
            manifest_path=config["data"]["test_manifest"],
            batch_size=config["evaluation"]["batch_size"],
            shuffle=False,
            num_workers=training_config["num_workers"],
            frame_transform=frame_transform,
            include_frames=True,
        )
        checkpoint_path = run_context.checkpoints_dir / "best.pt"
        if checkpoint_path.is_file():
            load_model_checkpoint(model, checkpoint_path)
        result = evaluate_video_classifier(
            model=model,
            dataloader=test_loader,
            device=device,
            max_batches=args.max_batches,
        )
        _write_test_evaluation(result, run_context)


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


def evaluate_audio_video_baseline(args: argparse.Namespace) -> None:
    """Compare audio/video predictions and evaluate mean-probability fusion."""
    ensemble_config = load_config(args.config)
    audio_config = load_config(ensemble_config["audio_config"])
    video_config = load_config(ensemble_config["video_config"])
    validate_audio_baseline_config(audio_config)
    validate_video_baseline_config(video_config)

    split = args.split or ensemble_config["evaluation"].get("split", "test")
    data_key = f"{split}_manifest"
    audio_manifest = Path(audio_config["data"][data_key])
    video_manifest = Path(video_config["data"][data_key])
    if audio_manifest.resolve() != video_manifest.resolve():
        raise ValueError(
            "Audio and video ensemble configs must use the same split manifest"
        )

    run_context = create_run_context(
        config=ensemble_config,
        config_path=args.config,
        runs_root=args.runs_root,
        run_id=args.run_id,
    )
    device_name = args.device or audio_config["training"]["device"]
    device = resolve_device(device_name)
    batch_size = (
        args.batch_size
        or ensemble_config["evaluation"]["batch_size"]
    )
    frame_transform = build_frame_resize_transform(
        video_config["video"]["frame_size"]
    )
    dataloader = create_dataloader(
        manifest_path=audio_manifest,
        batch_size=batch_size,
        shuffle=False,
        num_workers=video_config["training"]["num_workers"],
        frame_transform=frame_transform,
        include_frames=True,
    )
    feature_extractor = build_audio_feature_extractor(
        features_config=audio_config["features"],
        sample_rate=audio_config["audio"]["sample_rate"],
    )
    audio_model = build_audio_model(audio_config["model"])
    video_model = build_video_model(video_config["model"])
    load_model_checkpoint(audio_model, args.audio_checkpoint)
    load_model_checkpoint(video_model, args.video_checkpoint)

    result = evaluate_audio_video_ensemble(
        audio_model=audio_model,
        video_model=video_model,
        feature_extractor=feature_extractor,
        dataloader=dataloader,
        device=device,
        max_batches=args.max_batches,
    )
    predictions_path = (
        run_context.predictions_dir / f"{split}_ensemble_comparison.csv"
    )
    metrics_path = run_context.metrics_dir / f"{split}_ensemble_metrics.json"
    comparison_path = run_context.metrics_dir / f"{split}_model_comparison.json"
    confusion_matrix_path = (
        run_context.plots_dir / f"{split}_ensemble_confusion_matrix.svg"
    )
    write_ensemble_evaluation_outputs(
        result,
        predictions_path,
        metrics_path,
        comparison_path,
    )
    plot_confusion_matrix_svg(metrics_path, confusion_matrix_path)

    print(
        f"{split} agreement={result.agreement_rate:.4f} "
        f"disagreements={result.disagreement_count}/{len(result.predictions)}"
    )
    print(
        f"audio accuracy={result.audio_metrics.accuracy:.4f} "
        f"video accuracy={result.video_metrics.accuracy:.4f} "
        f"ensemble accuracy={result.ensemble_metrics.accuracy:.4f}"
    )
    print(f"Run directory: {run_context.run_dir}")
    print(f"Comparison metrics: {comparison_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Ensemble confusion matrix: {confusion_matrix_path}")


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

    if args.command == "ensemble-eval":
        evaluate_audio_video_baseline(args)


if __name__ == "__main__":
    main()
