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
from src.data.fgi_face_crops import (
    OpenCVHaarFaceDetector,
    OpenCVYuNetFaceDetector,
    create_fgi_face_crop_dataset,
    write_face_crop_contact_sheet,
)
from src.data.fgi import build_fgi_input_pipeline
from src.data.fgi_multimodal import validate_fgi_multimodal_config
from src.data.preprocessing_pipeline import preprocess_dataset, write_manifest
from src.data.video import build_video_input_pipeline
from src.evaluation.evaluator import (
    calibrate_threshold_from_predictions,
    evaluate_audio_classifier,
    evaluate_audio_video_ensemble,
    evaluate_fgi_classifier,
    evaluate_video_classifier,
    write_ensemble_evaluation_outputs,
    write_evaluation_outputs,
)
from src.evaluation.metrics import ThresholdCalibrationResult
from src.evaluation.plots import (
    plot_confusion_matrix_svg,
    plot_metric_history_svg,
    plot_training_history_svg,
)
from src.models.audio_models import build_audio_model
from src.models.fgi import build_fgi_encoders, build_fgi_model
from src.models.video import build_video_model
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
    evaluate_fgi_model,
    evaluate_video_model,
    resolve_device,
    train_fgi_one_epoch,
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

    face_crops_parser = subparsers.add_parser(
        "fgi-face-crops",
        help="Create stable face crops from an existing clip manifest.",
    )
    face_crops_parser.add_argument("--manifest", type=Path, required=True)
    face_crops_parser.add_argument("--output-dir", type=Path, required=True)
    face_crops_parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
    )
    face_crops_parser.add_argument("--output-size", type=int, default=256)
    face_crops_parser.add_argument("--margin", type=float, default=0.3)
    face_crops_parser.add_argument(
        "--detector",
        choices=["yunet", "haar"],
        default="yunet",
    )
    face_crops_parser.add_argument("--detector-model", type=Path, default=None)
    face_crops_parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.9,
    )
    face_crops_parser.add_argument(
        "--min-detection-fraction",
        type=float,
        default=0.5,
    )
    face_crops_parser.add_argument(
        "--missing-face-policy",
        choices=["error", "skip"],
        default="error",
    )
    face_crops_parser.add_argument("--contact-sheet", type=Path, default=None)

    fgi_smoke_parser = subparsers.add_parser(
        "fgi-data-smoke",
        help="Load one strict synchronized FGI batch without a model.",
    )
    fgi_smoke_parser.add_argument("--config", type=Path, required=True)
    fgi_smoke_parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
    )
    fgi_smoke_parser.add_argument("--batch-size", type=int, default=1)

    fgi_encoder_parser = subparsers.add_parser(
        "fgi-encoder-smoke",
        help="Run one synchronized batch through the FGI encoders.",
    )
    fgi_encoder_parser.add_argument("--config", type=Path, required=True)
    fgi_encoder_parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
    )
    fgi_encoder_parser.add_argument("--batch-size", type=int, default=1)
    fgi_encoder_parser.add_argument("--device", type=str, default="cpu")

    fgi_model_parser = subparsers.add_parser(
        "fgi-model-smoke",
        help="Run one synchronized batch through the complete FGI model.",
    )
    fgi_model_parser.add_argument("--config", type=Path, required=True)
    fgi_model_parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
    )
    fgi_model_parser.add_argument("--batch-size", type=int, default=1)
    fgi_model_parser.add_argument("--device", type=str, default="cpu")

    train_parser = subparsers.add_parser(
        "train",
        help="Train a supported model from a YAML config.",
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
        help="Evaluate a supported model from a YAML config.",
    )
    eval_parser.add_argument("--config", type=Path, required=True)
    eval_parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    eval_parser.add_argument("--checkpoint", type=Path, default=None)
    eval_parser.add_argument("--max-batches", type=int, default=None)
    eval_parser.add_argument("--batch-size", type=int, default=None)
    eval_parser.add_argument("--run-id", type=str, default=None)
    eval_parser.add_argument("--runs-root", type=Path, default=None)
    eval_parser.add_argument("--device", type=str, default=None)
    eval_parser.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help="Override the configured fake-class decision threshold.",
    )

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
    calibration: ThresholdCalibrationResult | None = None,
) -> None:
    """Write test metrics, predictions, and confusion matrix into a training run."""
    predictions_path = run_context.predictions_dir / "test_predictions.csv"
    metrics_path = run_context.metrics_dir / "test_metrics.json"
    video_predictions_path = (
        run_context.predictions_dir / "test_video_predictions.csv"
    )
    video_metrics_path = run_context.metrics_dir / "test_video_metrics.json"
    confusion_matrix_path = run_context.plots_dir / "test_confusion_matrix.svg"
    write_evaluation_outputs(
        result,
        predictions_path,
        metrics_path,
        video_predictions_path,
        video_metrics_path,
    )
    if calibration is not None:
        write_json(
            run_context.metrics_dir / "threshold_calibration.json",
            asdict(calibration),
        )
    plot_confusion_matrix_svg(metrics_path, confusion_matrix_path)
    print(
        "test "
        f"accuracy={result.metrics.accuracy:.4f} "
        f"balanced_accuracy={result.metrics.balanced_accuracy:.4f} "
        f"f1={result.metrics.f1:.4f} "
        f"threshold={result.decision_threshold:.4f} "
        f"samples={result.metrics.num_samples}"
    )
    print(f"Test metrics: {metrics_path}")
    print(f"Test predictions: {predictions_path}")
    if result.video_metrics is not None:
        print(f"Test video metrics: {video_metrics_path}")
        print(f"Test video predictions: {video_predictions_path}")
    print(f"Test confusion matrix: {confusion_matrix_path}")


def _configured_decision_threshold(
    config: dict,
    override: float | None = None,
) -> float | None:
    """Return a fixed threshold, or ``None`` when calibration is requested."""
    if override is not None:
        threshold = float(override)
    else:
        configured = config["evaluation"].get("decision_threshold", 0.5)
        if configured == "calibrated":
            return None
        threshold = float(configured)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("evaluation.decision_threshold must be in [0, 1]")
    return threshold


def _calibrate_from_result(
    result,
    config: dict,
    allow_incomplete: bool = False,
) -> ThresholdCalibrationResult | None:
    metric_name = config["evaluation"].get(
        "calibration_metric",
        "balanced_accuracy",
    )
    try:
        return calibrate_threshold_from_predictions(
            result.predictions,
            metric_name=metric_name,
        )
    except ValueError as error:
        if allow_incomplete and "requires both classes" in str(error):
            print(
                "Threshold calibration skipped because the limited validation "
                "batches do not contain both classes; using 0.5"
            )
            return None
        raise


def _write_split_evaluation(
    result,
    run_context,
    split: str,
    calibration: ThresholdCalibrationResult | None = None,
) -> tuple[Path, Path, Path]:
    predictions_path = run_context.predictions_dir / f"{split}_predictions.csv"
    metrics_path = run_context.metrics_dir / f"{split}_metrics.json"
    video_predictions_path = (
        run_context.predictions_dir / f"{split}_video_predictions.csv"
    )
    video_metrics_path = run_context.metrics_dir / f"{split}_video_metrics.json"
    confusion_matrix_path = run_context.plots_dir / f"{split}_confusion_matrix.svg"
    write_evaluation_outputs(
        result,
        predictions_path,
        metrics_path,
        video_predictions_path,
        video_metrics_path,
    )
    if calibration is not None:
        write_json(
            run_context.metrics_dir / "threshold_calibration.json",
            asdict(calibration),
        )
    plot_confusion_matrix_svg(metrics_path, confusion_matrix_path)
    return predictions_path, metrics_path, confusion_matrix_path


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
        decision_threshold = _configured_decision_threshold(config)
        calibration = None
        if decision_threshold is None:
            if val_loader is None:
                raise ValueError(
                    "Threshold calibration requires a validation manifest"
                )
            validation_result = evaluate_audio_classifier(
                model=model,
                feature_extractor=feature_extractor,
                dataloader=val_loader,
                device=device,
                max_batches=args.max_batches,
            )
            calibration = _calibrate_from_result(
                validation_result,
                config,
                allow_incomplete=args.max_batches is not None,
            )
            decision_threshold = (
                calibration.threshold if calibration is not None else 0.5
            )
        result = evaluate_audio_classifier(
            model=model,
            feature_extractor=feature_extractor,
            dataloader=test_loader,
            device=device,
            max_batches=args.max_batches,
            decision_threshold=decision_threshold,
        )
        _write_test_evaluation(result, run_context, calibration)


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
    input_pipeline = build_video_input_pipeline(config["video"])

    train_loader = input_pipeline.create_dataloader(
        manifest_path=config["data"]["train_manifest"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=training_config["num_workers"],
    )
    val_manifest_path = Path(config["data"]["val_manifest"])
    val_loader = None
    if val_manifest_path.is_file():
        val_loader = input_pipeline.create_dataloader(
            manifest_path=val_manifest_path,
            batch_size=config["validation"]["batch_size"],
            shuffle=False,
            num_workers=training_config["num_workers"],
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
        test_loader = input_pipeline.create_dataloader(
            manifest_path=config["data"]["test_manifest"],
            batch_size=config["evaluation"]["batch_size"],
            shuffle=False,
            num_workers=training_config["num_workers"],
        )
        checkpoint_path = run_context.checkpoints_dir / "best.pt"
        if checkpoint_path.is_file():
            load_model_checkpoint(model, checkpoint_path)
        decision_threshold = _configured_decision_threshold(config)
        calibration = None
        if decision_threshold is None:
            if val_loader is None:
                raise ValueError(
                    "Threshold calibration requires a validation manifest"
                )
            validation_result = evaluate_video_classifier(
                model=model,
                dataloader=val_loader,
                device=device,
                max_batches=args.max_batches,
            )
            calibration = _calibrate_from_result(
                validation_result,
                config,
                allow_incomplete=args.max_batches is not None,
            )
            decision_threshold = (
                calibration.threshold if calibration is not None else 0.5
            )
        result = evaluate_video_classifier(
            model=model,
            dataloader=test_loader,
            device=device,
            max_batches=args.max_batches,
            decision_threshold=decision_threshold,
        )
        _write_test_evaluation(result, run_context, calibration)


def train_fgi_classifier(args: argparse.Namespace) -> None:
    """Train the synchronized audio-video FGI classifier."""
    config = load_config(args.config)
    validate_fgi_multimodal_config(config)

    run_context = create_run_context(
        config=config,
        config_path=args.config,
        runs_root=args.runs_root,
        run_id=args.run_id,
    )
    training_config = config["training"]
    device = resolve_device(args.device or training_config["device"])
    epochs = args.epochs or training_config["epochs"]
    batch_size = args.batch_size or training_config["batch_size"]
    pipeline = build_fgi_input_pipeline(config)

    train_loader = pipeline.create_dataloader(
        manifest_path=config["data"]["train_manifest"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=training_config["num_workers"],
    )
    val_manifest_path = Path(config["data"]["val_manifest"])
    val_loader = None
    if val_manifest_path.is_file():
        val_loader = pipeline.create_dataloader(
            manifest_path=val_manifest_path,
            batch_size=config["validation"]["batch_size"],
            shuffle=False,
            num_workers=training_config["num_workers"],
        )

    model = build_fgi_model(config["model"])
    optimizer = build_optimizer(model, training_config["optimizer"])
    criterion = build_classification_criterion(
        loss_config=training_config["loss"],
        train_manifest_path=config["data"]["train_manifest"],
        label_mapping=config["data"]["label_mapping"],
        device=device,
    )
    if criterion.weight is not None:
        print(f"Class weights: {criterion.weight.detach().cpu().tolist()}")

    start_epoch, history, resumed_metric_name, best_metric_value = (
        _restore_training_state(
            model,
            optimizer,
            getattr(args, "resume", None),
            run_context,
        )
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
    if val_loader is None and best_metric_name.startswith("val_"):
        best_metric_name = "loss"
    early_stopping = build_early_stopping_state(
        training_config,
        best_metric_name,
    )

    for epoch in range(start_epoch + 1, epochs + 1):
        metrics = train_fgi_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_batches=args.max_batches,
        )
        epoch_metrics = {"epoch": epoch, **asdict(metrics)}

        if val_loader is not None:
            val_metrics = evaluate_fgi_model(
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
        test_loader = pipeline.create_dataloader(
            manifest_path=config["data"]["test_manifest"],
            batch_size=config["evaluation"]["batch_size"],
            shuffle=False,
            num_workers=training_config["num_workers"],
        )
        checkpoint_path = run_context.checkpoints_dir / "best.pt"
        if checkpoint_path.is_file():
            load_model_checkpoint(model, checkpoint_path)
        decision_threshold = _configured_decision_threshold(config)
        calibration = None
        if decision_threshold is None:
            if val_loader is None:
                raise ValueError(
                    "Threshold calibration requires a validation manifest"
                )
            validation_result = evaluate_fgi_classifier(
                model=model,
                dataloader=val_loader,
                device=device,
                max_batches=args.max_batches,
            )
            calibration = _calibrate_from_result(
                validation_result,
                config,
                allow_incomplete=args.max_batches is not None,
            )
            decision_threshold = (
                calibration.threshold if calibration is not None else 0.5
            )
        result = evaluate_fgi_classifier(
            model=model,
            dataloader=test_loader,
            device=device,
            max_batches=args.max_batches,
            decision_threshold=decision_threshold,
        )
        _write_test_evaluation(result, run_context, calibration)


def train_fgi_classifier(args: argparse.Namespace) -> None:
    """Train the synchronized audio-video FGI classifier."""
    config = load_config(args.config)
    validate_fgi_multimodal_config(config)

    run_context = create_run_context(
        config=config,
        config_path=args.config,
        runs_root=args.runs_root,
        run_id=args.run_id,
    )
    training_config = config["training"]
    device = resolve_device(args.device or training_config["device"])
    epochs = args.epochs or training_config["epochs"]
    batch_size = args.batch_size or training_config["batch_size"]
    pipeline = build_fgi_input_pipeline(config)

    train_loader = pipeline.create_dataloader(
        manifest_path=config["data"]["train_manifest"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=training_config["num_workers"],
    )
    val_manifest_path = Path(config["data"]["val_manifest"])
    val_loader = None
    if val_manifest_path.is_file():
        val_loader = pipeline.create_dataloader(
            manifest_path=val_manifest_path,
            batch_size=config["validation"]["batch_size"],
            shuffle=False,
            num_workers=training_config["num_workers"],
        )

    model = build_fgi_model(config["model"])
    optimizer = build_optimizer(model, training_config["optimizer"])
    criterion = build_classification_criterion(
        loss_config=training_config["loss"],
        train_manifest_path=config["data"]["train_manifest"],
        label_mapping=config["data"]["label_mapping"],
        device=device,
    )
    if criterion.weight is not None:
        print(f"Class weights: {criterion.weight.detach().cpu().tolist()}")

    start_epoch, history, resumed_metric_name, best_metric_value = (
        _restore_training_state(
            model,
            optimizer,
            getattr(args, "resume", None),
            run_context,
        )
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
    if val_loader is None and best_metric_name.startswith("val_"):
        best_metric_name = "loss"
    early_stopping = build_early_stopping_state(
        training_config,
        best_metric_name,
    )

    for epoch in range(start_epoch + 1, epochs + 1):
        metrics = train_fgi_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_batches=args.max_batches,
        )
        epoch_metrics = {"epoch": epoch, **asdict(metrics)}

        if val_loader is not None:
            val_metrics = evaluate_fgi_model(
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
        test_loader = pipeline.create_dataloader(
            manifest_path=config["data"]["test_manifest"],
            batch_size=config["evaluation"]["batch_size"],
            shuffle=False,
            num_workers=training_config["num_workers"],
        )
        checkpoint_path = run_context.checkpoints_dir / "best.pt"
        if checkpoint_path.is_file():
            load_model_checkpoint(model, checkpoint_path)
        result = evaluate_fgi_classifier(
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

    decision_threshold = _configured_decision_threshold(
        config,
        getattr(args, "decision_threshold", None),
    )
    calibration = None
    if decision_threshold is None:
        validation_loader = create_dataloader(
            manifest_path=config["data"]["val_manifest"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=config["training"]["num_workers"],
            include_frames=False,
        )
        validation_result = evaluate_audio_classifier(
            model=model,
            feature_extractor=feature_extractor,
            dataloader=validation_loader,
            device=device,
            max_batches=args.max_batches,
        )
        calibration = _calibrate_from_result(
            validation_result,
            config,
            allow_incomplete=args.max_batches is not None,
        )
        decision_threshold = (
            calibration.threshold if calibration is not None else 0.5
        )
    result = evaluate_audio_classifier(
        model=model,
        feature_extractor=feature_extractor,
        dataloader=dataloader,
        device=device,
        max_batches=args.max_batches,
        decision_threshold=decision_threshold,
    )
    predictions_path, metrics_path, confusion_matrix_path = (
        _write_split_evaluation(
            result,
            run_context,
            args.split,
            calibration,
        )
    )

    print(
        f"{args.split} "
        f"accuracy={result.metrics.accuracy:.4f} "
        f"balanced_accuracy={result.metrics.balanced_accuracy:.4f} "
        f"f1={result.metrics.f1:.4f} "
        f"threshold={result.decision_threshold:.4f} "
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
    input_pipeline = build_video_input_pipeline(config["video"])

    dataloader = input_pipeline.create_dataloader(
        manifest_path=manifest_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )
    model = build_video_model(config["model"])

    if args.checkpoint is not None:
        load_model_checkpoint(model, args.checkpoint)

    decision_threshold = _configured_decision_threshold(
        config,
        getattr(args, "decision_threshold", None),
    )
    calibration = None
    if decision_threshold is None:
        validation_loader = input_pipeline.create_dataloader(
            manifest_path=config["data"]["val_manifest"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=config["training"]["num_workers"],
        )
        validation_result = evaluate_video_classifier(
            model=model,
            dataloader=validation_loader,
            device=device,
            max_batches=args.max_batches,
        )
        calibration = _calibrate_from_result(
            validation_result,
            config,
            allow_incomplete=args.max_batches is not None,
        )
        decision_threshold = (
            calibration.threshold if calibration is not None else 0.5
        )
    result = evaluate_video_classifier(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=args.max_batches,
        decision_threshold=decision_threshold,
    )
    predictions_path, metrics_path, confusion_matrix_path = (
        _write_split_evaluation(
            result,
            run_context,
            args.split,
            calibration,
        )
    )

    print(
        f"{args.split} "
        f"accuracy={result.metrics.accuracy:.4f} "
        f"balanced_accuracy={result.metrics.balanced_accuracy:.4f} "
        f"f1={result.metrics.f1:.4f} "
        f"threshold={result.decision_threshold:.4f} "
        f"samples={result.metrics.num_samples}"
    )
    print(f"Run directory: {run_context.run_dir}")
    print(f"Evaluation metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Confusion matrix: {confusion_matrix_path}")


def evaluate_fgi_classifier_run(args: argparse.Namespace) -> None:
    """Evaluate an FGI checkpoint and write standard run artifacts."""
    config = load_config(args.config)
    validate_fgi_multimodal_config(config)
    run_context = create_run_context(
        config=config,
        config_path=args.config,
        runs_root=args.runs_root,
        run_id=args.run_id,
    )

    manifest_path = config["data"][f"{args.split}_manifest"]
    device = resolve_device(args.device or config["training"]["device"])
    batch_size = args.batch_size or config["evaluation"]["batch_size"]
    pipeline = build_fgi_input_pipeline(config)
    dataloader = pipeline.create_dataloader(
        manifest_path=manifest_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )
    model = build_fgi_model(config["model"])
    if args.checkpoint is not None:
        load_model_checkpoint(model, args.checkpoint)

    decision_threshold = _configured_decision_threshold(
        config,
        getattr(args, "decision_threshold", None),
    )
    calibration = None
    if decision_threshold is None:
        validation_loader = pipeline.create_dataloader(
            manifest_path=config["data"]["val_manifest"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=config["training"]["num_workers"],
        )
        validation_result = evaluate_fgi_classifier(
            model=model,
            dataloader=validation_loader,
            device=device,
            max_batches=args.max_batches,
        )
        calibration = _calibrate_from_result(
            validation_result,
            config,
            allow_incomplete=args.max_batches is not None,
        )
        decision_threshold = (
            calibration.threshold if calibration is not None else 0.5
        )
    result = evaluate_fgi_classifier(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=args.max_batches,
        decision_threshold=decision_threshold,
    )
    predictions_path, metrics_path, confusion_matrix_path = (
        _write_split_evaluation(
            result,
            run_context,
            args.split,
            calibration,
        )
    )

    print(
        f"{args.split} "
        f"accuracy={result.metrics.accuracy:.4f} "
        f"balanced_accuracy={result.metrics.balanced_accuracy:.4f} "
        f"f1={result.metrics.f1:.4f} "
        f"threshold={result.decision_threshold:.4f} "
        f"samples={result.metrics.num_samples}"
    )
    print(f"Run directory: {run_context.run_dir}")
    print(f"Evaluation metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Confusion matrix: {confusion_matrix_path}")


def evaluate_fgi_classifier_run(args: argparse.Namespace) -> None:
    """Evaluate an FGI checkpoint and write standard run artifacts."""
    config = load_config(args.config)
    validate_fgi_multimodal_config(config)
    run_context = create_run_context(
        config=config,
        config_path=args.config,
        runs_root=args.runs_root,
        run_id=args.run_id,
    )

    manifest_path = config["data"][f"{args.split}_manifest"]
    device = resolve_device(args.device or config["training"]["device"])
    batch_size = args.batch_size or config["evaluation"]["batch_size"]
    pipeline = build_fgi_input_pipeline(config)
    dataloader = pipeline.create_dataloader(
        manifest_path=manifest_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )
    model = build_fgi_model(config["model"])
    if args.checkpoint is not None:
        load_model_checkpoint(model, args.checkpoint)

    result = evaluate_fgi_classifier(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=args.max_batches,
    )
    predictions_path = run_context.predictions_dir / (
        f"{args.split}_predictions.csv"
    )
    metrics_path = run_context.metrics_dir / f"{args.split}_metrics.json"
    confusion_matrix_path = run_context.plots_dir / (
        f"{args.split}_confusion_matrix.svg"
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
    input_pipeline = build_video_input_pipeline(video_config["video"])
    dataloader = input_pipeline.create_dataloader(
        manifest_path=audio_manifest,
        batch_size=batch_size,
        shuffle=False,
        num_workers=video_config["training"]["num_workers"],
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

    if args.command == "fgi-face-crops":
        if args.detector == "yunet":
            if args.detector_model is None:
                raise ValueError(
                    "--detector-model is required when --detector=yunet"
                )
            detector = OpenCVYuNetFaceDetector(
                model_path=args.detector_model,
                score_threshold=args.score_threshold,
            )
        else:
            detector = OpenCVHaarFaceDetector()

        result = create_fgi_face_crop_dataset(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            output_manifest_path=args.output_manifest,
            detector=detector,
            output_size=args.output_size,
            margin=args.margin,
            min_detection_fraction=args.min_detection_fraction,
            missing_face_policy=args.missing_face_policy,
        )
        print(
            "FGI face crops complete. "
            f"Processed {result.processed_clips} clips; "
            f"skipped {result.skipped_clips}."
        )
        print(f"Manifest written to {result.manifest_path}")
        if args.contact_sheet is not None and result.processed_clips:
            contact_sheet_path = write_face_crop_contact_sheet(
                manifest_path=result.manifest_path,
                output_path=args.contact_sheet,
            )
            print(f"Contact sheet written to {contact_sheet_path}")

    if args.command == "fgi-data-smoke":
        config = load_config(args.config)
        validate_fgi_multimodal_config(config)
        manifest_path = config["data"][f"{args.split}_manifest"]
        pipeline = build_fgi_input_pipeline(config)
        dataloader = pipeline.create_dataloader(
            manifest_path=manifest_path,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=config["training"]["num_workers"],
        )
        batch = next(iter(dataloader))
        print(f"frames shape={tuple(batch['frames'].shape)}")
        print(
            f"frames range=[{batch['frames'].min().item():.4f}, "
            f"{batch['frames'].max().item():.4f}]"
        )
        print(f"audio shape={tuple(batch['audio'].shape)}")
        print(
            f"audio range=[{batch['audio'].min().item():.4f}, "
            f"{batch['audio'].max().item():.4f}]"
        )
        print(f"labels shape={tuple(batch['label'].shape)}")

    if args.command == "fgi-encoder-smoke":
        config = load_config(args.config)
        validate_fgi_multimodal_config(config)
        manifest_path = config["data"][f"{args.split}_manifest"]
        pipeline = build_fgi_input_pipeline(config)
        batch = next(
            iter(
                pipeline.create_dataloader(
                    manifest_path=manifest_path,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=config["training"]["num_workers"],
                )
            )
        )
        device = resolve_device(args.device)
        encoders = build_fgi_encoders(config["model"]).to(device)
        encoders.eval()
        with torch.no_grad():
            video_features, audio_features = encoders(
                batch["frames"].to(device),
                batch["audio"].to(device),
            )
        print(f"video features shape={tuple(video_features.shape)}")
        print(f"audio features shape={tuple(audio_features.shape)}")

    if args.command == "fgi-model-smoke":
        config = load_config(args.config)
        validate_fgi_multimodal_config(config)
        manifest_path = config["data"][f"{args.split}_manifest"]
        pipeline = build_fgi_input_pipeline(config)
        batch = next(
            iter(
                pipeline.create_dataloader(
                    manifest_path=manifest_path,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=config["training"]["num_workers"],
                )
            )
        )
        device = resolve_device(args.device)
        model = build_fgi_model(config["model"]).to(device)
        model.eval()
        with torch.no_grad():
            output = model(
                batch["frames"].to(device),
                batch["audio"].to(device),
            )
        print(f"logits shape={tuple(output.logits.shape)}")
        print(
            "inconsistency map shape="
            f"{tuple(output.inconsistency_map.shape)}"
        )
        if output.attention_map is None:
            print("attention map shape=None")
        else:
            print(f"attention map shape={tuple(output.attention_map.shape)}")

    if args.command == "train":
        config = load_config(args.config)
        model_name = _model_name_from_config(config)
        if model_name == "audio_cnn_baseline":
            train_audio_baseline(args)
        elif model_name in {"video_cnn_baseline", "r3d18"}:
            train_video_baseline(args)
        elif model_name == "fgi_inspired":
            train_fgi_classifier(args)
        else:
            raise ValueError(f"Unsupported model for training: {model_name}")

    if args.command == "eval":
        config = load_config(args.config)
        model_name = _model_name_from_config(config)
        if model_name == "audio_cnn_baseline":
            evaluate_audio_baseline(args)
        elif model_name in {"video_cnn_baseline", "r3d18"}:
            evaluate_video_baseline(args)
        elif model_name == "fgi_inspired":
            evaluate_fgi_classifier_run(args)
        else:
            raise ValueError(f"Unsupported model for evaluation: {model_name}")

    if args.command == "ensemble-eval":
        evaluate_audio_video_baseline(args)


if __name__ == "__main__":
    main()
