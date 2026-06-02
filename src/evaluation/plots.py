"""Plot helpers for experiment metrics."""

from __future__ import annotations

import json
from pathlib import Path


__all__ = [
    "load_training_history",
    "plot_confusion_matrix_svg",
    "plot_metric_history_svg",
    "plot_training_history_svg",
]


SERIES_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
]


def load_training_history(metrics_path: str | Path) -> list[dict]:
    """Load training history from a JSON metrics file.

    Args:
        metrics_path: Path to a JSON file containing a list of epoch metrics.

    Returns:
        List of epoch metric dictionaries.

    Raises:
        FileNotFoundError: If ``metrics_path`` does not exist.
        ValueError: If the file does not contain a non-empty list.
    """
    metrics_path = Path(metrics_path)

    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics file does not exist: {metrics_path}")

    history = json.loads(metrics_path.read_text(encoding="utf-8"))

    if not isinstance(history, list) or not history:
        raise ValueError("Training history must be a non-empty list")

    return history


def _scale_values(values: list[float], height: int, padding: int) -> list[float]:
    """Scale values to SVG y coordinates.

    Args:
        values: Metric values to scale.
        height: SVG height in pixels.
        padding: Plot padding in pixels.

    Returns:
        Y coordinates where lower metric values are lower on the plot.
    """
    min_value = min(values)
    max_value = max(values)
    drawable_height = height - 2 * padding

    if max_value == min_value:
        return [height / 2 for _ in values]

    return [
        height
        - padding
        - ((value - min_value) / (max_value - min_value)) * drawable_height
        for value in values
    ]


def _scale_epochs(epochs: list[int], width: int, padding: int) -> list[float]:
    """Scale epoch numbers to SVG x coordinates.

    Args:
        epochs: Epoch numbers to scale.
        width: SVG width in pixels.
        padding: Plot padding in pixels.

    Returns:
        X coordinates.
    """
    drawable_width = width - 2 * padding

    if len(epochs) == 1:
        return [width / 2]

    return [
        padding + index * drawable_width / (len(epochs) - 1)
        for index, _ in enumerate(epochs)
    ]


def _polyline(points: list[tuple[float, float]]) -> str:
    """Format SVG polyline points.

    Args:
        points: List of ``(x, y)`` points.

    Returns:
        SVG-compatible point string.
    """
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _validate_history_keys(history: list[dict], required_keys: set[str]) -> None:
    """Validate that each history row contains required keys.

    Args:
        history: Training history rows.
        required_keys: Keys required in every row.

    Raises:
        ValueError: If at least one row is missing a required key.
    """
    for item in history:
        missing_keys = required_keys - set(item)

        if missing_keys:
            raise ValueError(
                f"Training metric row is missing keys: {sorted(missing_keys)}"
            )


def plot_metric_history_svg(
    metrics_path: str | Path,
    output_path: str | Path,
    metric_keys: dict[str, str],
    title: str,
    y_label: str,
    width: int = 800,
    height: int = 420,
) -> None:
    """Create an SVG plot for one or more metrics over epochs.

    Args:
        metrics_path: Path to ``train_metrics.json``.
        output_path: Destination SVG file.
        metric_keys: Mapping from legend label to metric key in the history.
            For example, ``{"train loss": "loss"}`` or
            ``{"train": "train_loss", "val": "val_loss"}``.
        title: Plot title.
        y_label: Y-axis label.
        width: SVG width in pixels.
        height: SVG height in pixels.

    Raises:
        ValueError: If ``metric_keys`` is empty or required keys are missing.
    """
    if not metric_keys:
        raise ValueError("metric_keys must contain at least one series")

    history = load_training_history(metrics_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_keys = {"epoch", *metric_keys.values()}
    _validate_history_keys(history, required_keys)

    epochs = [int(item["epoch"]) for item in history]
    padding = 60
    x_values = _scale_epochs(epochs, width, padding)
    series_values = {
        label: [float(item[key]) for item in history]
        for label, key in metric_keys.items()
    }
    all_values = [
        value
        for values in series_values.values()
        for value in values
    ]
    y_values = _scale_values(all_values, height, padding)
    y_lookup = {}
    cursor = 0

    for label, values in series_values.items():
        y_lookup[label] = y_values[cursor : cursor + len(values)]
        cursor += len(values)

    series_svg = []
    legend_svg = []

    for series_index, (label, values) in enumerate(series_values.items()):
        color = SERIES_COLORS[series_index % len(SERIES_COLORS)]
        points = list(zip(x_values, y_lookup[label], strict=True))
        series_svg.append(
            f"""
  <polyline points="{_polyline(points)}" fill="none" stroke="{color}" stroke-width="3"/>
  <circle cx="{points[-1][0]:.2f}" cy="{points[-1][1]:.2f}" r="4" fill="{color}"/>"""
        )
        legend_y = padding - 22 + series_index * 20
        legend_svg.append(
            f"""  <text x="{width - padding}" y="{legend_y}" text-anchor="end" font-family="sans-serif" font-size="13" fill="{color}">{label} {values[-1]:.4f}</text>"""
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2:.0f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>
  <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#333"/>
  <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#333"/>
  <text x="{width / 2:.0f}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="13">Epoch</text>
  <text x="20" y="{height / 2:.0f}" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 20 {height / 2:.0f})">{y_label}</text>
{''.join(series_svg)}
{chr(10).join(legend_svg)}
  <text x="{padding}" y="{height - padding + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{epochs[0]}</text>
  <text x="{width - padding}" y="{height - padding + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{epochs[-1]}</text>
</svg>
"""

    output_path.write_text(svg, encoding="utf-8")


def plot_training_history_svg(
    metrics_path: str | Path,
    output_path: str | Path,
    width: int = 800,
    height: int = 420,
) -> None:
    """Create an SVG plot for training loss and accuracy over epochs.

    Args:
        metrics_path: Path to ``train_metrics.json``.
        output_path: Destination SVG file.
        width: SVG width in pixels.
        height: SVG height in pixels.

    Raises:
        ValueError: If required metric keys are missing.
    """
    history = load_training_history(metrics_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_keys = {"epoch", "loss", "accuracy"}
    _validate_history_keys(history, required_keys)

    epochs = [int(item["epoch"]) for item in history]
    loss_values = [float(item["loss"]) for item in history]
    accuracy_values = [float(item["accuracy"]) for item in history]
    padding = 60
    x_values = _scale_epochs(epochs, width, padding)
    loss_y_values = _scale_values(loss_values, height, padding)
    accuracy_y_values = _scale_values(accuracy_values, height, padding)
    loss_points = list(zip(x_values, loss_y_values, strict=True))
    accuracy_points = list(zip(x_values, accuracy_y_values, strict=True))

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2:.0f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">Training History</text>
  <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#333"/>
  <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#333"/>
  <text x="{width / 2:.0f}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="13">Epoch</text>
  <text x="20" y="{height / 2:.0f}" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 20 {height / 2:.0f})">Scaled value</text>
  <polyline points="{_polyline(loss_points)}" fill="none" stroke="#d62728" stroke-width="3"/>
  <polyline points="{_polyline(accuracy_points)}" fill="none" stroke="#1f77b4" stroke-width="3"/>
  <circle cx="{loss_points[-1][0]:.2f}" cy="{loss_points[-1][1]:.2f}" r="4" fill="#d62728"/>
  <circle cx="{accuracy_points[-1][0]:.2f}" cy="{accuracy_points[-1][1]:.2f}" r="4" fill="#1f77b4"/>
  <text x="{width - padding}" y="{padding - 20}" text-anchor="end" font-family="sans-serif" font-size="13" fill="#d62728">loss {loss_values[-1]:.4f}</text>
  <text x="{width - padding}" y="{padding}" text-anchor="end" font-family="sans-serif" font-size="13" fill="#1f77b4">accuracy {accuracy_values[-1]:.4f}</text>
  <text x="{padding}" y="{height - padding + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{epochs[0]}</text>
  <text x="{width - padding}" y="{height - padding + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{epochs[-1]}</text>
</svg>
"""

    output_path.write_text(svg, encoding="utf-8")


def _text_color_for_cell(value: int, max_value: int) -> str:
    """Choose a readable text color for a confusion-matrix cell.

    Args:
        value: Cell count.
        max_value: Maximum cell count in the matrix.

    Returns:
        Hex color string.
    """
    if max_value == 0:
        return "#111111"

    return "white" if value / max_value > 0.55 else "#111111"


def _cell_fill(value: int, max_value: int) -> str:
    """Create a blue color for a confusion-matrix cell.

    Args:
        value: Cell count.
        max_value: Maximum cell count in the matrix.

    Returns:
        Hex color string.
    """
    if max_value == 0:
        intensity = 0
    else:
        intensity = int(210 * value / max_value)

    red = 235 - int(150 * intensity / 210)
    green = 245 - int(100 * intensity / 210)
    blue = 255

    return f"#{red:02x}{green:02x}{blue:02x}"


def plot_confusion_matrix_svg(
    metrics_path: str | Path,
    output_path: str | Path,
    width: int = 520,
    height: int = 460,
) -> None:
    """Create an SVG confusion-matrix plot from evaluation metrics.

    Args:
        metrics_path: Path to an evaluation metrics JSON file.
        output_path: Destination SVG file.
        width: SVG width in pixels.
        height: SVG height in pixels.

    Raises:
        FileNotFoundError: If ``metrics_path`` does not exist.
        ValueError: If required confusion-matrix keys are missing.
    """
    metrics_path = Path(metrics_path)
    output_path = Path(output_path)

    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics file does not exist: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    required_keys = {
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    }
    missing_keys = required_keys - set(metrics)

    if missing_keys:
        raise ValueError(f"Metrics file is missing keys: {sorted(missing_keys)}")

    true_negatives = int(metrics["true_negatives"])
    false_positives = int(metrics["false_positives"])
    false_negatives = int(metrics["false_negatives"])
    true_positives = int(metrics["true_positives"])
    matrix = [
        [true_negatives, false_positives],
        [false_negatives, true_positives],
    ]
    max_value = max(true_negatives, false_positives, false_negatives, true_positives)
    cell_size = 140
    left = 150
    top = 105
    labels = [["TN", "FP"], ["FN", "TP"]]
    actual_labels = ["Actual real", "Actual fake"]
    predicted_labels = ["Pred real", "Pred fake"]
    cells = []

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            x = left + col_index * cell_size
            y = top + row_index * cell_size
            fill = _cell_fill(value, max_value)
            text_color = _text_color_for_cell(value, max_value)
            cells.append(
                f"""
  <rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>
  <text x="{x + cell_size / 2:.0f}" y="{y + 58}" text-anchor="middle" font-family="sans-serif" font-size="20" fill="{text_color}">{labels[row_index][col_index]}</text>
  <text x="{x + cell_size / 2:.0f}" y="{y + 92}" text-anchor="middle" font-family="sans-serif" font-size="32" font-weight="bold" fill="{text_color}">{value}</text>"""
            )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2:.0f}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20">Confusion Matrix</text>
  <text x="{left + cell_size:.0f}" y="68" text-anchor="middle" font-family="sans-serif" font-size="14">Predicted label</text>
  <text x="35" y="{top + cell_size:.0f}" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 35 {top + cell_size:.0f})">Actual label</text>
  <text x="{left + cell_size / 2:.0f}" y="92" text-anchor="middle" font-family="sans-serif" font-size="13">{predicted_labels[0]}</text>
  <text x="{left + cell_size + cell_size / 2:.0f}" y="92" text-anchor="middle" font-family="sans-serif" font-size="13">{predicted_labels[1]}</text>
  <text x="{left - 18}" y="{top + cell_size / 2 + 5:.0f}" text-anchor="end" font-family="sans-serif" font-size="13">{actual_labels[0]}</text>
  <text x="{left - 18}" y="{top + cell_size + cell_size / 2 + 5:.0f}" text-anchor="end" font-family="sans-serif" font-size="13">{actual_labels[1]}</text>
{''.join(cells)}
  <text x="{width / 2:.0f}" y="{height - 34}" text-anchor="middle" font-family="sans-serif" font-size="13">real = 0, fake = 1</text>
</svg>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
