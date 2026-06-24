"""Plot helpers for experiment metrics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

__all__ = [
    "load_training_history",
    "plot_confusion_matrix_svg",
    "plot_metric_history_svg",
    "plot_roc_comparison_svg",
    "plot_roc_curve_svg",
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


def _value_bounds(values: list[float]) -> tuple[float, float]:
    """Return readable bounds that include all values."""
    min_value = min(values)
    max_value = max(values)

    if math.isclose(min_value, max_value):
        margin = max(abs(min_value) * 0.1, 0.05)
        return min_value - margin, max_value + margin

    margin = (max_value - min_value) * 0.08
    return min_value - margin, max_value + margin


def _scale_values(
    values: list[float],
    height: int,
    padding: int,
    bounds: tuple[float, float] | None = None,
) -> list[float]:
    """Scale values to SVG y coordinates.

    Args:
        values: Metric values to scale.
        height: SVG height in pixels.
        padding: Plot padding in pixels.

    Returns:
        Y coordinates where lower metric values are lower on the plot.
    """
    min_value, max_value = bounds or _value_bounds(values)
    drawable_height = height - 2 * padding

    return [
        height - padding - ((value - min_value) / (max_value - min_value)) * drawable_height
        for value in values
    ]


def _axis_ticks(min_value: float, max_value: float, count: int = 5) -> list[float]:
    """Create evenly spaced axis tick values."""
    return [min_value + index * (max_value - min_value) / (count - 1) for index in range(count)]


def _epoch_tick_indices(epochs: list[int], max_ticks: int = 6) -> list[int]:
    """Select representative epoch indices, including both endpoints."""
    if len(epochs) <= max_ticks:
        return list(range(len(epochs)))

    return sorted(
        {round(index * (len(epochs) - 1) / (max_ticks - 1)) for index in range(max_ticks)}
    )


def _axes_svg(
    epochs: list[int],
    x_values: list[float],
    bounds: tuple[float, float],
    width: int,
    height: int,
    padding: int,
) -> str:
    """Render grid lines and labeled ticks for a metric plot."""
    min_value, max_value = bounds
    y_ticks = _axis_ticks(min_value, max_value)
    parts = []

    for value in y_ticks:
        y = _scale_values([value], height, padding, bounds)[0]
        parts.append(
            f'  <line x1="{padding}" y1="{y:.2f}" x2="{width - padding}" '
            f'y2="{y:.2f}" stroke="#e5e7eb"/>\n'
            f'  <text x="{padding - 9}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{value:.3f}</text>'
        )

    for index in _epoch_tick_indices(epochs):
        x = x_values[index]
        parts.append(
            f'  <line x1="{x:.2f}" y1="{padding}" x2="{x:.2f}" '
            f'y2="{height - padding}" stroke="#f0f0f0"/>\n'
            f'  <text x="{x:.2f}" y="{height - padding + 22}" '
            f'text-anchor="middle" font-family="sans-serif" '
            f'font-size="11">{epochs[index]}</text>'
        )

    return "\n".join(parts)


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

    return [padding + index * drawable_width / (len(epochs) - 1) for index, _ in enumerate(epochs)]


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
            raise ValueError(f"Training metric row is missing keys: {sorted(missing_keys)}")


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
        label: [float(item[key]) for item in history] for label, key in metric_keys.items()
    }
    all_values = [value for values in series_values.values() for value in values]
    bounds = _value_bounds(all_values)
    y_values = _scale_values(all_values, height, padding, bounds)
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
{_axes_svg(epochs, x_values, bounds, width, height, padding)}
  <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#333"/>
  <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#333"/>
  <text x="{width / 2:.0f}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="13">Epoch</text>
  <text x="20" y="{height / 2:.0f}" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 20 {height / 2:.0f})">{y_label}</text>
{"".join(series_svg)}
{chr(10).join(legend_svg)}
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
    bounds = _value_bounds(loss_values + accuracy_values)
    loss_y_values = _scale_values(loss_values, height, padding, bounds)
    accuracy_y_values = _scale_values(accuracy_values, height, padding, bounds)
    loss_points = list(zip(x_values, loss_y_values, strict=True))
    accuracy_points = list(zip(x_values, accuracy_y_values, strict=True))

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2:.0f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">Training History</text>
{_axes_svg(epochs, x_values, bounds, width, height, padding)}
  <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#333"/>
  <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#333"/>
  <text x="{width / 2:.0f}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="13">Epoch</text>
  <text x="20" y="{height / 2:.0f}" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 20 {height / 2:.0f})">Value</text>
  <polyline points="{_polyline(loss_points)}" fill="none" stroke="#d62728" stroke-width="3"/>
  <polyline points="{_polyline(accuracy_points)}" fill="none" stroke="#1f77b4" stroke-width="3"/>
  <circle cx="{loss_points[-1][0]:.2f}" cy="{loss_points[-1][1]:.2f}" r="4" fill="#d62728"/>
  <circle cx="{accuracy_points[-1][0]:.2f}" cy="{accuracy_points[-1][1]:.2f}" r="4" fill="#1f77b4"/>
  <text x="{width - padding}" y="{padding - 20}" text-anchor="end" font-family="sans-serif" font-size="13" fill="#d62728">loss {loss_values[-1]:.4f}</text>
  <text x="{width - padding}" y="{padding}" text-anchor="end" font-family="sans-serif" font-size="13" fill="#1f77b4">accuracy {accuracy_values[-1]:.4f}</text>
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
{"".join(cells)}
  <text x="{width / 2:.0f}" y="{height - 34}" text-anchor="middle" font-family="sans-serif" font-size="13">real = 0, fake = 1</text>
</svg>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def plot_roc_curve_svg(
    predictions_path: str | Path,
    output_path: str | Path,
    title: str = "ROC Curve",
    width: int = 560,
    height: int = 500,
) -> None:
    """Create an SVG ROC curve from prediction probabilities."""
    predictions_path = Path(predictions_path)
    output_path = Path(output_path)
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Predictions file does not exist: {predictions_path}")

    labels: list[int] = []
    scores: list[float] = []
    with predictions_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"label_idx", "prob_fake"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Predictions file is missing columns: {sorted(missing_columns)}")
        for row in reader:
            labels.append(int(row["label_idx"]))
            scores.append(float(row["prob_fake"]))

    num_positive = sum(label == 1 for label in labels)
    num_negative = sum(label == 0 for label in labels)
    if num_positive == 0 or num_negative == 0:
        raise ValueError("ROC curve requires both classes")

    ranked = sorted(zip(scores, labels, strict=True), reverse=True)
    points = [(0.0, 0.0)]
    true_positives = 0
    false_positives = 0
    index = 0
    while index < len(ranked):
        score = ranked[index][0]
        while index < len(ranked) and ranked[index][0] == score:
            if ranked[index][1] == 1:
                true_positives += 1
            else:
                false_positives += 1
            index += 1
        points.append(
            (
                false_positives / num_negative,
                true_positives / num_positive,
            )
        )

    auc = sum(
        (right[0] - left[0]) * (right[1] + left[1]) / 2
        for left, right in zip(points, points[1:], strict=False)
    )
    padding = 70
    drawable_width = width - 2 * padding
    drawable_height = height - 2 * padding
    svg_points = [
        (
            padding + false_positive_rate * drawable_width,
            height - padding - true_positive_rate * drawable_height,
        )
        for false_positive_rate, true_positive_rate in points
    ]
    grid = []
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = padding + tick * drawable_width
        y = height - padding - tick * drawable_height
        grid.append(
            f'  <line x1="{x:.2f}" y1="{padding}" x2="{x:.2f}" '
            f'y2="{height - padding}" stroke="#eeeeee"/>\n'
            f'  <line x1="{padding}" y1="{y:.2f}" x2="{width - padding}" '
            f'y2="{y:.2f}" stroke="#eeeeee"/>\n'
            f'  <text x="{x:.2f}" y="{height - padding + 22}" '
            f'text-anchor="middle" font-family="sans-serif" '
            f'font-size="11">{tick:.2f}</text>\n'
            f'  <text x="{padding - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{tick:.2f}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2:.0f}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>
{chr(10).join(grid)}
  <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{padding}" stroke="#999999" stroke-width="2" stroke-dasharray="7 6"/>
  <polyline points="{_polyline(svg_points)}" fill="none" stroke="#1f77b4" stroke-width="3"/>
  <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#333333"/>
  <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#333333"/>
  <text x="{width / 2:.0f}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="13">False positive rate</text>
  <text x="20" y="{height / 2:.0f}" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 20 {height / 2:.0f})">True positive rate</text>
  <text x="{width - padding}" y="{padding - 18}" text-anchor="end" font-family="sans-serif" font-size="13" fill="#1f77b4">AUC = {auc:.4f}</text>
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def _load_roc_points(
    predictions_path: str | Path,
    score_column: str,
) -> tuple[list[tuple[float, float]], float]:
    """Load binary labels and return ROC points with trapezoidal AUC."""
    predictions_path = Path(predictions_path)
    with predictions_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"label_idx", score_column}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Predictions file is missing columns: {sorted(missing_columns)}")
        rows = [(float(row[score_column]), int(row["label_idx"])) for row in reader]

    num_positive = sum(label == 1 for _, label in rows)
    num_negative = sum(label == 0 for _, label in rows)
    if num_positive == 0 or num_negative == 0:
        raise ValueError("ROC curve requires both classes")

    ranked = sorted(rows, reverse=True)
    points = [(0.0, 0.0)]
    true_positives = 0
    false_positives = 0
    index = 0
    while index < len(ranked):
        score = ranked[index][0]
        while index < len(ranked) and ranked[index][0] == score:
            if ranked[index][1] == 1:
                true_positives += 1
            else:
                false_positives += 1
            index += 1
        points.append(
            (
                false_positives / num_negative,
                true_positives / num_positive,
            )
        )

    auc = sum(
        (right[0] - left[0]) * (right[1] + left[1]) / 2
        for left, right in zip(points, points[1:], strict=False)
    )
    return points, auc


def plot_roc_comparison_svg(
    series: dict[str, tuple[str | Path, str]],
    output_path: str | Path,
    title: str = "ROC Curve Comparison",
    width: int = 620,
    height: int = 520,
) -> None:
    """Plot multiple ROC series from prediction CSV files."""
    if not series:
        raise ValueError("series must contain at least one ROC curve")

    padding = 70
    drawable_width = width - 2 * padding
    drawable_height = height - 2 * padding
    curves = []
    for series_index, (label, (path, score_column)) in enumerate(series.items()):
        points, auc = _load_roc_points(path, score_column)
        svg_points = [
            (
                padding + false_positive_rate * drawable_width,
                height - padding - true_positive_rate * drawable_height,
            )
            for false_positive_rate, true_positive_rate in points
        ]
        curves.append(
            (
                label,
                auc,
                SERIES_COLORS[series_index % len(SERIES_COLORS)],
                svg_points,
            )
        )

    grid = []
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = padding + tick * drawable_width
        y = height - padding - tick * drawable_height
        grid.append(
            f'  <line x1="{x:.2f}" y1="{padding}" x2="{x:.2f}" '
            f'y2="{height - padding}" stroke="#eeeeee"/>\n'
            f'  <line x1="{padding}" y1="{y:.2f}" x2="{width - padding}" '
            f'y2="{y:.2f}" stroke="#eeeeee"/>\n'
            f'  <text x="{x:.2f}" y="{height - padding + 22}" '
            f'text-anchor="middle" font-family="sans-serif" '
            f'font-size="11">{tick:.2f}</text>\n'
            f'  <text x="{padding - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{tick:.2f}</text>'
        )

    curve_svg = []
    legend_svg = []
    for index, (label, auc, color, points) in enumerate(curves):
        curve_svg.append(
            f'  <polyline points="{_polyline(points)}" fill="none" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        legend_svg.append(
            f'  <text x="{width - padding}" y="{padding + 20 + index * 20}" '
            f'text-anchor="end" font-family="sans-serif" font-size="13" '
            f'fill="{color}">{label}: AUC = {auc:.4f}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2:.0f}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>
{chr(10).join(grid)}
  <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{padding}" stroke="#999999" stroke-width="2" stroke-dasharray="7 6"/>
{chr(10).join(curve_svg)}
  <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#333333"/>
  <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#333333"/>
  <text x="{width / 2:.0f}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="13">False positive rate</text>
  <text x="20" y="{height / 2:.0f}" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 20 {height / 2:.0f})">True positive rate</text>
{chr(10).join(legend_svg)}
</svg>
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
