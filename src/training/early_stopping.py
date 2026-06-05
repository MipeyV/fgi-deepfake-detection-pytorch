"""Early stopping helpers for training loops."""

from __future__ import annotations

from dataclasses import dataclass

from src.training.checkpoints import checkpoint_metric_is_better


__all__ = ["EarlyStoppingState", "build_early_stopping_state"]


@dataclass
class EarlyStoppingState:
    """Track metric improvements and decide when training should stop.

    Args:
        metric_name: Metric key used to monitor progress.
        patience: Number of consecutive non-improving epochs allowed.
        min_delta: Minimum absolute change required to count as improvement.
        enabled: Whether early stopping is active.
    """

    metric_name: str
    patience: int
    min_delta: float = 0.0
    enabled: bool = True
    best_value: float | None = None
    bad_epochs: int = 0

    def update(self, metrics: dict) -> bool:
        """Update state from epoch metrics.

        Args:
            metrics: Metrics dictionary for one epoch.

        Returns:
            True when training should stop.

        Raises:
            ValueError: If the monitored metric is missing.
        """
        if not self.enabled:
            return False

        if self.metric_name not in metrics:
            raise ValueError(
                f"Early stopping metric is missing from epoch metrics: "
                f"{self.metric_name}"
            )

        current_value = float(metrics[self.metric_name])
        comparison_value = self._comparison_value()

        if checkpoint_metric_is_better(
            current_value,
            comparison_value,
            self.metric_name,
        ):
            self.best_value = current_value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def _comparison_value(self) -> float | None:
        if self.best_value is None:
            return None

        if "loss" in self.metric_name.lower():
            return self.best_value - self.min_delta

        return self.best_value + self.min_delta


def build_early_stopping_state(
    training_config: dict,
    metric_name: str,
) -> EarlyStoppingState | None:
    """Build early stopping state from a training config section."""
    early_stopping_config = training_config.get("early_stopping", {})

    if not early_stopping_config.get("enabled", False):
        return None

    patience = int(early_stopping_config.get("patience", 5))
    min_delta = float(early_stopping_config.get("min_delta", 0.0))

    if patience <= 0:
        raise ValueError("training.early_stopping.patience must be greater than 0")

    if min_delta < 0:
        raise ValueError("training.early_stopping.min_delta must be non-negative")

    return EarlyStoppingState(
        metric_name=metric_name,
        patience=patience,
        min_delta=min_delta,
        enabled=True,
    )
