import pytest

from src.training.early_stopping import (
    EarlyStoppingState,
    build_early_stopping_state,
)


def test_early_stopping_tracks_loss_with_min_delta() -> None:
    state = EarlyStoppingState(
        metric_name="val_loss",
        patience=2,
        min_delta=0.01,
    )

    assert not state.update({"val_loss": 0.50})
    assert not state.update({"val_loss": 0.495})
    assert state.bad_epochs == 1
    assert not state.update({"val_loss": 0.48})
    assert state.bad_epochs == 0


def test_early_stopping_stops_after_patience() -> None:
    state = EarlyStoppingState(
        metric_name="val_accuracy",
        patience=2,
        min_delta=0.01,
    )

    assert not state.update({"val_accuracy": 0.70})
    assert not state.update({"val_accuracy": 0.705})
    assert state.update({"val_accuracy": 0.706})


def test_early_stopping_rejects_missing_metric() -> None:
    state = EarlyStoppingState(metric_name="val_loss", patience=1)

    with pytest.raises(ValueError, match="missing"):
        state.update({"loss": 0.5})


def test_build_early_stopping_state_returns_none_when_disabled() -> None:
    state = build_early_stopping_state(
        {"early_stopping": {"enabled": False}},
        metric_name="val_loss",
    )

    assert state is None


def test_build_early_stopping_state_reads_config() -> None:
    state = build_early_stopping_state(
        {"early_stopping": {"enabled": True, "patience": 3, "min_delta": 0.02}},
        metric_name="val_loss",
    )

    assert state is not None
    assert state.metric_name == "val_loss"
    assert state.patience == 3
    assert state.min_delta == 0.02
