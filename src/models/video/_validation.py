"""Shared validation helpers for video model configuration."""

from collections.abc import Sequence


def validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")


def validate_positive_sequence(values: Sequence[int], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must contain at least one value")

    for value in values:
        validate_positive_int(value, name)
