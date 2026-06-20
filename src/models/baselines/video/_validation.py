"""Shared validation helpers for video model configuration."""

from collections.abc import Sequence


def validate_positive_int(value: int, name: str) -> None:
    """Require an integer-like model parameter to be strictly positive.

    Args:
        value: Parameter value to validate.
        name: Parameter name included in validation errors.

    Raises:
        ValueError: If ``value`` is not greater than zero.
    """
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")


def validate_positive_sequence(values: Sequence[int], name: str) -> None:
    """Require a non-empty sequence containing only positive values.

    Args:
        values: Sequence of channel or layer sizes.
        name: Parameter name included in validation errors.

    Raises:
        ValueError: If the sequence is empty or contains a non-positive value.
    """
    if not values:
        raise ValueError(f"{name} must contain at least one value")

    for value in values:
        validate_positive_int(value, name)
