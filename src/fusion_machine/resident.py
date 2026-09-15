"""Persistent algorithmic state and bounded-history reconstruction for v1."""

from __future__ import annotations

import math
from collections.abc import Sequence


def step(state: float, drive: float, alpha: float) -> float:
    """One normalized leaky-state update."""
    return alpha * state + (1.0 - alpha) * drive


def run_state(
    drives: Sequence[float],
    alpha: float,
    initial: float = 0.0,
) -> float:
    """Run the recurrence across a sequence of drives."""
    state = float(initial)
    for drive in drives:
        state = step(state, float(drive), alpha)
    return state


def reconstruct_from_tail(
    boundary_state: float,
    hidden_drives: Sequence[float],
    alpha: float,
    k: int,
) -> float:
    """Reconstruct after a hidden gap using only its last k drives.

    The known boundary state is decayed across the full gap. Remembered tail
    drives are then added with their exact recurrence weights. Unremembered
    earlier drives are treated as zero.
    """
    gap = len(hidden_drives)
    if not 0 <= k <= gap:
        raise ValueError("k must satisfy 0 <= k <= len(hidden_drives)")

    state = (alpha**gap) * float(boundary_state)
    start = gap - k
    for index in range(start, gap):
        exponent = gap - 1 - index
        state += (1.0 - alpha) * (alpha**exponent) * float(hidden_drives[index])
    return state


def analytic_replay_rmse(alpha: float, gap: int, k: int) -> float:
    """Expected switch-time RMSE from omitted iid Rademacher drives."""
    if gap < 0:
        raise ValueError("gap must be non-negative")
    if not 0 <= k <= gap:
        raise ValueError("k must satisfy 0 <= k <= gap")
    if k == gap or gap == 0:
        return 0.0
    if alpha == 0.0:
        return 0.0 if k >= 1 else 1.0

    variance = (
        (1.0 - alpha) ** 2
        * (alpha ** (2 * k))
        * (1.0 - alpha ** (2 * (gap - k)))
        / (1.0 - alpha**2)
    )
    return math.sqrt(variance)
