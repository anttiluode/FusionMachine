"""Exact v0 construction: two resident computations and one contextual boundary."""

from __future__ import annotations

from itertools import product


def algorithm_a(x0: int, x1: int, x2: int) -> int:
    """Direct computation."""
    return int(x0)


def algorithm_b(x0: int, x1: int, x2: int) -> int:
    """Relational computation."""
    return int(x1 * x2)


def nonlinear_boundary(a: float, b: float, context: int) -> float:
    """Select A for context=-1 and B for context=+1 via one product term."""
    return 0.5 * (a + b) + 0.5 * context * (b - a)


def correlated_rows() -> list[dict[str, int]]:
    """Ordinary world where direct and relational computations are indistinguishable."""
    rows: list[dict[str, int]] = []
    for x1, x2 in product((-1, 1), repeat=2):
        x0 = x1 * x2
        a = algorithm_a(x0, x1, x2)
        b = algorithm_b(x0, x1, x2)
        rows.append({"x0": x0, "x1": x1, "x2": x2, "a": a, "b": b})
    return rows


def intervention_rows() -> list[dict[str, int]]:
    """Complete counterfactual world with independent cues and selection context."""
    rows: list[dict[str, int]] = []
    for x0, x1, x2, context in product((-1, 1), repeat=4):
        a = algorithm_a(x0, x1, x2)
        b = algorithm_b(x0, x1, x2)
        target = a if context == -1 else b
        rows.append(
            {
                "x0": x0,
                "x1": x1,
                "x2": x2,
                "a": a,
                "b": b,
                "context": context,
                "target": target,
            }
        )
    return rows
