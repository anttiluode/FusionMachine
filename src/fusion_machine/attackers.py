"""Strong v0 attackers for linear fusion and information-collapsed state."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _class_label(value: float) -> int:
    return 1 if value >= 0.0 else -1


def fit_linear_boundary(rows: list[dict[str, int]]) -> dict[str, object]:
    """Fit the best affine readout of separated A/B/context without products."""
    x = np.asarray([[1.0, r["a"], r["b"], r["context"]] for r in rows], dtype=float)
    y = np.asarray([r["target"] for r in rows], dtype=float)
    weights, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ weights
    mse = float(np.mean((pred - y) ** 2))
    accuracy = float(np.mean([_class_label(p) == int(t) for p, t in zip(pred, y)]))
    return {
        "weights": [float(v) for v in weights],
        "predictions": [float(v) for v in pred],
        "mse": mse,
        "accuracy": accuracy,
    }


def best_collapsed_lookup(rows: list[dict[str, int]]) -> dict[str, object]:
    """Best minimum-MSE deterministic predictor that sees only collapsed state r and context."""
    groups: dict[tuple[float, int], list[int]] = defaultdict(list)
    for row in rows:
        key = (0.5 * (row["a"] + row["b"]), row["context"])
        groups[key].append(row["target"])

    table = {key: float(np.mean(targets)) for key, targets in groups.items()}
    predictions = [table[(0.5 * (r["a"] + r["b"]), r["context"])] for r in rows]
    targets = [r["target"] for r in rows]
    mse = float(np.mean([(p - t) ** 2 for p, t in zip(predictions, targets)]))
    accuracy = float(np.mean([_class_label(p) == t for p, t in zip(predictions, targets)]))
    return {
        "table": {f"{key[0]:g},{key[1]}": value for key, value in sorted(table.items())},
        "predictions": predictions,
        "mse": mse,
        "accuracy": accuracy,
    }


def collapse_witness(rows: list[dict[str, int]]) -> tuple[dict[str, int], dict[str, int]]:
    """Return two states that collapse to the same observable but need opposite outputs."""
    seen: dict[tuple[float, int], dict[str, int]] = {}
    for row in rows:
        key = (0.5 * (row["a"] + row["b"]), row["context"])
        previous = seen.get(key)
        if previous is not None and previous["target"] != row["target"]:
            return previous, row
        seen[key] = row
    raise ValueError("no collapse witness exists")
