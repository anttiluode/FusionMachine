"""Frozen v0 scientific receipt for FusionMachine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fusion_machine.attackers import (
    best_collapsed_lookup,
    collapse_witness,
    fit_factorized_boundary,
    fit_linear_boundary,
)
from fusion_machine.core import correlated_rows, intervention_rows, nonlinear_boundary


def _clean(value: float) -> float:
    value = float(value)
    if abs(value) < 1e-12:
        return 0.0
    return float(round(value, 12))


def _accuracy(predictions: list[float], targets: list[int]) -> float:
    labels = [1 if p >= 0.0 else -1 for p in predictions]
    return float(np.mean([p == t for p, t in zip(labels, targets)]))


def build_receipt() -> dict[str, object]:
    correlated = correlated_rows()
    rows = intervention_rows()
    targets = [r["target"] for r in rows]

    nonlinear_predictions = [
        nonlinear_boundary(r["a"], r["b"], r["context"]) for r in rows
    ]
    nonlinear_mse = float(
        np.mean([(p - t) ** 2 for p, t in zip(nonlinear_predictions, targets)])
    )

    linear = fit_linear_boundary(rows)
    collapsed = best_collapsed_lookup(rows)
    agree = next(r for r in rows if r["a"] == 1 and r["b"] == 1)
    disagree = next(r for r in rows if r["a"] == -1 and r["b"] == 1 and r["context"] == -1)
    calibrated = fit_factorized_boundary([agree, disagree], rows)
    witness_left, witness_right = collapse_witness(rows)

    return {
        "version": "v0",
        "correlated": {
            "rows": len(correlated),
            "agreement_fraction": float(np.mean([r["a"] == r["b"] for r in correlated])),
        },
        "intervention": {
            "rows": len(rows),
            "algorithm_disagreement_fraction": float(np.mean([r["a"] != r["b"] for r in rows])),
        },
        "nonlinear_boundary": {
            "mse": _clean(nonlinear_mse),
            "accuracy": _accuracy(nonlinear_predictions, targets),
        },
        "linear_attacker": {
            "mse": _clean(float(linear["mse"])),
            "accuracy": float(linear["accuracy"]),
            "weights_bias_a_b_context": [_clean(v) for v in linear["weights"]],
        },
        "collapsed_attacker": {
            "mse": _clean(float(collapsed["mse"])),
            "accuracy": float(collapsed["accuracy"]),
        },
        "two_row_calibration": {
            "calibration_rows": 2,
            "weights_sum_gate": [_clean(v) for v in calibrated["weights_sum_gate"]],
            "mse": _clean(float(calibrated["mse"])),
            "accuracy": float(calibrated["accuracy"]),
        },
        "collapse_witness": {
            "left": witness_left,
            "right": witness_right,
            "collapsed_state": _clean(0.5 * (witness_left["a"] + witness_left["b"])),
            "context": witness_left["context"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/v0.json")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_receipt(), indent=2, sort_keys=True) + "\n")
    print(out)


if __name__ == "__main__":
    main()
