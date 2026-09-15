"""FusionMachine v1: resident algorithmic state versus bounded replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fusion_machine.resident import analytic_replay_rmse

ALPHAS = (0.50, 0.80, 0.95, 0.98)
REPLAY_WINDOWS = (0, 4, 8, 16, 32, 64, 80)
GAP = 80
RNG_SEED = 20260915


def _clean(value: float) -> float:
    value = float(value)
    if abs(value) < 5e-13:
        return 0.0
    return float(round(value, 12))


def _measured_rmse(drives: np.ndarray, alpha: float, k: int) -> float:
    """Monte Carlo switch error from the exact omitted-history contribution."""
    if k >= drives.shape[1]:
        return 0.0
    gap = drives.shape[1]
    omitted = gap - k
    exponents = np.arange(gap - 1, k - 1, -1, dtype=float)
    weights = (1.0 - alpha) * np.power(alpha, exponents)
    errors = drives[:, :omitted] @ weights
    return float(np.sqrt(np.mean(errors**2)))


def _horizon(alpha: float, fraction: float = 0.1) -> int:
    baseline = analytic_replay_rmse(alpha, GAP, 0)
    if baseline == 0.0:
        return 0
    for k in range(GAP + 1):
        if analytic_replay_rmse(alpha, GAP, k) <= fraction * baseline:
            return k
    return GAP


def build_receipt(seeds: int = 4096) -> dict[str, object]:
    rng = np.random.default_rng(RNG_SEED)
    drives = rng.choice(np.array([-1.0, 1.0]), size=(seeds, GAP))

    sweep: dict[str, list[dict[str, float | int]]] = {}
    max_relative_discrepancy = 0.0
    max_location: dict[str, float | int] | None = None

    for alpha in ALPHAS:
        curve: list[dict[str, float | int]] = []
        for k in REPLAY_WINDOWS:
            measured = _measured_rmse(drives, alpha, k)
            analytic = analytic_replay_rmse(alpha, GAP, k)
            if analytic > 0.0:
                relative_discrepancy = abs(measured - analytic) / analytic
                if relative_discrepancy > max_relative_discrepancy:
                    max_relative_discrepancy = relative_discrepancy
                    max_location = {"alpha": alpha, "k": k}
            else:
                relative_discrepancy = 0.0

            curve.append(
                {
                    "k": k,
                    "measured_rmse": _clean(measured),
                    "analytic_rmse": _clean(analytic),
                    "relative_discrepancy": _clean(relative_discrepancy),
                }
            )
        sweep[f"{alpha:.2f}"] = curve

    return {
        "version": "v1",
        "seeds": seeds,
        "rng_seed": RNG_SEED,
        "gap": GAP,
        "alphas": [*ALPHAS],
        "replay_windows": [*REPLAY_WINDOWS],
        "primary_alpha_0.95": sweep["0.95"],
        "sweep": sweep,
        "analytic_audit": {
            "max_relative_discrepancy": _clean(max_relative_discrepancy),
            "location": max_location,
        },
        "replay_horizon_for_10pct_of_zero_history_rmse": {
            f"{alpha:.2f}": _horizon(alpha, 0.1) for alpha in ALPHAS
        },
        "work_accounting_gap_80": {
            "resident_dual": {
                "dormant_updates_during_gap": GAP,
                "stored_dormant_drives": 0,
                "switch_replay_updates": 0,
                "resident_state_scalars": 1,
            },
            "lazy_full_replay": {
                "dormant_updates_during_gap": 0,
                "stored_dormant_drives": GAP,
                "switch_replay_updates": GAP,
                "resident_state_scalars": 1,
            },
        },
        "interpretation": (
            "resident state trades continuous local updates for immediate readiness; "
            "bounded replay trades readiness for retained history and switch-time reconstruction"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/v1.json")
    parser.add_argument("--seeds", type=int, default=4096)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_receipt(args.seeds), indent=2, sort_keys=True) + "\n")
    print(out)


if __name__ == "__main__":
    main()
