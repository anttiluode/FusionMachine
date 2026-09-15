import json
from pathlib import Path

from experiments.run_v1 import build_receipt


def test_primary_replay_curve_is_nonincreasing_and_full_replay_is_exact():
    receipt = build_receipt(4096)
    curve = receipt["primary_alpha_0.95"]
    measured = [point["measured_rmse"] for point in curve]
    assert measured[0] > 0.0
    assert measured[-1] == 0.0
    assert all(left >= right for left, right in zip(measured, measured[1:]))


def test_monte_carlo_curve_matches_closed_form_within_five_percent():
    receipt = build_receipt(4096)
    assert receipt["analytic_audit"]["max_relative_discrepancy"] < 0.05


def test_work_accounting_makes_the_readiness_tradeoff_explicit():
    work = build_receipt(4096)["work_accounting_gap_80"]
    assert work["resident_dual"] == {
        "dormant_updates_during_gap": 80,
        "stored_dormant_drives": 0,
        "switch_replay_updates": 0,
        "resident_state_scalars": 1,
    }
    assert work["lazy_full_replay"] == {
        "dormant_updates_during_gap": 0,
        "stored_dormant_drives": 80,
        "switch_replay_updates": 80,
        "resident_state_scalars": 1,
    }


def test_slow_modes_require_longer_replay_horizons():
    horizons = build_receipt(4096)["replay_horizon_for_10pct_of_zero_history_rmse"]
    assert horizons == {"0.50": 4, "0.80": 11, "0.95": 45, "0.98": 75}


def test_checked_in_v1_receipt_matches_regeneration():
    frozen = json.loads(Path("results/v1.json").read_text())
    assert frozen == build_receipt(4096)
