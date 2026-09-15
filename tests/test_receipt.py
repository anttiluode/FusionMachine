import json
from pathlib import Path

from experiments.run_v0 import build_receipt


def test_v0_receipt_has_the_exact_expected_scientific_spine():
    receipt = build_receipt()
    assert receipt["correlated"]["agreement_fraction"] == 1.0
    assert receipt["intervention"]["algorithm_disagreement_fraction"] == 0.5
    assert receipt["nonlinear_boundary"]["mse"] == 0.0
    assert receipt["nonlinear_boundary"]["accuracy"] == 1.0
    assert abs(receipt["linear_attacker"]["mse"] - 0.5) < 1e-12
    assert receipt["linear_attacker"]["accuracy"] == 0.75
    assert abs(receipt["collapsed_attacker"]["mse"] - 0.5) < 1e-12
    assert receipt["collapsed_attacker"]["accuracy"] == 0.75
    assert receipt["two_row_calibration"]["calibration_rows"] == 2
    assert receipt["two_row_calibration"]["weights_sum_gate"] == [0.5, 0.5]
    assert receipt["two_row_calibration"]["mse"] == 0.0
    assert receipt["two_row_calibration"]["accuracy"] == 1.0


def test_checked_in_receipt_matches_generated_receipt():
    frozen = json.loads(Path("results/v0.json").read_text())
    assert frozen == build_receipt()
