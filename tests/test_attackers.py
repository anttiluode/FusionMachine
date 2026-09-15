from fusion_machine.attackers import (
    best_collapsed_lookup,
    collapse_witness,
    fit_factorized_boundary,
    fit_linear_boundary,
)
from fusion_machine.core import intervention_rows


def test_best_linear_boundary_cannot_implement_contextual_selection_exactly():
    result = fit_linear_boundary(intervention_rows())
    assert result["mse"] > 0.0
    assert result["accuracy"] < 1.0


def test_collapsed_representation_has_irreducible_error():
    result = best_collapsed_lookup(intervention_rows())
    assert result["mse"] > 0.0
    assert result["accuracy"] < 1.0


def test_collapse_witness_has_same_collapsed_state_and_context_but_opposite_targets():
    left, right = collapse_witness(intervention_rows())
    left_r = 0.5 * (left["a"] + left["b"])
    right_r = 0.5 * (right["a"] + right["b"])
    assert left_r == right_r
    assert left["context"] == right["context"]
    assert left["target"] == -right["target"]


def test_two_calibration_rows_identify_the_factorized_boundary():
    rows = intervention_rows()
    agree = next(r for r in rows if r["a"] == 1 and r["b"] == 1)
    disagree = next(r for r in rows if r["a"] == -1 and r["b"] == 1 and r["context"] == -1)
    result = fit_factorized_boundary([agree, disagree], rows)
    assert result["weights_sum_gate"] == [0.5, 0.5]
    assert result["mse"] == 0.0
    assert result["accuracy"] == 1.0
