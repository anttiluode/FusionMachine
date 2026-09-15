from fusion_machine.core import (
    algorithm_a,
    algorithm_b,
    correlated_rows,
    intervention_rows,
    nonlinear_boundary,
)


def test_correlated_world_makes_algorithms_behaviorally_identical():
    rows = correlated_rows()
    assert len(rows) == 4
    assert all(row["a"] == row["b"] for row in rows)


def test_intervention_world_is_complete_and_algorithms_disagree_half_the_time():
    rows = intervention_rows()
    assert len(rows) == 16
    triples = {(r["x0"], r["x1"], r["x2"]) for r in rows}
    assert len(triples) == 8
    disagreeing_triples = {
        (r["x0"], r["x1"], r["x2"])
        for r in rows
        if r["a"] != r["b"]
    }
    assert len(disagreeing_triples) == 4


def test_direct_and_relational_algorithms_have_expected_definitions():
    assert algorithm_a(-1, 1, -1) == -1
    assert algorithm_a(1, -1, -1) == 1
    assert algorithm_b(1, 1, -1) == -1
    assert algorithm_b(-1, -1, -1) == 1


def test_nonlinear_boundary_selects_algorithm_by_context_exactly():
    for row in intervention_rows():
        selected = nonlinear_boundary(row["a"], row["b"], row["context"])
        assert selected == row["target"]
