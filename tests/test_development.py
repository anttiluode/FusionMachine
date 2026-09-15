import numpy as np

from fusion_machine.development import (
    activity_alignment,
    chemical_scores,
    graph_accuracy,
    grow_routes,
    make_problem,
    routing_accuracy,
)


def test_problem_is_deterministic_and_has_permutation_truth():
    a = make_problem(7)
    b = make_problem(7)
    assert np.array_equal(a["truth"], b["truth"])
    assert np.allclose(a["source_fp"], b["source_fp"])
    assert np.allclose(a["target_fp"], b["target_fp"])
    assert np.allclose(a["source_activity"], b["source_activity"])
    assert np.allclose(a["target_activity"], b["target_activity"])
    assert sorted(a["truth"].tolist()) == list(range(len(a["truth"])))


def test_grow_routes_selects_exactly_one_target_per_source():
    problem = make_problem(3)
    chem = chemical_scores(problem["source_fp"], problem["target_fp"])
    act = activity_alignment(problem["source_activity"], problem["target_activity"])
    routes = grow_routes(chem, act, beta=1.0, top_k=1)
    assert routes.shape == chem.shape
    assert np.all(routes.sum(axis=1) == 1)


def test_combined_development_beats_single_cues_on_frozen_seed_bank():
    combined = []
    chem_only = []
    activity_only = []
    shuffled = []
    for seed in range(32):
        problem = make_problem(seed)
        chem = chemical_scores(problem["source_fp"], problem["target_fp"])
        act = activity_alignment(problem["source_activity"], problem["target_activity"])
        truth = problem["truth"]

        combined_routes = grow_routes(chem, act, beta=1.0)
        chem_routes = grow_routes(chem, np.zeros_like(act), beta=0.0)
        activity_routes = grow_routes(np.zeros_like(chem), act, beta=1.0)
        shuffled_act = act[np.random.default_rng(seed + 1000).permutation(act.shape[0])]
        shuffled_routes = grow_routes(chem, shuffled_act, beta=1.0)

        combined.append(graph_accuracy(combined_routes, truth))
        chem_only.append(graph_accuracy(chem_routes, truth))
        activity_only.append(graph_accuracy(activity_routes, truth))
        shuffled.append(routing_accuracy(shuffled_routes, truth))

    assert float(np.mean(combined)) >= float(np.mean(chem_only)) + 0.05
    assert float(np.mean(combined)) >= float(np.mean(activity_only)) + 0.05
    assert float(np.mean(combined)) >= float(np.mean(shuffled)) + 0.05
