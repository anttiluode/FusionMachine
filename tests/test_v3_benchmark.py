import numpy as np

from fusion_machine.benchmark_v3 import (
    fit_latent_probe,
    generate_long_dormancy,
    generate_rapid_switch,
    generate_training_batch,
    publication_slots,
    teacher_rollout,
)


def test_training_batch_matches_frozen_resource_contract():
    train = generate_training_batch()
    assert train.x.shape == (96, 64, 4)
    assert train.teacher_state.shape == (96, 64, 4)
    assert train.target.shape == (96, 64)
    assert train.query_mask.shape == (96, 64)
    assert train.query_mask.sum(axis=1).min() >= 2
    assert set(np.unique(train.context)) == {0, 1}
    assert np.array_equal(train.x[..., 3].astype(int), train.context)
    assert np.array_equal(train.route_target, (train.source_id[:, None] + 2 * train.context) % 4)


def test_teacher_first_update_matches_frozen_matrices():
    x = np.array([[[1.0, -1.0, -1.0, 0.0]]])
    context = np.array([[0]], dtype=int)
    states, target = teacher_rollout(x, context)
    expected = np.tanh(np.array([0.15, -0.85, -0.15, 0.15]))
    assert np.allclose(states[0, 0], expected)
    assert np.isclose(target[0, 0], 0.80 * expected[0] - 0.35 * expected[1])


def test_rapid_switch_and_long_dormancy_are_frozen_regimes():
    rapid = generate_rapid_switch()
    assert rapid.x.shape == (256, 64, 4)
    for c in rapid.context:
        starts = np.r_[0, 1 + np.flatnonzero(c[1:] != c[:-1])]
        ends = np.r_[starts[1:], len(c)]
        lengths = ends - starts
        assert np.all((lengths[:-1] >= 2) & (lengths[:-1] <= 6))
        assert 1 <= lengths[-1] <= 6

    long_gap = generate_long_dormancy()
    assert long_gap.x.shape == (256, 96, 4)
    assert np.all(long_gap.context[:, :48] == long_gap.context[:, [0]])
    assert np.all(long_gap.context[:, 48:] == 1 - long_gap.context[:, [0]])
    assert np.all(long_gap.switch_mask[:, 48])
    assert np.sum(long_gap.switch_mask) == 256


def test_publication_slots_are_exactly_shared_sixteen_opportunities():
    slots = publication_slots(64)
    assert np.flatnonzero(slots).tolist() == list(range(0, 64, 4))
    assert int(slots.sum()) == 16


def test_latent_probe_is_one_for_exact_linear_copy_and_bad_for_constant_state():
    rng = np.random.default_rng(9)
    teacher_train = rng.normal(size=(100, 4))
    teacher_test = rng.normal(size=(50, 4))
    hidden_train = teacher_train @ np.diag([2.0, -1.0, 0.5, 3.0])
    hidden_test = teacher_test @ np.diag([2.0, -1.0, 0.5, 3.0])
    exact = fit_latent_probe(hidden_train, teacher_train, hidden_test, teacher_test)
    assert exact["r2"] > 0.999999

    constant = fit_latent_probe(
        np.zeros_like(hidden_train), teacher_train, np.zeros_like(hidden_test), teacher_test
    )
    assert constant["r2"] < 0.05
