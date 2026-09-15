import math

from fusion_machine.resident import (
    analytic_replay_rmse,
    reconstruct_from_tail,
    run_state,
    step,
)


def test_step_is_the_leaky_state_update():
    assert abs(step(0.5, -1.0, 0.8) - 0.2) < 1e-15


def test_full_tail_replay_matches_resident_execution_exactly():
    drives = [1.0, -1.0, 1.0, 1.0, -1.0]
    alpha = 0.9
    boundary = 0.37
    resident = run_state(drives, alpha, boundary)
    replayed = reconstruct_from_tail(boundary, drives, alpha, len(drives))
    assert abs(resident - replayed) < 1e-15


def test_zero_history_replay_only_decays_the_known_boundary_state():
    drives = [1.0, -1.0, 1.0, 1.0]
    alpha = 0.8
    boundary = -0.3
    estimate = reconstruct_from_tail(boundary, drives, alpha, 0)
    assert abs(estimate - (alpha ** len(drives)) * boundary) < 1e-15


def test_analytic_replay_rmse_matches_direct_coefficient_sum():
    alpha = 0.95
    gap = 80
    k = 16
    direct_variance = (1 - alpha) ** 2 * sum(alpha ** (2 * r) for r in range(k, gap))
    assert abs(analytic_replay_rmse(alpha, gap, k) - math.sqrt(direct_variance)) < 1e-15


def test_full_replay_has_zero_analytic_error():
    assert analytic_replay_rmse(0.95, 80, 80) == 0.0
