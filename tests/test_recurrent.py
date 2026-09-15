import numpy as np

from fusion_machine.recurrent import (
    effective_parameter_count,
    init_params,
    loss_and_grads,
    make_fusion_masks,
    make_generic_masks,
)


def _tiny_batch():
    x = np.array(
        [[[1.0, -1.0, -1.0, 0.0], [-1.0, -1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
        dtype=float,
    )
    context = np.array([[0, 1, 1]], dtype=int)
    target = np.array([[0.2, -0.3, 0.1]], dtype=float)
    query = np.array([[True, True, True]])
    source_id = np.array([2], dtype=int)
    route_target = np.array([[2, 0, 0]], dtype=int)
    return x, context, target, query, source_id, route_target


def test_resource_contracts():
    fusion = make_fusion_masks()
    generic = make_generic_masks()
    assert fusion.hidden_size == generic.hidden_size == 4
    assert fusion.route_parameter_count == generic.route_parameter_count == 28
    assert effective_parameter_count(fusion, include_route=False) == 30
    assert effective_parameter_count(generic, include_route=False) == 46


def test_fusion_masks_block_context_and_cross_mode_paths():
    masks = make_fusion_masks()
    assert np.all(masks.W_hh[:2, 2:] == 0)
    assert np.all(masks.W_hh[2:, :2] == 0)
    assert np.all(masks.W_x[:, 3] == 0)
    assert np.all(masks.W_out[0, 2:] == 0)
    assert np.all(masks.W_out[1, :2] == 0)


def test_generic_masks_leave_all_recurrent_and_context_paths_open():
    masks = make_generic_masks()
    assert np.all(masks.W_hh == 1)
    assert np.all(masks.W_x == 1)
    assert np.all(masks.W_out == 1)


def test_bptt_matches_finite_difference_for_unmasked_parameters_and_masks_gradients():
    masks = make_fusion_masks()
    params = init_params(seed=4)
    x, context, target, query, source_id, route_target = _tiny_batch()
    loss, grads, _ = loss_and_grads(
        params,
        masks,
        x=x,
        context=context,
        target=target,
        query_mask=query,
        source_id=source_id,
        route_target=route_target,
        route_loss_weight=0.2,
        weight_decay=0.0,
    )
    assert np.isfinite(loss)

    checks = [
        ("W_hh", (0, 0)),
        ("W_x", (2, 1)),
        ("W_out", (1, 2)),
    ]
    eps = 1e-6
    for name, index in checks:
        plus = params.copy()
        minus = params.copy()
        getattr(plus, name)[index] += eps
        getattr(minus, name)[index] -= eps
        lp, _, _ = loss_and_grads(
            plus,
            masks,
            x=x,
            context=context,
            target=target,
            query_mask=query,
            source_id=source_id,
            route_target=route_target,
            route_loss_weight=0.2,
            weight_decay=0.0,
        )
        lm, _, _ = loss_and_grads(
            minus,
            masks,
            x=x,
            context=context,
            target=target,
            query_mask=query,
            source_id=source_id,
            route_target=route_target,
            route_loss_weight=0.2,
            weight_decay=0.0,
        )
        numeric = (lp - lm) / (2 * eps)
        analytic = getattr(grads, name)[index]
        assert np.isclose(analytic, numeric, rtol=2e-4, atol=2e-5), (name, index, analytic, numeric)

    assert grads.W_hh[0, 3] == 0.0
    assert grads.W_x[1, 3] == 0.0
    assert grads.W_out[0, 3] == 0.0
