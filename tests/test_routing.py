from fusion_machine.routing import (
    DendriticCompartment,
    Route,
    ais_event,
    route_event,
    run_async_chain,
)


def test_same_bit_reaches_different_targets_by_axon_identity():
    routes = {
        0: [Route(target=1, weight=1.0)],
        1: [Route(target=2, weight=1.0)],
    }
    assert route_event(0, 1, routes) == [(1, 1, 1.0)]
    assert route_event(1, 1, routes) == [(1, 2, 1.0)]


def test_zero_event_does_not_route():
    routes = {0: [Route(target=1, weight=1.0)]}
    assert route_event(0, 0, routes) == []


def test_ais_suppression_blocks_publication_not_state_update():
    d = DendriticCompartment(state=0.0, leak=0.5)
    before = d.step(1.0)
    assert before > 0.0
    assert ais_event(d.state, threshold=0.1, suppressed=True) == 0
    after = d.step(1.0)
    assert after > before
    assert ais_event(d.state, threshold=0.1, suppressed=False) == 1


def test_reroute_changes_target_without_changing_event():
    event = 1
    left = route_event(0, event, {0: [Route(target=1, weight=1.0)]})
    right = route_event(0, event, {0: [Route(target=2, weight=1.0)]})
    assert left != right
    assert left[0][2] == right[0][2] == 1.0


def test_async_chain_uses_route_delays_and_logs_causal_hops():
    routes = {
        0: [Route(target=1, weight=1.0, delay=1)],
        1: [Route(target=2, weight=1.0, delay=2)],
        2: [Route(target=3, weight=1.0, delay=1)],
    }
    result = run_async_chain(
        routes=routes,
        n_compartments=4,
        initial_source=0,
        threshold=0.5,
        max_time=8,
    )
    log = result["events"]
    assert [(e["source"], e["target"]) for e in log] == [(0, 1), (1, 2), (2, 3)]
    assert [(e["emit_time"], e["arrival_time"]) for e in log] == [(0, 1), (1, 3), (3, 4)]
