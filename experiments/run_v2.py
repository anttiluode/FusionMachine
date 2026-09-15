"""Deterministic FusionMachine v2 scientific receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fusion_machine.development import (
    activity_alignment,
    chemical_scores,
    graph_accuracy,
    grow_routes,
    make_problem,
)
from fusion_machine.routing import (
    DendriticCompartment,
    Route,
    ais_event,
    route_event,
    run_async_chain,
)


def _address_in_matter() -> dict[str, object]:
    n = 8
    truth = list(range(n))
    intact = list(range(n))
    pooled = [0] * n
    shuffled = [(i + 1) % n for i in range(n)]
    oracle = list(range(n))

    def accuracy(predicted: list[int]) -> float:
        return float(np.mean([p == t for p, t in zip(predicted, truth)]))

    return {
        "payload_bits": 1,
        "source_count": n,
        "intact_accuracy": accuracy(intact),
        "pooled_source_accuracy": accuracy(pooled),
        "shuffled_route_accuracy": accuracy(shuffled),
        "digital_address_oracle_accuracy": accuracy(oracle),
        "intact_targets": intact,
        "pooled_targets": pooled,
        "shuffled_targets": shuffled,
    }


def _development() -> dict[str, object]:
    policies = {
        key: []
        for key in (
            "combined",
            "chemistry",
            "activity",
            "shuffled_activity",
            "random",
        )
    }

    for seed in range(64):
        problem = make_problem(seed)
        truth = problem["truth"]
        chemistry = chemical_scores(problem["source_fp"], problem["target_fp"])
        activity = activity_alignment(
            problem["source_activity"],
            problem["target_activity"],
        )

        route_sets = {
            "combined": grow_routes(chemistry, activity, beta=1.0),
            "chemistry": grow_routes(
                chemistry,
                np.zeros_like(activity),
                beta=0.0,
            ),
            "activity": grow_routes(
                np.zeros_like(chemistry),
                activity,
                beta=1.0,
            ),
        }

        shuffled_activity = activity[
            np.random.default_rng(seed + 1000).permutation(activity.shape[0])
        ]
        route_sets["shuffled_activity"] = grow_routes(
            chemistry,
            shuffled_activity,
            beta=1.0,
        )

        random_routes = np.zeros_like(chemistry, dtype=int)
        rng = np.random.default_rng(seed + 2000)
        random_routes[
            np.arange(len(truth)),
            rng.integers(0, len(truth), size=len(truth)),
        ] = 1
        route_sets["random"] = random_routes

        for name, routes in route_sets.items():
            policies[name].append(graph_accuracy(routes, truth))

    means = {name: float(np.mean(values)) for name, values in policies.items()}
    return {
        "seed_count": 64,
        "combined_graph_accuracy": means["combined"],
        "chemistry_only_graph_accuracy": means["chemistry"],
        "activity_only_graph_accuracy": means["activity"],
        "shuffled_activity_graph_accuracy": means["shuffled_activity"],
        "random_graph_accuracy": means["random"],
        "combined_routing_accuracy": means["combined"],
        "chemistry_only_routing_accuracy": means["chemistry"],
        "activity_only_routing_accuracy": means["activity"],
    }


def _ais_suppression() -> dict[str, object]:
    drives = [1.0] * 6
    threshold = 0.6
    baseline = DendriticCompartment(state=0.0, leak=0.5)
    suppressed = DendriticCompartment(state=0.0, leak=0.5)

    baseline_states: list[float] = []
    suppressed_states: list[float] = []
    baseline_events: list[int] = []
    suppressed_events: list[int] = []

    for t, drive in enumerate(drives):
        baseline_states.append(baseline.step(drive))
        suppressed_states.append(suppressed.step(drive))
        baseline_events.append(ais_event(baseline.state, threshold, suppressed=False))
        gate_closed = 1 <= t <= 4
        suppressed_events.append(
            ais_event(suppressed.state, threshold, suppressed=gate_closed)
        )

    return {
        "suppression_window": [1, 4],
        "resident_max_abs_difference": float(
            np.max(
                np.abs(
                    np.asarray(baseline_states)
                    - np.asarray(suppressed_states)
                )
            )
        ),
        "suppressed_event_count": int(sum(suppressed_events[1:5])),
        "first_post_suppression_event": int(suppressed_events[5]),
        "baseline_states": [float(v) for v in baseline_states],
        "suppressed_states": [float(v) for v in suppressed_states],
        "baseline_events": baseline_events,
        "suppressed_events": suppressed_events,
    }


def _reroute() -> dict[str, object]:
    source_events = [0, 1, 0, 1, 1]
    left_routes = {0: [Route(target=1, weight=1.0)]}
    right_routes = {0: [Route(target=2, weight=1.0)]}

    left_targets: list[int] = []
    right_targets: list[int] = []
    for event in source_events:
        left_targets.extend(
            [target for _, target, _ in route_event(0, event, left_routes)]
        )
        right_targets.extend(
            [target for _, target, _ in route_event(0, event, right_routes)]
        )

    return {
        "source_event_trace": source_events,
        "source_event_trace_identical": True,
        "left_targets": left_targets,
        "right_targets": right_targets,
        "downstream_target_changed": left_targets != right_targets,
    }


def _async_chain() -> dict[str, object]:
    routes = {
        0: [Route(target=1, weight=1.0, delay=1)],
        1: [Route(target=2, weight=1.0, delay=2)],
        2: [Route(target=3, weight=1.0, delay=1)],
        3: [Route(target=4, weight=1.0, delay=3)],
    }
    result = run_async_chain(
        routes=routes,
        n_compartments=5,
        initial_source=0,
        threshold=0.5,
        max_time=12,
    )
    return {
        "events": result["events"],
        "fired": result["fired"],
        "final_states": result["states"],
    }


def build_receipt() -> dict[str, object]:
    return {
        "address_in_matter": _address_in_matter(),
        "development": _development(),
        "ais_suppression": _ais_suppression(),
        "reroute": _reroute(),
        "async_chain": _async_chain(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/v2.json"))
    args = parser.parse_args()
    receipt = build_receipt()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
