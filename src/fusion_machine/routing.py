"""Binary event publication and material routing for FusionMachine v2."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Iterable


@dataclass
class DendriticCompartment:
    """Small resident state updated locally from incoming drive."""

    state: float = 0.0
    leak: float = 0.5

    def step(self, drive: float) -> float:
        if not 0.0 <= self.leak <= 1.0:
            raise ValueError("leak must be in [0, 1]")
        self.state = self.leak * self.state + (1.0 - self.leak) * float(drive)
        return self.state


@dataclass(frozen=True)
class Route:
    """One material axonal branch from a source identity to a target address."""

    target: int
    weight: float
    delay: int = 1

    def __post_init__(self) -> None:
        if self.delay < 1:
            raise ValueError("route delay must be >= 1")


def ais_event(state: float, threshold: float, suppressed: bool = False) -> int:
    """Publish one bit without modifying the resident state."""
    if suppressed:
        return 0
    return int(float(state) >= float(threshold))


def route_event(
    source: int,
    event: int,
    routes: dict[int, list[Route]],
) -> list[tuple[int, int, float]]:
    """Map one bit through the source's material route identity.

    Returns `(delay, target, weighted_payload)` tuples. The payload itself is
    only 0/1; destination information is supplied by the route table.
    """
    if event not in (0, 1):
        raise ValueError("event must be binary")
    if event == 0:
        return []
    return [
        (route.delay, route.target, route.weight * event)
        for route in routes.get(source, [])
    ]


def apply_writes(
    states: list[DendriticCompartment],
    writes: Iterable[tuple[int, float]],
) -> list[float]:
    """Apply local writes to addressed target compartments."""
    grouped: dict[int, float] = {}
    for target, value in writes:
        grouped[target] = grouped.get(target, 0.0) + float(value)
    for target, drive in grouped.items():
        states[target].step(drive)
    return [state.state for state in states]


def run_async_chain(
    *,
    routes: dict[int, list[Route]],
    n_compartments: int,
    initial_source: int,
    threshold: float = 0.5,
    max_time: int = 32,
) -> dict[str, object]:
    """Run an inspectable event-driven chain with route-specific delays.

    The initial source emits at t=0. Every target integrates only when a
    scheduled write arrives; if it crosses threshold it emits once. There is
    no layer clock and no synchronous sweep over a feed-forward depth index.
    """
    if not 0 <= initial_source < n_compartments:
        raise ValueError("initial_source out of range")

    compartments = [DendriticCompartment(state=0.0, leak=0.0) for _ in range(n_compartments)]
    fired = {initial_source}
    queue: list[tuple[int, int, int, float]] = []
    events: list[dict[str, float | int]] = []

    def emit(source: int, emit_time: int) -> None:
        for delay, target, value in route_event(source, 1, routes):
            arrival = emit_time + delay
            if arrival <= max_time:
                heappush(queue, (arrival, source, target, value))
                events.append(
                    {
                        "source": source,
                        "target": target,
                        "emit_time": emit_time,
                        "arrival_time": arrival,
                        "weight": float(value),
                    }
                )

    emit(initial_source, 0)

    while queue:
        arrival, source, target, value = heappop(queue)
        if arrival > max_time:
            break
        same_time = [(target, value)]
        while queue and queue[0][0] == arrival and queue[0][2] == target:
            _, _, target2, value2 = heappop(queue)
            same_time.append((target2, value2))
        drive = sum(v for _, v in same_time)
        compartments[target].step(drive)
        if target not in fired and ais_event(compartments[target].state, threshold):
            fired.add(target)
            emit(target, arrival)

    return {
        "events": events,
        "states": [float(comp.state) for comp in compartments],
        "fired": sorted(fired),
    }
