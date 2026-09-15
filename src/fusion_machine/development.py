"""Synthetic developmental routing for FusionMachine v2.

Chemistry carries a coarse target-family prior while activity carries a
complementary within-family signature. The combination is deliberately
constructed to be informative without making either cue sufficient alone.
"""

from __future__ import annotations

import numpy as np


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def make_problem(
    seed: int,
    n_sources: int = 12,
    n_targets: int = 12,
    dim: int = 6,
) -> dict[str, np.ndarray]:
    """Create the frozen complementary-cue development control.

    Targets occupy 3 coarse chemical families with 4 activity-defined slots
    per family. Chemistry is therefore ambiguous within a family; activity is
    ambiguous across families. Noise makes both cues imperfect rather than
    exact categorical labels.
    """
    if n_sources != n_targets or n_targets != 12 or dim < 3:
        raise ValueError("frozen v2 control uses 12 sources/targets and dim >= 3")

    rng = np.random.default_rng(seed)
    truth = rng.permutation(n_targets)
    n_families = 3
    slots_per_family = 4

    family_basis = np.zeros((n_families, dim), dtype=float)
    family_basis[:, :n_families] = np.eye(n_families)

    target_fp = np.empty((n_targets, dim), dtype=float)
    for target in range(n_targets):
        family = target // slots_per_family
        target_fp[target] = family_basis[family] + 0.28 * rng.normal(size=dim)

    source_fp = np.empty((n_sources, dim), dtype=float)
    for source, target in enumerate(truth):
        family = target // slots_per_family
        source_fp[source] = family_basis[family] + 0.42 * rng.normal(size=dim)

    target_fp = _normalize_rows(target_fp)
    source_fp = _normalize_rows(source_fp)

    episodes = 16
    slot_templates = _normalize_rows(rng.normal(size=(slots_per_family, episodes)))

    target_activity = np.empty((n_targets, episodes), dtype=float)
    for target in range(n_targets):
        slot = target % slots_per_family
        target_activity[target] = slot_templates[slot] + 0.22 * rng.normal(size=episodes)

    source_activity = np.empty((n_sources, episodes), dtype=float)
    for source, target in enumerate(truth):
        slot = target % slots_per_family
        source_activity[source] = slot_templates[slot] + 0.34 * rng.normal(size=episodes)

    return {
        "truth": truth.astype(int),
        "source_fp": source_fp,
        "target_fp": target_fp,
        "source_activity": source_activity,
        "target_activity": target_activity,
    }


def chemical_scores(source_fp: np.ndarray, target_fp: np.ndarray) -> np.ndarray:
    """Cosine compatibility between source and target chemical fingerprints."""
    source = _normalize_rows(np.asarray(source_fp, dtype=float))
    target = _normalize_rows(np.asarray(target_fp, dtype=float))
    return source @ target.T


def activity_alignment(
    source_activity: np.ndarray,
    target_activity: np.ndarray,
) -> np.ndarray:
    """Centered cosine alignment between developmental activity traces."""
    source = np.asarray(source_activity, dtype=float)
    target = np.asarray(target_activity, dtype=float)
    source = _normalize_rows(source - source.mean(axis=1, keepdims=True))
    target = _normalize_rows(target - target.mean(axis=1, keepdims=True))
    return source @ target.T


def _row_zscore(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (scores - mean) / std


def grow_routes(
    chemistry: np.ndarray,
    activity: np.ndarray,
    beta: float,
    top_k: int = 1,
) -> np.ndarray:
    """Freeze the strongest routes after combining normalized cues."""
    chemistry = np.asarray(chemistry, dtype=float)
    activity = np.asarray(activity, dtype=float)
    if chemistry.shape != activity.shape:
        raise ValueError("chemistry and activity must have matching shapes")
    if not 1 <= top_k <= chemistry.shape[1]:
        raise ValueError("top_k out of range")

    combined = _row_zscore(chemistry) + float(beta) * _row_zscore(activity)
    routes = np.zeros_like(combined, dtype=int)
    winners = np.argpartition(combined, -top_k, axis=1)[:, -top_k:]
    for row, cols in enumerate(winners):
        routes[row, cols] = 1
    return routes


def graph_accuracy(routes: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of sources whose true target is retained by the frozen graph."""
    routes = np.asarray(routes)
    truth = np.asarray(truth, dtype=int)
    return float(np.mean([routes[i, truth[i]] > 0 for i in range(len(truth))]))


def routing_accuracy(routes: np.ndarray, truth: np.ndarray) -> float:
    """Held-out one-bit routing accuracy for one-target-per-source controls."""
    return graph_accuracy(routes, truth)
