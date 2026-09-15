"""Frozen nonlinear world and evaluation regimes for FusionMachine v3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray

M_A = np.array([[0.78, 0.22], [-0.18, 0.70]], dtype=float)
U_A = np.array([[0.55, 0.10, 0.30], [-0.15, 0.50, 0.20]], dtype=float)
Y_A = np.array([0.80, -0.35], dtype=float)
M_B = np.array([[-0.45, 0.62], [0.30, 0.74]], dtype=float)
U_B = np.array([[0.10, 0.60, -0.35], [0.45, -0.10, 0.40]], dtype=float)
Y_B = np.array([-0.40, 0.90], dtype=float)


@dataclass(frozen=True)
class EpisodeBatch:
    x: Array
    context: Array
    target: Array
    query_mask: Array
    source_id: Array
    route_target: Array
    teacher_state: Array
    switch_mask: Array


def _markov_bits(rng: np.random.Generator, batch: int, steps: int, flip_prob: float = 0.15) -> tuple[Array, Array]:
    u = np.empty((batch, steps), dtype=float)
    v = np.empty((batch, steps), dtype=float)
    u[:, 0] = rng.choice([-1.0, 1.0], size=batch)
    v[:, 0] = rng.choice([-1.0, 1.0], size=batch)
    if steps > 1:
        u_flips = rng.random((batch, steps - 1)) < flip_prob
        v_flips = rng.random((batch, steps - 1)) < flip_prob
        for t in range(1, steps):
            u[:, t] = u[:, t - 1] * np.where(u_flips[:, t - 1], -1.0, 1.0)
            v[:, t] = v[:, t - 1] * np.where(v_flips[:, t - 1], -1.0, 1.0)
    return u, v


def _blocked_context(
    rng: np.random.Generator,
    batch: int,
    steps: int,
    block_min: int,
    block_max: int,
) -> tuple[Array, Array, Array]:
    context = np.empty((batch, steps), dtype=int)
    query = np.zeros((batch, steps), dtype=bool)
    switch = np.zeros((batch, steps), dtype=bool)
    for b in range(batch):
        mode = int(rng.integers(0, 2))
        pos = 0
        first = True
        while pos < steps:
            block = int(rng.integers(block_min, block_max + 1))
            end = min(steps, pos + block)
            context[b, pos:end] = mode
            if not first:
                switch[b, pos] = True
            query[b, end - 1] = True
            first = False
            pos = end
            mode = 1 - mode
    query[:, -1] = True
    return context, query, switch


def teacher_rollout(x: Array, context: Array) -> tuple[Array, Array]:
    x = np.asarray(x, dtype=float)
    context = np.asarray(context, dtype=int)
    if x.ndim != 3 or x.shape[2] != 4:
        raise ValueError("x must have shape (batch, steps, 4)")
    if context.shape != x.shape[:2]:
        raise ValueError("context must match x[:2]")
    batch, steps, _ = x.shape
    state = np.zeros((batch, 4), dtype=float)
    history = np.zeros((batch, steps, 4), dtype=float)
    target = np.zeros((batch, steps), dtype=float)
    for t in range(steps):
        raw = x[:, t, :3]
        a = np.tanh(state[:, :2] @ M_A.T + raw @ U_A.T)
        b = np.tanh(state[:, 2:] @ M_B.T + raw @ U_B.T)
        state = np.concatenate([a, b], axis=1)
        history[:, t] = state
        y_a = a @ Y_A
        y_b = b @ Y_B
        target[:, t] = np.where(context[:, t] == 0, y_a, y_b)
    return history, target


def _make_batch(
    *,
    seed: int,
    batch: int,
    steps: int,
    block_min: int,
    block_max: int,
    all_queries: bool = False,
) -> EpisodeBatch:
    rng = np.random.default_rng(seed)
    u, v = _markov_bits(rng, batch, steps)
    context, query, switch = _blocked_context(rng, batch, steps, block_min, block_max)
    if all_queries:
        query[:] = True
    p = u * v
    x = np.stack([u, v, p, context.astype(float)], axis=-1)
    source_id = rng.integers(0, 4, size=batch, dtype=int)
    route_target = (source_id[:, None] + 2 * context) % 4
    teacher_state, target = teacher_rollout(x, context)
    return EpisodeBatch(
        x=x,
        context=context,
        target=target,
        query_mask=query,
        source_id=source_id,
        route_target=route_target,
        teacher_state=teacher_state,
        switch_mask=switch,
    )


def generate_training_batch() -> EpisodeBatch:
    return _make_batch(seed=3101, batch=96, steps=64, block_min=16, block_max=32)


def generate_in_distribution() -> EpisodeBatch:
    return _make_batch(seed=4101, batch=256, steps=64, block_min=16, block_max=32)


def generate_rapid_switch() -> EpisodeBatch:
    return _make_batch(
        seed=4201,
        batch=256,
        steps=64,
        block_min=2,
        block_max=6,
        all_queries=True,
    )


def generate_long_dormancy() -> EpisodeBatch:
    seed = 4301
    batch = 256
    steps = 96
    rng = np.random.default_rng(seed)
    u, v = _markov_bits(rng, batch, steps)
    first_mode = rng.integers(0, 2, size=batch, dtype=int)
    context = np.empty((batch, steps), dtype=int)
    context[:, :48] = first_mode[:, None]
    context[:, 48:] = (1 - first_mode)[:, None]
    query = np.ones((batch, steps), dtype=bool)
    switch = np.zeros((batch, steps), dtype=bool)
    switch[:, 48] = True
    p = u * v
    x = np.stack([u, v, p, context.astype(float)], axis=-1)
    source_id = rng.integers(0, 4, size=batch, dtype=int)
    route_target = (source_id[:, None] + 2 * context) % 4
    teacher_state, target = teacher_rollout(x, context)
    return EpisodeBatch(
        x=x,
        context=context,
        target=target,
        query_mask=query,
        source_id=source_id,
        route_target=route_target,
        teacher_state=teacher_state,
        switch_mask=switch,
    )


def generate_probe_train() -> EpisodeBatch:
    return _make_batch(seed=4401, batch=256, steps=64, block_min=8, block_max=20, all_queries=True)


def generate_probe_test() -> EpisodeBatch:
    return _make_batch(seed=4501, batch=256, steps=64, block_min=8, block_max=20, all_queries=True)


def publication_slots(steps: int = 64) -> Array:
    mask = np.zeros(steps, dtype=bool)
    mask[::4] = True
    return mask


def fit_latent_probe(
    hidden_train: Array,
    teacher_train: Array,
    hidden_test: Array,
    teacher_test: Array,
) -> dict[str, object]:
    hidden_train = np.asarray(hidden_train, dtype=float).reshape(-1, hidden_train.shape[-1])
    teacher_train = np.asarray(teacher_train, dtype=float).reshape(-1, teacher_train.shape[-1])
    hidden_test = np.asarray(hidden_test, dtype=float).reshape(-1, hidden_test.shape[-1])
    teacher_test = np.asarray(teacher_test, dtype=float).reshape(-1, teacher_test.shape[-1])
    X_train = np.concatenate([hidden_train, np.ones((len(hidden_train), 1))], axis=1)
    X_test = np.concatenate([hidden_test, np.ones((len(hidden_test), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(X_train, teacher_train, rcond=None)
    pred = X_test @ coef
    sse = float(np.sum((teacher_test - pred) ** 2))
    centered = teacher_test - np.mean(teacher_test, axis=0, keepdims=True)
    sst = float(np.sum(centered**2))
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    return {"r2": float(r2), "coef": coef}
