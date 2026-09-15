"""Masked recurrent models and deterministic NumPy BPTT for FusionMachine v3."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class ModelMasks:
    W_hh: Array
    W_x: Array
    W_out: Array
    name: str = "model"

    @property
    def hidden_size(self) -> int:
        return int(self.W_hh.shape[0])

    @property
    def route_parameter_count(self) -> int:
        return 28


@dataclass
class RNNParams:
    W_hh: Array
    W_x: Array
    b_h: Array
    W_out: Array
    b_out: Array
    R: Array
    r_bias: Array

    def copy(self) -> "RNNParams":
        return RNNParams(**{f.name: getattr(self, f.name).copy() for f in fields(self)})


@dataclass
class ForwardResult:
    hidden: Array
    heads: Array
    prediction: Array
    route_features: Array
    route_logits: Array


@dataclass
class AdamState:
    m: RNNParams
    v: RNNParams
    step: int = 0


def _base_masks() -> tuple[Array, Array, Array]:
    return (
        np.ones((4, 4), dtype=float),
        np.ones((4, 4), dtype=float),
        np.ones((2, 4), dtype=float),
    )


def make_fusion_masks(*, context_leak: bool = False, dense_recurrence: bool = False) -> ModelMasks:
    W_hh, W_x, W_out = _base_masks()
    if not dense_recurrence:
        W_hh[:2, 2:] = 0.0
        W_hh[2:, :2] = 0.0
    if not context_leak:
        W_x[:, 3] = 0.0
    W_out[0, 2:] = 0.0
    W_out[1, :2] = 0.0
    name = "fusion"
    if context_leak:
        name += "_context_leak"
    if dense_recurrence:
        name += "_dense_recurrence"
    return ModelMasks(W_hh=W_hh, W_x=W_x, W_out=W_out, name=name)


def make_generic_masks(*, context_input: bool = True) -> ModelMasks:
    W_hh, W_x, W_out = _base_masks()
    if not context_input:
        W_x[:, 3] = 0.0
    return ModelMasks(
        W_hh=W_hh,
        W_x=W_x,
        W_out=W_out,
        name="generic" if context_input else "generic_no_context",
    )


def effective_parameter_count(masks: ModelMasks, *, include_route: bool = True) -> int:
    count = int(np.sum(masks.W_hh) + np.sum(masks.W_x) + np.sum(masks.W_out))
    count += 4 + 2
    if include_route:
        count += masks.route_parameter_count
    return count


def init_params(seed: int, scale: float = 0.15) -> RNNParams:
    rng = np.random.default_rng(seed)
    return RNNParams(
        W_hh=rng.normal(0.0, scale, size=(4, 4)),
        W_x=rng.normal(0.0, scale, size=(4, 4)),
        b_h=np.zeros(4, dtype=float),
        W_out=rng.normal(0.0, scale, size=(2, 4)),
        b_out=np.zeros(2, dtype=float),
        R=rng.normal(0.0, scale, size=(4, 6)),
        r_bias=np.zeros(4, dtype=float),
    )


def _zeros_like(params: RNNParams) -> RNNParams:
    return RNNParams(**{f.name: np.zeros_like(getattr(params, f.name)) for f in fields(params)})


def _route_features(source_id: Array, context: Array) -> Array:
    batch, steps = context.shape
    source_id = np.asarray(source_id, dtype=int)
    if source_id.shape != (batch,):
        raise ValueError("source_id must have shape (batch,)")
    feat = np.zeros((batch, steps, 6), dtype=float)
    feat[np.arange(batch), :, source_id] = 1.0
    rows = np.arange(batch)[:, None]
    cols = np.arange(steps)[None, :]
    feat[rows, cols, 4 + context] = 1.0
    return feat


def forward_sequence(
    params: RNNParams,
    masks: ModelMasks,
    *,
    x: Array,
    context: Array,
    source_id: Array,
) -> ForwardResult:
    x = np.asarray(x, dtype=float)
    context = np.asarray(context, dtype=int)
    if x.ndim != 3 or x.shape[2] != 4:
        raise ValueError("x must have shape (batch, steps, 4)")
    if context.shape != x.shape[:2]:
        raise ValueError("context must match x[:2]")

    batch, steps, _ = x.shape
    hidden = np.zeros((batch, steps + 1, 4), dtype=float)
    W_hh = params.W_hh * masks.W_hh
    W_x = params.W_x * masks.W_x
    W_out = params.W_out * masks.W_out

    for t in range(steps):
        pre = hidden[:, t] @ W_hh.T + x[:, t] @ W_x.T + params.b_h
        hidden[:, t + 1] = np.tanh(pre)

    h = hidden[:, 1:]
    heads = h @ W_out.T + params.b_out
    rows = np.arange(batch)[:, None]
    cols = np.arange(steps)[None, :]
    prediction = heads[rows, cols, context]

    route_features = _route_features(np.asarray(source_id), context)
    route_logits = route_features @ params.R.T + params.r_bias
    return ForwardResult(
        hidden=hidden,
        heads=heads,
        prediction=prediction,
        route_features=route_features,
        route_logits=route_logits,
    )


def _softmax(logits: Array) -> Array:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def loss_and_grads(
    params: RNNParams,
    masks: ModelMasks,
    *,
    x: Array,
    context: Array,
    target: Array,
    query_mask: Array,
    source_id: Array,
    route_target: Array,
    route_loss_weight: float = 0.20,
    weight_decay: float = 1e-4,
) -> tuple[float, RNNParams, dict[str, float]]:
    target = np.asarray(target, dtype=float)
    query_mask = np.asarray(query_mask, dtype=bool)
    route_target = np.asarray(route_target, dtype=int)
    result = forward_sequence(params, masks, x=x, context=context, source_id=source_id)
    if target.shape != result.prediction.shape or query_mask.shape != target.shape:
        raise ValueError("target/query_mask must match prediction shape")
    if route_target.shape != target.shape:
        raise ValueError("route_target must match target shape")

    q = query_mask.astype(float)
    n_query = float(np.sum(q))
    if n_query <= 0:
        raise ValueError("query_mask must select at least one point")

    err = result.prediction - target
    mse = float(np.sum(q * err * err) / n_query)

    probs = _softmax(result.route_logits)
    batch, steps = target.shape
    rows = np.arange(batch)[:, None]
    cols = np.arange(steps)[None, :]
    chosen = np.clip(probs[rows, cols, route_target], 1e-12, 1.0)
    route_ce = float(-np.sum(q * np.log(chosen)) / n_query)

    W_hh_eff = params.W_hh * masks.W_hh
    W_x_eff = params.W_x * masks.W_x
    W_out_eff = params.W_out * masks.W_out
    decay_loss = 0.5 * weight_decay * (
        np.sum(W_hh_eff * W_hh_eff)
        + np.sum(W_x_eff * W_x_eff)
        + np.sum(W_out_eff * W_out_eff)
        + np.sum(params.R * params.R)
    )
    loss = float(mse + route_loss_weight * route_ce + decay_loss)

    grads = _zeros_like(params)

    d_prediction = (2.0 / n_query) * q * err
    d_heads = np.zeros_like(result.heads)
    d_heads[rows, cols, context] = d_prediction
    h = result.hidden[:, 1:]
    grads.W_out = np.einsum("bto,bth->oh", d_heads, h)
    grads.b_out = np.sum(d_heads, axis=(0, 1))
    dh = d_heads @ W_out_eff

    dh_future = np.zeros((batch, 4), dtype=float)
    for t in range(steps - 1, -1, -1):
        dh_total = dh[:, t] + dh_future
        h_t = result.hidden[:, t + 1]
        da = dh_total * (1.0 - h_t * h_t)
        h_prev = result.hidden[:, t]
        grads.W_hh += da.T @ h_prev
        grads.W_x += da.T @ x[:, t]
        grads.b_h += np.sum(da, axis=0)
        dh_future = da @ W_hh_eff

    d_logits = probs.copy()
    d_logits[rows, cols, route_target] -= 1.0
    d_logits *= (q / n_query)[..., None]
    d_logits *= route_loss_weight
    grads.R = np.einsum("btk,btf->kf", d_logits, result.route_features)
    grads.r_bias = np.sum(d_logits, axis=(0, 1))

    grads.W_hh *= masks.W_hh
    grads.W_x *= masks.W_x
    grads.W_out *= masks.W_out
    grads.W_hh += weight_decay * W_hh_eff
    grads.W_x += weight_decay * W_x_eff
    grads.W_out += weight_decay * W_out_eff
    grads.R += weight_decay * params.R

    metrics = {"mse": mse, "route_ce": route_ce, "loss": loss}
    return loss, grads, metrics


def init_adam(params: RNNParams) -> AdamState:
    return AdamState(m=_zeros_like(params), v=_zeros_like(params), step=0)


def _global_norm(grads: RNNParams) -> float:
    total = 0.0
    for f in fields(grads):
        arr = getattr(grads, f.name)
        total += float(np.sum(arr * arr))
    return float(np.sqrt(total))


def adam_step(
    params: RNNParams,
    grads: RNNParams,
    state: AdamState,
    *,
    learning_rate: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    clip_norm: float = 5.0,
) -> tuple[RNNParams, AdamState]:
    norm = _global_norm(grads)
    scale = 1.0 if norm <= clip_norm or norm == 0.0 else clip_norm / norm
    state.step += 1
    t = state.step

    for f in fields(params):
        name = f.name
        g = getattr(grads, name) * scale
        m = getattr(state.m, name)
        v = getattr(state.v, name)
        m *= beta1
        m += (1.0 - beta1) * g
        v *= beta2
        v += (1.0 - beta2) * (g * g)
        mhat = m / (1.0 - beta1**t)
        vhat = v / (1.0 - beta2**t)
        getattr(params, name)[:] -= learning_rate * mhat / (np.sqrt(vhat) + epsilon)
    return params, state
