"""Frozen FusionMachine v3 equal-state recurrent-attacker benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from fusion_machine.benchmark_v3 import (
    EpisodeBatch,
    fit_latent_probe,
    generate_in_distribution,
    generate_long_dormancy,
    generate_probe_test,
    generate_probe_train,
    generate_rapid_switch,
    generate_training_batch,
    publication_slots,
)
from fusion_machine.recurrent import (
    ModelMasks,
    RNNParams,
    adam_step,
    effective_parameter_count,
    forward_sequence,
    init_adam,
    init_params,
    loss_and_grads,
    make_fusion_masks,
    make_generic_masks,
)

PRIMARY_METRICS = (
    "train_query_mse",
    "id_all_mse",
    "id_query_mse",
    "rapid_all_mse",
    "rapid_first_post_switch_mse",
    "long_offset0_abs",
    "sparse_receiver_mse",
    "routing_slot_accuracy",
    "latent_r2",
)


@dataclass
class TrainResult:
    params: RNNParams
    masks: ModelMasks
    final_metrics: dict[str, float]


def masks_for_variant(variant: str) -> ModelMasks:
    if variant == "fusion":
        return make_fusion_masks()
    if variant == "fusion_context_leak":
        return make_fusion_masks(context_leak=True)
    if variant == "fusion_dense_recurrence":
        return make_fusion_masks(dense_recurrence=True)
    if variant == "generic":
        return make_generic_masks(context_input=True)
    if variant == "generic_no_context":
        return make_generic_masks(context_input=False)
    raise ValueError(f"unknown variant: {variant}")


def train_one(seed: int, variant: str, steps: int = 1200) -> TrainResult:
    train = generate_training_batch()
    masks = masks_for_variant(variant)
    params = init_params(seed)
    adam = init_adam(params)
    metrics: dict[str, float] = {}
    for _ in range(steps):
        _, grads, metrics = loss_and_grads(
            params,
            masks,
            x=train.x,
            context=train.context,
            target=train.target,
            query_mask=train.query_mask,
            source_id=train.source_id,
            route_target=train.route_target,
            route_loss_weight=0.20,
            weight_decay=1e-4,
        )
        params, adam = adam_step(
            params,
            grads,
            adam,
            learning_rate=0.010,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            clip_norm=5.0,
        )
    _, _, metrics = loss_and_grads(
        params,
        masks,
        x=train.x,
        context=train.context,
        target=train.target,
        query_mask=train.query_mask,
        source_id=train.source_id,
        route_target=train.route_target,
        route_loss_weight=0.20,
        weight_decay=1e-4,
    )
    return TrainResult(
        params=params,
        masks=masks,
        final_metrics={k: float(v) for k, v in metrics.items()},
    )


def _mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    err2 = (np.asarray(pred) - np.asarray(target)) ** 2
    if mask is None:
        return float(np.mean(err2))
    selected = err2[np.asarray(mask, dtype=bool)]
    return float(np.mean(selected))


def _route_accuracy(logits: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    pred = np.argmax(logits, axis=-1)
    chosen = pred[np.asarray(mask, dtype=bool)]
    truth = target[np.asarray(mask, dtype=bool)]
    return float(np.mean(chosen == truth))


def _forward(params: RNNParams, masks: ModelMasks, batch: EpisodeBatch):
    return forward_sequence(
        params,
        masks,
        x=batch.x,
        context=batch.context,
        source_id=batch.source_id,
    )


def _sparse_receiver(pred: np.ndarray, target: np.ndarray, route_logits: np.ndarray, route_target: np.ndarray) -> dict[str, float]:
    slots = publication_slots(pred.shape[1])
    receiver = np.zeros(pred.shape[0], dtype=float)
    received = np.zeros_like(pred)
    for t in range(pred.shape[1]):
        if slots[t]:
            receiver = pred[:, t].copy()
        received[:, t] = receiver
    slot_mask = np.broadcast_to(slots[None, :], pred.shape)
    return {
        "receiver_mse": _mse(received, target),
        "publication_slots": int(np.sum(slots)),
        "route_accuracy": _route_accuracy(route_logits, route_target, slot_mask),
    }


def _latent_diagnostic(params: RNNParams, masks: ModelMasks) -> dict[str, float]:
    probe_train = generate_probe_train()
    probe_test = generate_probe_test()
    h_train = _forward(params, masks, probe_train).hidden[:, 1:]
    h_test = _forward(params, masks, probe_test).hidden[:, 1:]
    full = fit_latent_probe(
        h_train,
        probe_train.teacher_state,
        h_test,
        probe_test.teacher_state,
    )
    out: dict[str, float] = {"r2": float(full["r2"])}
    if masks.name.startswith("fusion"):
        a_from_b = fit_latent_probe(
            h_train[:, :, 2:],
            probe_train.teacher_state[:, :, :2],
            h_test[:, :, 2:],
            probe_test.teacher_state[:, :, :2],
        )
        b_from_a = fit_latent_probe(
            h_train[:, :, :2],
            probe_train.teacher_state[:, :, 2:],
            h_test[:, :, :2],
            probe_test.teacher_state[:, :, 2:],
        )
        out["teacher_A_from_B_half_r2"] = float(a_from_b["r2"])
        out["teacher_B_from_A_half_r2"] = float(b_from_a["r2"])
    return out


def evaluate_model(result: TrainResult) -> dict[str, object]:
    params, masks = result.params, result.masks
    train = generate_training_batch()
    id_batch = generate_in_distribution()
    rapid = generate_rapid_switch()
    long_gap = generate_long_dormancy()

    train_fwd = _forward(params, masks, train)
    id_fwd = _forward(params, masks, id_batch)
    rapid_fwd = _forward(params, masks, rapid)
    long_fwd = _forward(params, masks, long_gap)

    long_offsets = {}
    for offset in (0, 1, 2, 4, 8, 16):
        t = 48 + offset
        long_offsets[str(offset)] = float(np.mean(np.abs(long_fwd.prediction[:, t] - long_gap.target[:, t])))

    sparse = _sparse_receiver(
        rapid_fwd.prediction,
        rapid.target,
        rapid_fwd.route_logits,
        rapid.route_target,
    )

    routing_query = _route_accuracy(
        id_fwd.route_logits,
        id_batch.route_target,
        id_batch.query_mask,
    )

    return {
        "train": {
            "query_mse": _mse(train_fwd.prediction, train.target, train.query_mask),
            "objective_loss": float(result.final_metrics["loss"]),
        },
        "in_distribution": {
            "all_step_mse": _mse(id_fwd.prediction, id_batch.target),
            "query_mse": _mse(id_fwd.prediction, id_batch.target, id_batch.query_mask),
        },
        "rapid_switch": {
            "all_step_mse": _mse(rapid_fwd.prediction, rapid.target),
            "first_post_switch_mse": _mse(
                rapid_fwd.prediction,
                rapid.target,
                rapid.switch_mask,
            ),
        },
        "long_dormancy": {
            "mean_abs_error_by_offset": long_offsets,
            "offset0_abs_error": long_offsets["0"],
        },
        "sparse_publication": sparse,
        "routing": {
            "in_distribution_query_accuracy": routing_query,
            "publication_slot_accuracy": float(sparse["route_accuracy"]),
        },
        "latent_probe": _latent_diagnostic(params, masks),
        "representative_trace": {
            "target": [float(v) for v in rapid.target[0]],
            "prediction": [float(v) for v in rapid_fwd.prediction[0]],
            "context": [int(v) for v in rapid.context[0]],
            "switch": [bool(v) for v in rapid.switch_mask[0]],
        },
    }


def _primary_view(seed_result: dict[str, object]) -> dict[str, float]:
    return {
        "train_query_mse": float(seed_result["train"]["query_mse"]),
        "id_all_mse": float(seed_result["in_distribution"]["all_step_mse"]),
        "id_query_mse": float(seed_result["in_distribution"]["query_mse"]),
        "rapid_all_mse": float(seed_result["rapid_switch"]["all_step_mse"]),
        "rapid_first_post_switch_mse": float(seed_result["rapid_switch"]["first_post_switch_mse"]),
        "long_offset0_abs": float(seed_result["long_dormancy"]["offset0_abs_error"]),
        "sparse_receiver_mse": float(seed_result["sparse_publication"]["receiver_mse"]),
        "routing_slot_accuracy": float(seed_result["routing"]["publication_slot_accuracy"]),
        "latent_r2": float(seed_result["latent_probe"]["r2"]),
    }


def _compact_variant(variant: str, seed_results: list[dict[str, object]]) -> dict[str, object]:
    primary = {name: [] for name in PRIMARY_METRICS}
    long_offsets = {key: [] for key in ("0", "1", "2", "4", "8", "16")}
    diagnostic_values: dict[str, list[float]] = {}
    for evaluated in seed_results:
        view = _primary_view(evaluated)
        for name, value in view.items():
            primary[name].append(float(value))
        for key, value in evaluated["long_dormancy"]["mean_abs_error_by_offset"].items():
            long_offsets[key].append(float(value))
        for key, value in evaluated["latent_probe"].items():
            if key != "r2":
                diagnostic_values.setdefault(key, []).append(float(value))
    medians = {name: float(np.median(values)) for name, values in primary.items()}
    return {
        "variant": variant,
        "medians": medians,
        "raw_primary": primary,
        "long_offset_medians": {key: float(np.median(values)) for key, values in long_offsets.items()},
        "diagnostic_medians": {key: float(np.median(values)) for key, values in diagnostic_values.items()},
        "representative_trace": seed_results[0]["representative_trace"],
    }


def _run_variant(variant: str, seeds: Iterable[int], training_steps: int) -> dict[str, object]:
    seed_results = []
    for seed in seeds:
        trained = train_one(int(seed), variant, steps=training_steps)
        evaluated = evaluate_model(trained)
        seed_results.append({"seed": int(seed), **evaluated})
    return _compact_variant(variant, seed_results)


def _paired_summary(fusion: dict[str, object], generic: dict[str, object]) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for name in PRIMARY_METRICS:
        f = np.asarray(fusion["raw_primary"][name], dtype=float)
        g = np.asarray(generic["raw_primary"][name], dtype=float)
        metrics[name] = {
            "fusion_median": float(np.median(f)),
            "generic_median": float(np.median(g)),
            "fusion_minus_generic": [float(v) for v in (f - g)],
            "paired_median_difference": float(np.median(f - g)),
        }

    f_rapid = float(fusion["medians"]["rapid_first_post_switch_mse"])
    g_rapid = float(generic["medians"]["rapid_first_post_switch_mse"])
    f_long = float(fusion["medians"]["long_offset0_abs"])
    g_long = float(generic["medians"]["long_offset0_abs"])
    f_train = float(fusion["medians"]["train_query_mse"])
    g_train = float(generic["medians"]["train_query_mse"])
    f_sparse = float(fusion["medians"]["sparse_receiver_mse"])
    g_sparse = float(generic["medians"]["sparse_receiver_mse"])

    if g_rapid <= f_rapid and g_long <= f_long:
        classification = "generic_matches_or_wins_readiness"
    elif f_rapid < g_rapid and f_long < g_long and g_train <= f_train:
        classification = "fusion_readiness_inductive_bias"
    elif f_rapid < g_rapid and f_long < g_long:
        classification = "fusion_readiness_but_training_advantage_confounded"
    else:
        classification = "mixed_readiness_result"

    return {
        "metrics": metrics,
        "classification": classification,
        "sparse_publication_strengthens_fusion_case": bool(f_sparse < g_sparse),
        "no_significance_claim": True,
    }


def build_receipt(
    *,
    training_steps: int = 1200,
    seeds: tuple[int, ...] = tuple(range(8)),
) -> dict[str, object]:
    fusion_masks = make_fusion_masks()
    generic_masks = make_generic_masks()
    fusion = _run_variant("fusion", seeds, training_steps)
    generic = _run_variant("generic", seeds, training_steps)
    controls = {}
    for name in (
        "fusion_context_leak",
        "fusion_dense_recurrence",
        "generic_no_context",
    ):
        control = _run_variant(name, seeds, training_steps)
        control.pop("representative_trace", None)
        controls[name] = control
    return {
        "config": {
            "training_data_seed": 3101,
            "training_episodes": 96,
            "training_sequence_length": 64,
            "training_context_block_min": 16,
            "training_context_block_max": 32,
            "training_steps": int(training_steps),
            "optimizer_seeds": [int(s) for s in seeds],
            "learning_rate": 0.010,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "gradient_clip_norm": 5.0,
            "weight_decay": 1e-4,
            "route_loss_weight": 0.20,
        },
        "resource_budget": {
            "fusion_state_scalars": 4,
            "generic_state_scalars": 4,
            "fusion_route_parameters": 28,
            "generic_route_parameters": 28,
            "fusion_nonrouting_parameters": effective_parameter_count(fusion_masks, include_route=False),
            "generic_nonrouting_parameters": effective_parameter_count(generic_masks, include_route=False),
            "publication_slots_per_rapid_episode": int(np.sum(publication_slots(64))),
            "training_episodes_each": 96,
            "optimizer_updates_each": int(training_steps),
        },
        "fusion": fusion,
        "generic": generic,
        "controls": controls,
        "paired_summary": _paired_summary(fusion, generic),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/v3.json"))
    args = parser.parse_args()
    receipt = build_receipt()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
