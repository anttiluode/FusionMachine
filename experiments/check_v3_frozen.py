"""Verify the compact frozen v3 receipt against a full deterministic run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def compact_view(full: dict[str, object]) -> dict[str, object]:
    def variant_view(value: dict[str, object]) -> dict[str, object]:
        return {
            "diagnostic_medians": value["diagnostic_medians"],
            "long_offset_medians": value["long_offset_medians"],
            "medians": value["medians"],
        }

    paired = full["paired_summary"]
    return {
        "config": full["config"],
        "controls": {
            name: variant_view(value)
            for name, value in full["controls"].items()
        },
        "fusion": variant_view(full["fusion"]),
        "generic": variant_view(full["generic"]),
        "paired_summary": {
            "classification": paired["classification"],
            "metrics": {
                name: {
                    "fusion_median": value["fusion_median"],
                    "generic_median": value["generic_median"],
                    "paired_median_difference": value["paired_median_difference"],
                }
                for name, value in paired["metrics"].items()
            },
            "no_significance_claim": paired["no_significance_claim"],
            "sparse_publication_strengthens_fusion_case": paired[
                "sparse_publication_strengthens_fusion_case"
            ],
        },
        "resource_budget": full["resource_budget"],
    }


def assert_receipts_close(
    observed: Any,
    expected: Any,
    *,
    path: str = "$",
    rel_tol: float = 1e-12,
    abs_tol: float = 1e-15,
) -> None:
    """Compare receipt structure exactly and floating values numerically.

    Deterministic NumPy workloads can differ by a few final floating-point bits
    across supported Python/NumPy builds. Integers, booleans, strings, keys and
    sequence structure remain exact; only floats receive this tight tolerance.
    """
    if isinstance(expected, dict):
        assert isinstance(observed, dict), f"{path}: expected mapping"
        assert observed.keys() == expected.keys(), f"{path}: mapping keys differ"
        for key in expected:
            assert_receipts_close(
                observed[key],
                expected[key],
                path=f"{path}.{key}",
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
        return

    if isinstance(expected, list):
        assert isinstance(observed, list), f"{path}: expected list"
        assert len(observed) == len(expected), f"{path}: list length differs"
        for index, (observed_value, expected_value) in enumerate(zip(observed, expected)):
            assert_receipts_close(
                observed_value,
                expected_value,
                path=f"{path}[{index}]",
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
        return

    if isinstance(expected, float):
        assert isinstance(observed, (int, float)) and not isinstance(observed, bool), (
            f"{path}: expected numeric float"
        )
        assert math.isclose(
            float(observed),
            expected,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ), f"{path}: observed {observed!r} != frozen {expected!r}"
        return

    assert type(observed) is type(expected), f"{path}: type differs"
    assert observed == expected, f"{path}: observed {observed!r} != frozen {expected!r}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    args = parser.parse_args()
    full = json.loads(args.full.read_text())
    frozen = json.loads(args.frozen.read_text())
    observed = compact_view(full)
    try:
        assert_receipts_close(observed, frozen)
    except AssertionError as exc:
        raise SystemExit(f"v3 frozen receipt mismatch: {exc}") from exc
    print("v3 compact frozen receipt matches full deterministic run")


if __name__ == "__main__":
    main()
