"""Verify the compact frozen v3 receipt against a full deterministic run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    args = parser.parse_args()
    full = json.loads(args.full.read_text())
    frozen = json.loads(args.frozen.read_text())
    observed = compact_view(full)
    if observed != frozen:
        raise SystemExit("v3 frozen receipt does not match the deterministic full run")
    print("v3 compact frozen receipt matches full deterministic run")


if __name__ == "__main__":
    main()
