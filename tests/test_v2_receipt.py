import json
from pathlib import Path

from experiments.run_v2 import build_receipt


def test_v2_receipt_contains_all_scientific_gates():
    receipt = build_receipt()
    assert set(receipt) >= {
        "address_in_matter",
        "development",
        "ais_suppression",
        "reroute",
        "async_chain",
    }

    address = receipt["address_in_matter"]
    assert address["intact_accuracy"] == 1.0
    assert address["pooled_source_accuracy"] == 0.125
    assert address["shuffled_route_accuracy"] == 0.0
    assert address["digital_address_oracle_accuracy"] == 1.0

    development = receipt["development"]
    assert development["combined_graph_accuracy"] > development["chemistry_only_graph_accuracy"]
    assert development["combined_graph_accuracy"] > development["activity_only_graph_accuracy"]
    assert development["combined_graph_accuracy"] > development["shuffled_activity_graph_accuracy"]

    suppression = receipt["ais_suppression"]
    assert suppression["resident_max_abs_difference"] == 0.0
    assert suppression["suppressed_event_count"] == 0
    assert suppression["first_post_suppression_event"] == 1

    reroute = receipt["reroute"]
    assert reroute["source_event_trace_identical"] is True
    assert reroute["downstream_target_changed"] is True

    assert len(receipt["async_chain"]["events"]) >= 4


def test_checked_in_v2_receipt_matches_fresh_generation():
    frozen = json.loads(Path("results/v2.json").read_text())
    assert build_receipt() == frozen
