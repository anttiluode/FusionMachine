import pytest

from experiments.check_v3_frozen import assert_receipts_close


def test_receipt_comparison_accepts_last_bit_float_noise():
    expected = {"x": 0.11115676669177654, "nested": [1, True, "fixed"]}
    observed = {"x": 0.1111567666917766, "nested": [1, True, "fixed"]}
    assert_receipts_close(observed, expected)


def test_receipt_comparison_rejects_scientifically_meaningful_float_drift():
    with pytest.raises(AssertionError):
        assert_receipts_close({"x": 0.101}, {"x": 0.100})


def test_receipt_comparison_rejects_structure_or_exact_field_drift():
    with pytest.raises(AssertionError):
        assert_receipts_close({"classification": "different"}, {"classification": "frozen"})
    with pytest.raises(AssertionError):
        assert_receipts_close({"values": [1, 2]}, {"values": [1, 2, 3]})
