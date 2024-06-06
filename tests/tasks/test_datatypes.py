"""Tests for the datatypes module."""

import pytest

from laboneq_applications.tasks.datatypes import AttributeWrapper


def test_attribute_wrapper():
    """Test the AttributeWrapper class."""
    data = {
        "cal_trace/q0/g": 12345,
        "cal_trace/q1/g": 2345,
        "result/q0": 345,
    }
    wrapper = AttributeWrapper(data)
    assert wrapper.cal_trace.q0.g == 12345
    subtree = wrapper.cal_trace
    assert len(subtree) == 2
    keys = subtree.keys()
    assert set(keys) == {"q0", "q1"}
    assert set(iter(subtree)) == {"q0", "q1"}
    assert "q0" in subtree
    assert "q0" in keys
    assert "not_a_key" not in subtree
    assert "not_a_key" not in keys
    assert {v["g"] for v in subtree.values()} == set(data.values()) - {345}
    subsubtree = subtree.q0
    assert subsubtree.g == 12345
    assert list(subsubtree.values()) == [12345]
    assert subtree["q1"]["g"] == 2345
    assert wrapper["cal_trace"].q0["g"] == 12345

    with pytest.raises(
        AttributeError,
        match="Key 'result/q1' not found in the data.",
    ):
        _ = wrapper.result.q1

    with pytest.raises(AttributeError):
        _ = wrapper.results
