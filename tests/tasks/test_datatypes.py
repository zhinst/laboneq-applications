"""Tests for the datatypes module."""

import pytest

from laboneq_applications.tasks.datatypes import AttributeWrapper, RunExperimentResults


def test_attribute_wrapper():
    """Test the AttributeWrapper class."""
    data = {
        "cal_trace/q0/g": 12345,
        "cal_trace/q1/g": 2345,
        "result/q0": 345,
    }
    wrapper = AttributeWrapper(data)
    assert (
        str(wrapper)
        == "{'cal_trace.q0.g': 12345, 'cal_trace.q1.g': 2345, 'result.q0': 345}"
        == repr(wrapper)
    )
    assert (
        str(wrapper.cal_trace)
        == "{'q0.g': 12345, 'q1.g': 2345}"
        == repr(wrapper.cal_trace)
    )
    assert "q0" in dir(wrapper.cal_trace)
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


def test_run_experiment_results():
    data = {"cal_trace/q0/g": 12345, "cal_trace/q1/g": 2345, "sweep_data/q0": 345}
    neartime_results = {"nt1": 12345, "nt2": {"a": "b", "c": "d"}}
    errors = [(0, "error1", "error1 message"), (1, "error2", "error2 message")]
    results = RunExperimentResults(data, neartime_results, errors)
    assert set(results.keys()) == {
        "cal_trace",
        "sweep_data",
        "neartime_callbacks",
        "errors",
    }
    assert set(results) == {"cal_trace", "sweep_data", "neartime_callbacks", "errors"}
    assert all(
        k in dir(results)
        for k in ["cal_trace", "sweep_data", "neartime_callbacks", "errors"]
    )
    assert results.errors is errors
    assert "nt1" in results.neartime_callbacks
    assert results.neartime_callbacks.nt2["a"] == "b"
    assert (
        str(results)
        == "{'cal_trace.q0.g': 12345, 'cal_trace.q1.g': 2345, 'sweep_data.q0': 345, "
        f"'neartime_callbacks': {neartime_results}, 'errors': {errors}}}"
    )
    assert (
        repr(results)
        == "{'cal_trace.q0.g': 12345, 'cal_trace.q1.g': 2345, 'sweep_data.q0': 345, "
        f"'neartime_callbacks': {neartime_results!r}, 'errors': {errors}}}"
    )
