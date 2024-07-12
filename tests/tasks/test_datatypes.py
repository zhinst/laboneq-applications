"""Tests for the datatypes module."""

from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from textwrap import dedent

import pytest

import laboneq_applications.tasks.datatypes as tasks_datatypes
from laboneq_applications.tasks.datatypes import AttributeWrapper, RunExperimentResults


class PrettyPrinter:
    """A class to test the pretty printer."""

    def __init__(self) -> None:
        self.pretty_text: str | None = None

    def text(self, text: str):
        self.pretty_text = text


def test_attribute_wrapper():
    """Test the AttributeWrapper class."""
    data = {
        "cal_trace/q0/g": 12345,
        "cal_trace/q1/g": 2345,
        "result/q0": 345,
    }
    wrapper = AttributeWrapper(data)
    assert wrapper._as_str_dict() == {
        "cal_trace": {"q0": {"g": 12345}, "q1": {"g": 2345}},
        "result": {"q0": 345},
    }

    assert "q0" in dir(wrapper.cal_trace)
    assert isinstance(wrapper.cal_trace, AttributeWrapper)
    assert isinstance(wrapper.cal_trace.q0, AttributeWrapper)
    assert wrapper.cal_trace.q0.g == 12345
    subtree = wrapper.cal_trace
    assert isinstance(subtree, AttributeWrapper)
    assert len(subtree) == 2
    keys = subtree._keys()
    assert set(keys) == {"q0", "q1"}
    assert set(iter(subtree)) == {"q0", "q1"}
    assert "q0" in subtree
    assert "q0" in keys
    assert "not_a_key" not in subtree
    assert "not_a_key" not in keys
    assert {v["g"] for v in subtree._values()} == set(data.values()) - {345}
    items = subtree._items()
    assert {k for k, _ in items} == set(keys)
    assert [v for _, v in items] == list(subtree._values())
    subsubtree = subtree.q0
    assert isinstance(subsubtree, AttributeWrapper)
    assert subsubtree.g == 12345
    assert list(subsubtree._values()) == [12345]
    subtree_q1 = subtree["q1"]
    wrapper_caltrace = wrapper["cal_trace"]
    assert isinstance(subtree_q1, AttributeWrapper)
    assert isinstance(wrapper_caltrace, AttributeWrapper)
    assert isinstance(wrapper_caltrace.q0, AttributeWrapper)
    assert subtree_q1["g"] == 2345
    assert wrapper_caltrace.q0["g"] == 12345

    result = wrapper.result
    assert isinstance(result, AttributeWrapper)
    with pytest.raises(
        AttributeError,
        match="Key 'result/q1' not found in the data.",
    ):
        _ = result.q1

    with pytest.raises(AttributeError):
        _ = wrapper.results
    with pytest.raises(TypeError, match=r"Key 1 has to be of type str\."):
        _ = wrapper[1]


def test_attribute_wrapper_formatting():
    assert tasks_datatypes.use_rich_pprint
    try:
        data = {"cal_trace/q1/g": 2345}
        wrapper = AttributeWrapper(data)
        assert f"{wrapper}" == "{'cal_trace': {'q1': {'g': 2345}}}"
        assert f"{wrapper.cal_trace}" == "{'q1': {'g': 2345}}"
        assert (
            str(wrapper) == "{\n│   'cal_trace': {\n│   │   "
            "'q1': {\n│   │   │   'g': 2345\n│   │   }\n│   }\n}\n"
        )
        assert str(wrapper.cal_trace) == "{\n│   'q1': {\n│   │   'g': 2345\n│   }\n}\n"
        assert (
            repr(wrapper) == "AttributeWrapper(data={'cal_trace/q1/g': 2345}, "
            "path='', separator='/')"
        )
        assert (
            repr(wrapper.cal_trace)
            == "AttributeWrapper(data={'cal_trace/q1/g': 2345}, "
            "path='cal_trace', separator='/')"
        )
        p = PrettyPrinter()
        wrapper._repr_pretty_(p, None)
        assert str(wrapper) == p.pretty_text
        assert isinstance(wrapper.cal_trace, AttributeWrapper)
        wrapper.cal_trace._repr_pretty_(p, None)
        assert str(wrapper.cal_trace) == p.pretty_text
        tasks_datatypes.use_rich_pprint = False
        from pprint import pprint as pprint_pprint

        tasks_datatypes.pprint_pprint = pprint_pprint
        assert str(wrapper) == "{'cal_trace': {'q1': {'g': 2345}}}\n"
    finally:
        tasks_datatypes.use_rich_pprint = True


def test_run_experiment_results():
    data = {"cal_trace/q0/g": 12345, "cal_trace/q1/g": 2345, "sweep_data/q0": 345}
    neartime_results = {"nt1": 12345, "nt2": {"a": "b", "c": "d"}}
    errors = [(0, "error1", "error1 message"), (1, "error2", "error2 message")]
    results = RunExperimentResults(data, neartime_results, errors)
    assert set(results._keys()) == {
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
    assert results["errors"] is errors
    assert isinstance(results.neartime_callbacks, AttributeWrapper)
    assert results["neartime_callbacks"] == results.neartime_callbacks
    assert "nt1" in results.neartime_callbacks
    assert isinstance(results.neartime_callbacks, AttributeWrapper)
    assert results.neartime_callbacks.nt2 == neartime_results["nt2"]
    result_str = dedent("""\
            │   'sweep_data': {
            │   │   'q0': 345
            │   }""")
    q1_str = dedent("""\
            │   │   'q1': {
            │   │   │   'g': 2345
            │   │   }""")
    q0_str = dedent("""\
            │   │   'q0': {
            │   │   │   'g': 12345
            │   │   }""")
    error_str = dedent("""\
            │   'errors': [
            │   │   (
            │   │   │   0,
            │   │   │   'error1',
            │   │   │   'error1 message'
            │   │   ),
            │   │   (
            │   │   │   1,
            │   │   │   'error2',
            │   │   │   'error2 message'
            │   │   )
            │   ]""")
    nt1_str = dedent("""\
            │   │   'nt1': 12345""")
    nt2_str = dedent("""\
            │   │   'nt2': {
            │   │   │   'a': 'b',
            │   │   │   'c': 'd'
            │   │   }""")
    s = str(results)
    assert result_str in s
    assert q1_str in s
    assert q0_str in s
    assert error_str in s
    assert nt1_str in s
    assert nt2_str in s


def test_result_formatting():
    data = {"cal_trace/q0/g": 12345}
    neartime_results = {"nt2": {"a": "b", "c": "d"}}
    errors = [(0, "error1", "error1 message")]
    results = RunExperimentResults(data, neartime_results, errors)

    assert "'cal_trace': {'q0': {'g': 12345}}" in f"{results}"
    assert "'errors': [(0, 'error1', 'error1 message')]" in f"{results}"
    assert "'neartime_callbacks': {'nt2': {'a': 'b', 'c': 'd'}}" in f"{results}"
    assert (
        "'cal_trace': {\n│   │   'q0': {\n│   │   │   'g': 12345\n│   │   }\n│   }"
        in str(results)
    )
    assert (
        "'errors': [\n│   │   (\n│   │   │   0,\n│   │   │   'error1',\n│   │   │   "
        "'error1 message'\n│   │   )\n│   ]" in str(results)
    )
    assert (
        "'neartime_callbacks': {\n│   │   'nt2': {\n│   │   │   'a': 'b',\n│   │   │   "
        "'c': 'd'\n│   │   }\n│   }" in str(results)
    )
    assert (
        repr(results)
        == "RunExperimentResults(data={'cal_trace/q0/g': 12345}, near_time_callbacks="
        "{'nt2': {'a': 'b', 'c': 'd'}}, errors=[(0, 'error1', 'error1 message')], "
        "path = '', separator='/')"
    )
    assert f"{results.cal_trace}" == "{'q0': {'g': 12345}}"
    assert (
        repr(results.cal_trace)
        == "AttributeWrapper(data={'cal_trace/q0/g': 12345}, path='cal_trace', "
        "separator='/')"
    )

    p = PrettyPrinter()
    results._repr_pretty_(p, None)
    assert str(results) == p.pretty_text
    results.cal_trace._repr_pretty_(p, None)
    assert str(results.cal_trace) == p.pretty_text


@pytest.mark.parametrize(
    ("handles", "k1", "k2"),
    [
        (["a", "a.", "ab", "aa/a", "aa/b", "aa/b.", "aa/b../a"], None, None),
        (["a"], None, None),
        (["a/a"], None, None),
        ([], None, None),
        (
            ["a", "a.", "ab", "aa/a", "aa/b", "aa/b.", "a./b../a"],
            r"a\.",
            r"a\./b\.\./a",
        ),
        (["a", "a/a"], "a", "a/a"),
        (["a/a/a", "a/b", "a/a/a/a"], "a/a/a", "a/a/a/a"),
        (["a/a.", "a/./a", "a/a./a/a"], r"a/a\.", r"a/a\./a/a"),
        (
            ["q0/data", "q0/data/active_reset/a", "q0/data/active_reset/b"],
            "q0/data",
            "q0/data/active_reset/.",
        ),
    ],
)
def test_check_unique_key_classes(handles: list[str], k1: str | None, k2: str | None):
    d = {k: None for k in handles}
    if k1 is None:
        with does_not_raise():
            AttributeWrapper(d)
    else:
        with pytest.raises(ValueError, match=f"Key '{k1}' is a prefix of '{k2}'."):
            AttributeWrapper(d)


@pytest.mark.parametrize(
    ("handles", "keys"),
    [
        (["__init___", "__init_/_"], None),
        (["__init__"], "{'__init__'}"),
        (["_add_path"], "{'_add_path'}"),
        (["__len__/a"], "{'__len__'}"),
        (["__len__/a/b"], "{'__len__'}"),
        (["a/__len__/b"], "{'__len__'}"),
        (["a/__len__"], "{'__len__'}"),
        (
            ["__getitem__", "_as_str_dict"],
            "({'__getitem__', '_as_str_dict'})|({'_as_str_dict', '__getitem__'})",
        ),
    ],
)
def test_key_is_attribute(handles: list[str], keys: str | None):
    d = {k: None for k in handles}
    if keys is None:
        with does_not_raise():
            AttributeWrapper(d)
    else:
        with pytest.raises(ValueError, match=f"Handles {keys} aren't allowed names\\."):
            AttributeWrapper(d)
