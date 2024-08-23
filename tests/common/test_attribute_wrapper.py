"""Test the AttributeWrapper class."""

from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

import laboneq_applications.common.classformatter as common_classformatter
from laboneq_applications.common.attribute_wrapper import AttributeWrapper


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
    assert common_classformatter.use_rich_pprint
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
        common_classformatter.use_rich_pprint = False
        from pprint import pprint as pprint_pprint

        common_classformatter.pprint_pprint = pprint_pprint
        assert str(wrapper) == "{'cal_trace': {'q1': {'g': 2345}}}\n"
    finally:
        common_classformatter.use_rich_pprint = True


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
        with pytest.raises(
            ValueError,
            match=f"Handle '{k1}' is a prefix of handle '{k2}', which is not "
            "allowed, because a results entry cannot contain both data and "
            "another results subtree. Please rename one of the handles.",
        ):
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
