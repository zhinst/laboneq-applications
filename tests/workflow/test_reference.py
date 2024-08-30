import pytest

from laboneq_applications.workflow.reference import (
    Reference,
    add_overwrite,
    get_default,
    get_ref,
    notset,
    unwrap,
)


class TestReference:
    def test_getitem(self):
        head = Reference(None)
        child = head["key"]
        assert unwrap(head, {"key": 123}) == {"key": 123}

        head = Reference(None)
        child = head[0]["key"]
        assert unwrap(child, ({"key": 123},)) == 123

    @pytest.mark.parametrize(
        ("one", "other", "result"),
        [
            (2, 2, True),
            (2, 3, False),
            ("a", "a", True),
            ("a", "b", False),
            ([1], [1], True),
        ],
    )
    def test_eq(self, one, other, result):
        obj = Reference(None)
        expr = obj == other
        assert unwrap(expr, one) is result

    def test_is_comparison(self):
        ref = Reference(None)
        x = ref is True
        assert x is False

        x = ref is False
        assert x is False

    def test_getattr(self):
        class SomeObject:
            def __init__(self, x) -> None:
                self.x = x

        ref = Reference(None)
        head = ref.x
        assert unwrap(head, SomeObject(x=1)) == 1
        child = head.x
        assert unwrap(child, SomeObject(x=SomeObject(x=2))) == 2

        ref1 = Reference(None)
        head1 = ref1.y
        with pytest.raises(
            AttributeError,
            match="'SomeObject' object has no attribute 'y'",
        ):
            unwrap(head1, SomeObject(x=1))

        ref2 = Reference(None)
        child2 = ref2.x.y
        with pytest.raises(
            AttributeError,
            match="'SomeObject' object has no attribute 'y'",
        ):
            unwrap(child2, SomeObject(x=SomeObject(x=2)))

    def test_get_ref(self):
        ref = Reference(None)
        head = Reference(ref)
        assert get_ref(head) is ref

        child = head["key"]
        assert get_ref(child) is ref

    def test_get_default(self):
        ref = Reference(None)
        assert get_default(ref) == notset

        ref = Reference(None, default=None)
        assert get_default(ref) is None

        ref = Reference(None, default=1)
        assert get_default(ref) == 1

    def test_child_default(self):
        ref = Reference(None)
        child = ref[0]
        assert get_default(child) == notset

        ref = Reference(None, default=1)
        child = ref[0]
        assert get_default(child) == notset

    def test_iter(self):
        ref = Reference(None)
        with pytest.raises(
            NotImplementedError,
            match="Iterating a workflow Reference is not supported.",
        ):
            iter(ref)

    def test_repr(self):
        ref = Reference(None)
        assert repr(ref) == f"Reference(ref=None, default={notset})"

        ref = Reference(None, default=123)
        assert repr(ref) == "Reference(ref=None, default=123)"


def test_add_overwrite():
    a = Reference(1)
    b = Reference(2)
    add_overwrite(a, b)
    assert b._overwrites == []
    assert get_ref(a._overwrites[0]) == 2

    c = Reference(3)
    add_overwrite(a, c)
    assert c._overwrites == []
    assert get_ref(a._overwrites[0]) == 2
    assert get_ref(a._overwrites[1]) == 3

    constant = 5
    add_overwrite(a, constant)
    assert get_ref(a._overwrites[-1]) is None
    assert get_default(a._overwrites[-1]) == 5
    assert constant == 5
