import pytest

from laboneq_applications.workflow.engine.reference import (
    Reference,
)


class TestReference:
    def test_getitem(self):
        head = Reference(None)
        child = head["key"]
        assert head.unwrap({"key": 123}) == {"key": 123}

        head = Reference(None)
        child = head[0]["key"]
        assert child.unwrap(({"key": 123},)) == 123

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
        assert expr.unwrap(one) is result

    def test_ref(self):
        ref = Reference(None)
        head = Reference(ref)
        assert head.ref is ref

        child = head["key"]
        assert child.ref is ref
