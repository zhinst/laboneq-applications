import pytest

from laboneq_applications.workflow.reference import Reference, get_default, notset


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

    def test_getattr(self):
        class SomeObject:
            def __init__(self, x) -> None:
                self.x = x

        ref = Reference(None)
        head = ref.x
        assert head.unwrap(SomeObject(x=1)) == 1
        child = head.x
        assert child.unwrap(SomeObject(x=SomeObject(x=2))) == 2

        ref1 = Reference(None)
        head1 = ref1.y
        with pytest.raises(
            AttributeError,
            match="'SomeObject' object has no attribute 'y'",
        ):
            head1.unwrap(SomeObject(x=1))

        ref2 = Reference(None)
        child2 = ref2.x.y
        with pytest.raises(
            AttributeError,
            match="'SomeObject' object has no attribute 'y'",
        ):
            child2.unwrap(SomeObject(x=SomeObject(x=2)))

    def test_ref(self):
        ref = Reference(None)
        head = Reference(ref)
        assert head.ref is ref

        child = head["key"]
        assert child.ref is ref

    def test_default(self):
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
