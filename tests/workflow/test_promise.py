import pytest

from laboneq_applications.workflow.promise import (
    Promise,
    PromiseResultNotResolvedError,
    ReferencePromise,
)


class TestPromise:
    def test_head_getitem(self):
        head = Promise()
        child = head["key"]
        head.set_result({"key": 123})
        assert child.result() == 123

        head = Promise()
        child = head[0]["key"]
        head.set_result(({"key": 123},))
        assert child.result() == 123

    def test_promise_done_state(self):
        head = Promise()
        child = head["key"]
        assert head.done is False
        assert child.done is False
        head.set_result(1)
        assert head.done is True
        assert child.done is True

    def test_head_not_resolved_error(self):
        head = Promise()
        child = head["key"]
        with pytest.raises(
            PromiseResultNotResolvedError,
            match="Promise result is not resolved.",
        ):
            child.result()

    def test_child_cannot_be_resolved(self):
        head = Promise()
        child = head["key"]
        with pytest.raises(ValueError, match="Cannot resolve child promises"):
            child.set_result({"key": 0})

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
        obj = Promise()
        expr = obj == other
        obj.set_result(one)
        assert expr.result() is result


class Reference:
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "foobar"


class TestReferencePromise:
    def test_ref(self):
        ref = Reference()
        head = ReferencePromise(ref)
        assert head.ref is ref

        child = head["key"]
        assert child.ref is ref

    def test_head_not_resolved_error(self):
        head = ReferencePromise(Reference())
        child = head["key"]
        with pytest.raises(
            PromiseResultNotResolvedError,
            match="Result for 'foobar' is not resolved.",
        ):
            child.result()
