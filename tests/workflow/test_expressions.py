from unittest.mock import Mock, call

import pytest

from laboneq_applications.workflow import task
from laboneq_applications.workflow.expressions import ForExpression, IFExpression
from laboneq_applications.workflow.promise import Promise, PromiseResultNotResolvedError


@task
def a_function():
    return 123


class TestIFExpression:
    @pytest.mark.parametrize(
        ("condition", "result"),
        [
            (True, {"a_function": [123]}),
            (False, {}),
            (1, {"a_function": [123]}),
            (0, {}),
        ],
    )
    def test_constant_input(self, condition, result):
        expr = IFExpression(condition)
        with expr:
            a_function()
        log = expr.execute()
        assert log.log == result

    @pytest.mark.parametrize(
        ("condition", "result"),
        [
            (2, {"a_function": [123]}),
            (3, {}),
        ],
    )
    def test_promise_input(self, condition, result):
        promise = Promise()
        expr = IFExpression(promise == 2)
        with expr:
            a_function()
        promise.set_result(condition)
        log = expr.execute()
        assert log.log == result


@task
def addition(x, y):
    return x + y


class TestForLoopExpression:
    def test_execute(self):
        loop = ForExpression([0, 1])
        with loop as x:
            addition(x, 1)
        r = loop.execute()
        assert r.log == {"addition": [1, 2]}

        loop = ForExpression([])
        with loop as x:
            addition(x, 1)
        r = loop.execute()
        assert r.log == {}

    def test_empty_iterable(self):
        loop = ForExpression([])
        with loop as x:
            addition(x, 1)
        r = loop.execute()
        assert r.log == {}

    def test_not_iterable_raises_exception(self):
        loop = ForExpression(2)
        with pytest.raises(TypeError, match="'int' object is not iterable"):
            loop.execute()

    def test_input_promise(self):
        promise = Promise()
        loop = ForExpression(promise)
        with loop as x:
            addition(x, 1)
        promise.set_result([1, 2])
        r = loop.execute()
        assert r.log == {"addition": [2, 3]}

        promise = Promise()
        loop = ForExpression([promise, 5])
        with loop as x:
            addition(x, 1)
        promise.set_result(3)
        r = loop.execute()
        assert r.log == {"addition": [4, 6]}

    def test_input_promise_not_resolved(self):
        # Fail immediately
        promise = Promise()
        loop = ForExpression(promise)
        with pytest.raises(PromiseResultNotResolvedError):
            loop.execute()

        mock_obj = Mock()

        @task
        def addition(x, y):
            return mock_obj(x, y)

        # Fail immediately
        promise = Promise()
        loop = ForExpression([promise, 5])
        with loop as x:
            addition(x, 1)
        with pytest.raises(PromiseResultNotResolvedError):
            loop.execute()
        mock_obj.assert_not_called()
        mock_obj.reset_mock()

        # Fail when unresolved promise encountered
        promise = Promise()
        loop = ForExpression([5, 10, promise, 2])
        with loop as x:
            addition(x, 1)
        with pytest.raises(PromiseResultNotResolvedError):
            loop.execute()
        mock_obj.assert_has_calls([call(5, 1), call(10, 1)])
