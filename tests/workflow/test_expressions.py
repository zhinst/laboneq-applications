import pytest

from laboneq_applications.workflow import task
from laboneq_applications.workflow.expressions import IFExpression
from laboneq_applications.workflow.promise import Promise


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
