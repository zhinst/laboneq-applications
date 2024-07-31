import re

import pytest

from laboneq_applications.workflow import task
from laboneq_applications.workflow.engine.block import TaskBlock
from laboneq_applications.workflow.engine.executor import ExecutorState
from laboneq_applications.workflow.engine.expressions import (
    ForExpression,
    IFExpression,
)
from laboneq_applications.workflow.engine.reference import (
    Reference,
)


@task
def a_function():
    return 123


class TestIFExpression:
    @pytest.mark.parametrize(
        ("condition", "result"),
        [
            (True, 1),
            (False, 0),
        ],
    )
    def test_constant_input(self, condition, result):
        expr = IFExpression(condition)
        with expr:
            a_function()
        executor = ExecutorState()
        executor.set_state("condition", condition)
        expr.execute(executor)
        assert len(executor.states) == result + 1

    @pytest.mark.parametrize(
        ("condition", "result"),
        [
            (True, 1),
            (False, 0),
        ],
    )
    def test_reference_input(self, condition, result):
        expr = IFExpression(Reference("condition"))
        with expr:
            a_function()
        executor = ExecutorState()
        executor.set_state("condition", condition)
        expr.execute(executor)
        assert len(executor.states) == result + 1


@task
def addition(x, y):
    return x + y


class TestForLoopExpression:
    def test_execute(self):
        expr = ForExpression([0, 1])
        block = TaskBlock(addition, x=Reference(expr), y=1)
        expr.extend(block)

        executor = ExecutorState()
        expr.execute(executor)
        assert len(executor.states) == 1 + 1
        assert executor.states == {
            expr: 1,
            block: 2,
        }

    def test_empty_iterable(self):
        expr = ForExpression([])
        block = TaskBlock(addition, x=Reference(expr), y=1)
        expr.extend(block)

        executor = ExecutorState()
        expr.execute(executor)
        assert len(executor.states) == 0

    def test_not_iterable_raises_exception(self):
        expr = ForExpression(2)
        executor = ExecutorState()
        with pytest.raises(TypeError, match="'int' object is not iterable"):
            expr.execute(executor)

    def test_input_reference(self):
        expr = ForExpression(Reference("abc"))
        block = TaskBlock(addition, x=Reference(expr), y=1)
        expr.extend(block)

        executor = ExecutorState()
        executor.set_state("abc", [1, 2])
        expr.execute(executor)
        assert executor.states == {
            expr: 2,
            block: 3,
            "abc": [1, 2],
        }

    def test_input_reference_within_container_error(self):
        expr = ForExpression([Reference("abc"), 5])
        block = TaskBlock(addition, x=Reference(expr), y=1)
        expr.extend(block)
        executor = ExecutorState()
        executor.set_state("abc", [1, 2])

        with pytest.raises(
            TypeError,
            match=re.escape("unsupported operand type(s) for +: 'Reference' and 'int'"),
        ):
            expr.execute(executor)
