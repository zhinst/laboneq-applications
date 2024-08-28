import re

import pytest

from laboneq_applications.workflow import task
from laboneq_applications.workflow.engine.block import TaskBlock
from laboneq_applications.workflow.engine.executor import ExecutorState
from laboneq_applications.workflow.engine.expressions import (
    ForExpression,
    IFExpression,
)
from laboneq_applications.workflow.reference import (
    Reference,
)
from laboneq_applications.workflow.result import WorkflowResult


@task
def a_function():
    return 123


class TestIFExpression:
    @pytest.mark.parametrize(
        ("condition", "states"),
        [
            (True, 1),
            (False, 0),
        ],
    )
    def test_constant_input(self, condition, states):
        expr = IFExpression(condition)
        with expr:
            a_function()
        executor = ExecutorState()
        executor.set_state("condition", condition)
        result = WorkflowResult("test")
        with executor.set_active_workflow_settings(result):
            expr.execute(executor)
        assert len(executor.states) == states + 1
        assert len(result.tasks) == states

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
        with executor.set_active_workflow_settings(WorkflowResult("test")):
            expr.execute(executor)
        assert len(executor.states) == result + 1


@task
def addition(x, y):
    return x + y


class TestForExpression:
    def test_execute(self):
        expr = ForExpression([0, 1])
        block = TaskBlock(addition, x=expr.ref, y=1)
        expr.extend(block)

        executor = ExecutorState()
        result = WorkflowResult("test")
        with executor.set_active_workflow_settings(result):
            expr.execute(executor)
        assert len(executor.states) == 1 + 1
        assert executor.states == {
            expr: 1,
            block: 2,
        }
        assert len(result.tasks) == 2

    def test_empty_iterable(self):
        expr = ForExpression([])
        block = TaskBlock(addition, x=expr.ref, y=1)
        expr.extend(block)

        executor = ExecutorState()
        with executor.set_active_workflow_settings(WorkflowResult("test")):
            expr.execute(executor)
        assert len(executor.states) == 0

    def test_not_iterable_raises_exception(self):
        expr = ForExpression(2)
        executor = ExecutorState()
        with executor.set_active_workflow_settings(WorkflowResult("test")):
            with pytest.raises(TypeError, match="'int' object is not iterable"):
                expr.execute(executor)

    def test_input_reference(self):
        expr = ForExpression(Reference("abc"))
        block = TaskBlock(addition, x=expr.ref, y=1)
        expr.extend(block)

        executor = ExecutorState()
        executor.set_state("abc", [1, 2])
        with executor.set_active_workflow_settings(WorkflowResult("test")):
            expr.execute(executor)
        assert executor.states == {
            expr: 2,
            block: 3,
            "abc": [1, 2],
        }

    def test_input_reference_within_container_error(self):
        expr = ForExpression([Reference("abc"), 5])
        block = TaskBlock(addition, x=expr.ref, y=1)
        expr.extend(block)
        executor = ExecutorState()
        executor.set_state("abc", [1, 2])

        with executor.set_active_workflow_settings(WorkflowResult("test")):
            with pytest.raises(
                TypeError,
                match=re.escape(
                    "unsupported operand type(s) for +: 'Reference' and 'int'"
                ),
            ):
                expr.execute(executor)
