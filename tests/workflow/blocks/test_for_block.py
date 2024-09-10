import re

import pytest

from laboneq_applications.workflow import task
from laboneq_applications.workflow.blocks.for_block import (
    ForExpression,
)
from laboneq_applications.workflow.blocks.task_block import TaskBlock
from laboneq_applications.workflow.executor import ExecutorState
from laboneq_applications.workflow.reference import (
    Reference,
)
from laboneq_applications.workflow.result import WorkflowResult


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
