import pytest

from laboneq_applications.workflow import task
from laboneq_applications.workflow.blocks.if_block import IFExpression
from laboneq_applications.workflow.executor import ExecutorState
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
