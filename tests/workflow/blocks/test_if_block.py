from __future__ import annotations

import pytest

from laboneq_applications.workflow import task
from laboneq_applications.workflow.blocks.if_block import (
    ConditionalChain,
    ElseExpression,
    ElseIfExpression,
    IFExpression,
    elif_,
    else_,
    if_,
)
from laboneq_applications.workflow.executor import ExecutorState
from laboneq_applications.workflow.reference import (
    Reference,
)
from laboneq_applications.workflow.result import WorkflowResult


def test_if_without_context():
    with if_(True) as f:
        assert f is None


def test_elif_without_context():
    with elif_(True) as f:
        assert f is None


def test_else_without_context():
    with else_() as f:
        assert f is None


@task
def a_function(): ...


@task
def b_function(): ...


@task
def c_function(): ...


def execute(block: object, variables: dict | None = None) -> WorkflowResult:
    """Helper function for execution."""
    executor = ExecutorState()
    result = WorkflowResult("test")
    for k, v in (variables or {}).items():
        executor.set_variable(k, v)
    with executor.enter_workflow(result):
        block.execute(executor)
    return result


class TestIFExpression:
    def test_str(self):
        block = IFExpression(False)
        assert str(block) == "if_()"

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
        result = execute(expr, {"condition": condition})
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
        executor.set_variable(expr.parameters["condition"], condition)
        with executor.enter_workflow(WorkflowResult("test")):
            expr.execute(executor)
        assert len(executor.block_variables) == result + 1


class TestConditionalChain:
    @pytest.mark.parametrize(
        ("condition", "task_name"),
        [
            (True, "a_function"),
            (False, "b_function"),
        ],
    )
    def test_else_(self, condition, task_name):
        chain = ConditionalChain()
        with chain:
            with IFExpression(condition):
                a_function()
            with ElseIfExpression(True):
                b_function()
        result = execute(chain)
        assert len(result.tasks) == 1
        assert result.tasks[0].name == task_name

    @pytest.mark.parametrize(
        ("condition_if", "condition_elif", "task_name"),
        [
            (True, True, "a_function"),
            (True, False, "a_function"),
            (False, True, "b_function"),
        ],
    )
    def test_elif_expression(self, condition_if, condition_elif, task_name):
        chain = ConditionalChain()
        with chain:
            with IFExpression(condition=condition_if):
                a_function()
            with ElseIfExpression(condition=condition_elif):
                b_function()
            with ElseIfExpression(condition=condition_elif):
                b_function()
            with ElseExpression():
                c_function()
        result = execute(chain)
        assert len(result.tasks) == 1
        assert result.tasks[0].name == task_name

    def test_elif_until_else(self):
        chain = ConditionalChain()
        with chain:
            with IFExpression(condition=False):
                a_function()
            with ElseIfExpression(condition=False):
                b_function()
            with ElseIfExpression(condition=False):
                b_function()
            with ElseExpression():
                c_function()
        result = execute(chain)
        assert len(result.tasks) == 1
        assert result.tasks[0].name == "c_function"
