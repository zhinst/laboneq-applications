from __future__ import annotations

import re
import textwrap
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from laboneq_applications.workflow import exceptions, task
from laboneq_applications.workflow.engine import (
    Workflow,
    for_,
    if_,
    workflow,
)

if TYPE_CHECKING:
    from laboneq_applications.workflow.engine.core import WorkflowBuilder


@task
def addition(x, y) -> float:
    return x + y


@task
def substraction(x, y) -> float:
    return x - y


def test_task_within_task_normal_behaviour():
    @task
    def substraction(x, y) -> float:
        return x - y

    @task
    def addition(x, y):
        assert substraction(x, y) == -1

    wf = Workflow.from_callable(lambda: addition(1, 2))
    wf.run()


class TestWorkflow:
    def test_input(self):
        wf = Workflow.from_callable(lambda: addition(1, 2))
        assert wf.input == {}

        wf = Workflow.from_callable(lambda x, y: addition(x, y), 1, 2)
        assert wf.input == {"x": 1, "y": 2}

    def test_run_multiple_times(self):
        mock_obj = Mock()

        @task
        def create_subject():
            return mock_obj()

        wf = Workflow.from_callable(lambda: create_subject())
        wf.run()
        wf.run()
        wf.run()
        assert mock_obj.call_count == 3


class TestMultipleTasks:
    def test_each_call_produces_task(self):
        n_tasks = 10

        @workflow
        def wf():
            for _ in range(n_tasks):
                addition(1, 1)

        assert len(wf().run().tasklog["addition"]) == n_tasks

    def test_independent_tasks(self):
        @workflow
        def wf():
            addition(1, 1)
            substraction(3, 2)

        result = wf().run()
        assert result.tasklog["addition"] == [2]
        assert result.tasklog["substraction"] == [1]

    def test_context_passing_args(self):
        @workflow
        def wf():
            x = addition(1, 1)
            substraction(3, x)

        result = wf().run()
        assert result.tasklog["addition"] == [2]
        assert result.tasklog["substraction"] == [1]

    def test_context_passing_kwargs(self):
        @workflow
        def wf():
            y = addition(1, 1)
            substraction(3, y=y)

        result = wf().run()
        assert result.tasklog["addition"] == [2]
        assert result.tasklog["substraction"] == [1]

    def test_multiple_dependecy(self):
        @workflow
        def wf():
            x = addition(1, 1)
            substraction(3, x)
            addition(x, 5)

        result = wf().run()
        assert result.tasklog["addition"] == [2, 7]
        assert result.tasklog["substraction"] == [1]

    def test_nested_calls(self):
        @workflow
        def wf():
            substraction(addition(1, addition(0, 1)), addition(1, substraction(2, 1)))

        result = wf().run()
        assert result.tasklog["addition"] == [1, 2, 2]
        assert result.tasklog["substraction"] == [1, 0]

    def test_single_call_task_multiple_child_tasks(self):
        mock_obj = Mock()

        @task
        def create_subject():
            return mock_obj()

        def foobar(_): ...

        @workflow
        def wf():
            res = create_subject()
            foobar(res)
            foobar(res["a"])
            foobar(res["a"][0])

        wf().run()
        assert mock_obj.call_count == 1


class TestNestedWorkflows:
    def test_nested_workflow_definition(self):
        @workflow
        def outer():
            addition(1, 1)

            @workflow
            def inner():
                addition(1, 1)

            inner()

        with pytest.raises(
            exceptions.WorkflowError,
            match="Nesting Workflows is not allowed.",
        ):
            outer()

    def test_executing_workflow_within_workflow(self):
        @workflow
        def inner():
            addition(1, 1)

        inner_ = inner()

        @workflow
        def outer():
            inner_.run()

        with pytest.raises(
            exceptions.WorkflowError,
            match="Nesting Workflows is not allowed.",
        ):
            outer()


class TestWorkflowReferences:
    def test_task_reference_getattr(self):
        @task
        def return_mapping():
            return {"a": 123, "b": [1, 2]}

        @task
        def act_on_mapping_value(a, b):
            return a, b

        @workflow
        def wf():
            result = return_mapping()
            act_on_mapping_value(result["a"], result["b"][1])

        assert wf().run().tasklog == {
            "return_mapping": [{"a": 123, "b": [1, 2]}],
            "act_on_mapping_value": [(123, 2)],
        }


class TestWorkFlowDecorator:
    @pytest.fixture()
    def builder(self) -> WorkflowBuilder:
        @workflow
        def my_wf(x: int, y: int):
            addition(x, y)

        return my_wf

    def test_call_arguments(self, builder: WorkflowBuilder):
        result = builder(x=1, y=2).run()
        assert result.tasklog == {"addition": [3]}

        result = builder(1, y=2).run()
        assert result.tasklog == {"addition": [3]}

        result = builder(1, 2).run()
        assert result.tasklog == {"addition": [3]}

        # No arguments
        @workflow
        def my_wf1():
            addition(1, 1)

        result = my_wf1().run()
        assert result.tasklog == {"addition": [2]}

        # Keyword argument with default
        @workflow
        def my_wf2(x: int, y: int = 1):
            addition(x, y)

        result = my_wf2(1).run()
        assert result.tasklog == {"addition": [2]}

        # Only default arguments
        @workflow
        def my_wf3(x: int = 1, y: int = 1):
            addition(x, y)

        result = my_wf3().run()
        assert result.tasklog == {"addition": [2]}

        # Invalid arguments
        @workflow
        def my_wf4(x: int): ...

        with pytest.raises(TypeError, match="missing a required argument: 'x'"):
            my_wf4()
        with pytest.raises(TypeError, match="got an unexpected keyword argument 'z'"):
            my_wf4(x=1, z=5)

    def test_builder_has_wrapped_function_docstring(self):
        @workflow
        def my_wf(x: int, y: int):
            "HELLO TEST"
            addition(x, y)

        assert my_wf.__doc__ == ("HELLO TEST")

    def test_src(self):
        @workflow
        def my_wf(x: int):
            res = 1
            addition(res, 1)
            addition(res, 1)
            with if_(x):
                addition(res, 1)
            addition(res, 1)

        assert my_wf.src == textwrap.dedent("""\
            @workflow
            def my_wf(x: int):
                res = 1
                addition(res, 1)
                addition(res, 1)
                with if_(x):
                    addition(res, 1)
                addition(res, 1)
        """)


@task
def return_zero():
    return 0


class TestWorkflowIfExpression:
    def test_at_root(self):
        @workflow
        def my_wf(x):
            return_zero()
            with if_(x == 1):
                return_zero()

        res = my_wf(1).run()
        assert res.tasklog == {"return_zero": [0, 0]}
        res = my_wf(2).run()
        assert res.tasklog == {"return_zero": [0]}

    def test_nested(self):
        @workflow
        def my_wf(x):
            return_zero()
            with if_(x == 1):
                with if_(x == 1):
                    return_zero()

        res = my_wf(1).run()
        assert res.tasklog == {"return_zero": [0, 0]}
        res = my_wf(2).run()
        assert res.tasklog == {"return_zero": [0]}

    def test_task_result_input(self):
        @workflow
        def my_wf():
            res = return_zero()
            with if_(res == 0):
                return_zero()
            with if_(res == 2):
                return_zero()

        res = my_wf().run()
        assert res.tasklog == {"return_zero": [0, 0]}


class TestTaskDependencyOutsideOfBlock:
    def test_task_dependency_outside_of_assigned_block_nested(self):
        @workflow
        def my_wf(x):
            res = return_zero()
            with if_(x):
                addition(res, 1)
                with if_(x):
                    res2 = return_zero()
                    add_res = addition(res2, 2)
                    with if_(x):
                        addition(add_res, 3)

        res = my_wf(x=True).run()
        assert res.tasklog == {"return_zero": [0, 0], "addition": [1, 2, 5]}

    def test_task_dependency_outside_of_assigned_block_flat_executed(self):
        @workflow
        def my_wf(x):
            with if_(x):
                a = addition(1, 1)
            with if_(x):
                addition(a, 1)

        res = my_wf(x=True).run()
        assert res.tasklog == {"addition": [2, 3]}

    def test_task_dependency_outside_of_assigned_block_conditional_not_executed(self):
        @workflow
        def my_wf(x):
            with if_(x == 0):
                a = addition(1, 1)
            with if_(x == 1):
                addition(a, 1)

        error_msg = (
            "Result for 'TaskBlock(task=Task(name=addition), "
            "parameters={'x': 1, 'y': 1})' is not resolved."
        )
        with pytest.raises(
            exceptions.WorkflowError,
            match=re.escape(error_msg),
        ):
            my_wf(1).run()


def test_constant_defined_in_workflow():
    @workflow
    def my_wf(x):
        res = 1
        with if_(x):
            addition(res, 1)

    res = my_wf(x=True).run()
    assert res.tasklog == {"addition": [2]}


class TestForLoopExpression:
    def test_at_root(self):
        @workflow
        def my_wf(x):
            with for_(x) as val:
                addition(1, val)

        res = my_wf([1, 2]).run()
        assert res.tasklog == {"addition": [2, 3]}

    def test_nested(self):
        @workflow
        def my_wf(x):
            with for_(x) as first:
                with for_(x) as second:
                    addition(first, second)

        res = my_wf([1, 2]).run()
        assert res.tasklog == {"addition": [2, 3, 3, 4]}

    def test_nested_loop_to_loop(self):
        @workflow
        def my_wf(x):
            with for_(x) as first:
                with for_(first) as second:
                    addition(1, second)

        res = my_wf(x=[[1, 2], [3, 4]]).run()
        assert res.tasklog == {"addition": [2, 3, 4, 5]}
