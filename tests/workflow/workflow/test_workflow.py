from __future__ import annotations

import re
import textwrap
from unittest.mock import Mock

import pytest

from laboneq_applications.workflow import exceptions, task
from laboneq_applications.workflow.engine import (
    Workflow,
    for_,
    if_,
    workflow,
)
from laboneq_applications.workflow.engine.workflow import WorkflowBuilder


@task
def addition(x, y) -> float:
    return x + y


@task
def substraction(x, y) -> float:
    return x - y


class TestTaskFunctionKeepsNormalBehaviour:
    def test_task_no_context(self):
        assert addition(1, 2) == 3
        assert substraction(1, 2) == -1

    def test_task_within_task(self):
        @task
        def substraction(x, y) -> float:
            return x - y

        @task
        def addition(x, y) -> float:
            assert substraction(x, y) == -1

        with Workflow() as wf:
            addition(1, 2)
        wf.run()


def test_workflow_single():
    with Workflow() as wf:
        addition(1, 2)
    result = wf.run()
    assert result.tasklog["addition"][0] == 3


class TestWorkFlowInput:
    def test_valid_input(self):
        with Workflow() as wf:
            addition(wf.input["add_me"], y=wf.input["add_me"])
        result = wf.run(add_me=5)
        assert result.tasklog["addition"][0] == 10

        with Workflow() as wf:
            addition(wf.input["first"], wf.input["second"])

        result = wf.run(first=5, second=2)
        assert result.tasklog["addition"] == [7]

    def test_invalid_input(self):
        with Workflow() as wf:
            addition(wf.input["inp"]["add_me"], y=1)

        with pytest.raises(
            TypeError,
            match=re.escape(
                "Workflow missing input parameter(s): inp",
            ),
        ):
            wf.run()

        with pytest.raises(
            TypeError,
            match=re.escape(
                "Workflow got undefined input parameter(s): a",
            ),
        ):
            wf.run(a={})

    def test_mapping_input(self):
        @task
        def work_on_mapping(result):
            return result["sum"] + 5 + result["value"]

        with Workflow() as wf:
            work_on_mapping(wf.input["input"])

        result = wf.run(input={"sum": 5, "value": 1.2})
        assert result.tasklog["work_on_mapping"] == [5 + 5 + 1.2]


class TestWorkFlowRerunNoCache:
    def test_workflow_rerun_input_change(self):
        with Workflow() as wf:
            addition(1, wf.input["input"]["add_me"])
        result = wf.run(input={"add_me": 5})
        assert result.tasklog["addition"][0] == 6
        result = wf.run(input={"add_me": 6})
        assert result.tasklog["addition"][0] == 7

    def test_task_graph_up_executed_on_input_change(self):
        with Workflow() as wf:
            x = addition(1, wf.input["input"]["add_me"])
            addition(1, x)

        result = wf.run(input={"add_me": 1})
        assert result.tasklog["addition"] == [2, 3]
        result = wf.run(input={"add_me": 2})
        assert result.tasklog["addition"] == [3, 4]

    def test_task_graph_down_executed_on_input_change(self):
        with Workflow() as wf:
            x = addition(1, 1)
            addition(x, wf.input["input"]["add_me"])

        result = wf.run(input={"add_me": 1})
        assert result.tasklog["addition"] == [2, 3]
        result = wf.run(input={"add_me": 2})
        assert result.tasklog["addition"] == [2, 4]

    def test_rerun(self):
        mock_obj = Mock()

        @task
        def create_subject():
            return mock_obj()

        with Workflow() as wf:
            create_subject()

        wf.run()
        wf.run()
        wf.run()
        assert mock_obj.call_count == 3


class TestMultipleTasks:
    def test_each_call_produces_task(self):
        n_tasks = 10
        with Workflow() as wf:
            for _ in range(n_tasks):
                addition(1, 1)
        assert len(wf.run().tasklog["addition"]) == n_tasks

    def test_independent_tasks(self):
        with Workflow() as wf:
            addition(1, 1)
            substraction(3, 2)
        result = wf.run()
        assert result.tasklog["addition"] == [2]
        assert result.tasklog["substraction"] == [1]

    def test_context_passing_args(self):
        with Workflow() as wf:
            x = addition(1, 1)
            substraction(3, x)
        result = wf.run()
        assert result.tasklog["addition"] == [2]
        assert result.tasklog["substraction"] == [1]

    def test_context_passing_kwargs(self):
        with Workflow() as wf:
            y = addition(1, 1)
            substraction(3, y=y)
        result = wf.run()
        assert result.tasklog["addition"] == [2]
        assert result.tasklog["substraction"] == [1]

    def test_multiple_dependecy(self):
        with Workflow() as wf:
            x = addition(1, 1)
            substraction(3, x)
            addition(x, 5)
        result = wf.run()
        assert result.tasklog["addition"] == [2, 7]
        assert result.tasklog["substraction"] == [1]

    def test_nested_calls(self):
        with Workflow() as wf:
            substraction(addition(1, addition(0, 1)), addition(1, substraction(2, 1)))
        result = wf.run()
        assert result.tasklog["addition"] == [1, 2, 2]
        assert result.tasklog["substraction"] == [1, 0]

    def test_single_call_task_multiple_child_tasks(self):
        mock_obj = Mock()

        @task
        def create_subject():
            return mock_obj()

        def foobar(_): ...

        with Workflow() as wf:
            res = create_subject()
            foobar(res)
            foobar(res["a"])
            foobar(res["a"][0])

        wf.run()
        assert mock_obj.call_count == 1


class TestNestedWorkflows:
    def test_nested_workflow_definition(self):
        with Workflow():
            addition(1, 1)
            with pytest.raises(exceptions.WorkflowError) as err:
                with Workflow():
                    addition(1, 1)
        assert str(err.value) == "Nesting Workflows is not allowed."

    def test_executing_workflow_within_workflow(self):
        with Workflow() as wf1:
            addition(1, 2)

        with Workflow():
            addition(1, 1)
            with pytest.raises(exceptions.WorkflowError) as err:
                wf1.run()

        assert str(err.value) == "Nesting Workflows is not allowed."


class TestWorkflowPromises:
    def test_task_promise_getattr(self):
        @task
        def return_mapping():
            return {"a": 123, "b": [1, 2]}

        @task
        def act_on_mapping_value(a, b):
            return a, b

        with Workflow() as wf:
            result = return_mapping()
            act_on_mapping_value(result["a"], result["b"][1])
        assert wf.run().tasklog == {
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

    def test_call(self, builder: WorkflowBuilder):
        result = builder(x=1, y=2)
        assert result.tasklog == {"addition": [3]}

    def test_create(self, builder: WorkflowBuilder):
        assert isinstance(builder.create(), Workflow)

        wf1 = builder.create()
        wf2 = builder.create()
        assert wf1.run(x=1, y=2).tasklog == {"addition": [3]}
        assert wf1.run(x=1, y=3).tasklog == {"addition": [4]}
        assert wf2.run(x=1, y=4).tasklog == {"addition": [5]}
        assert wf2.run(x=1, y=5).tasklog == {"addition": [6]}

    def test_append_subtask(self, builder: WorkflowBuilder):
        # NOTE: Interesting feature
        wf = builder.create()

        @task
        def added_function():
            return 123

        with wf:
            added_function()
            added_function()

        result = wf.run(x=5, y=5)
        assert result.tasklog == {
            "addition": [
                10,
            ],
            "added_function": [
                123,
                123,
            ],
        }

        result = wf.run(x=5, y=5)
        assert result.tasklog == {
            "addition": [
                10,
            ],
            "added_function": [
                123,
                123,
            ],
        }

    def test_builder_has_wrapped_function_docstring(self):
        @workflow
        def my_wf(x: int, y: int):
            "HELLO TEST"
            addition(x, y)

        assert my_wf.__doc__ == (
            "HELLO TEST"
            "\n\nThis function is a `WorkflowBuilder` and has additional"
            " functionality described in the `WorkflowBuilder` documentation."
        )

        @workflow
        def my_wf(x: int, y: int):
            addition(x, y)

        assert my_wf.__doc__ == WorkflowBuilder.__doc__

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

        wf = my_wf.create()
        res = wf.run(x=1)
        assert res.tasklog == {"return_zero": [0, 0]}
        res = wf.run(x=2)
        assert res.tasklog == {"return_zero": [0]}

    def test_nested(self):
        @workflow
        def my_wf(x):
            return_zero()
            with if_(x == 1):
                with if_(x == 1):
                    return_zero()

        wf = my_wf.create()
        res = wf.run(x=1)
        assert res.tasklog == {"return_zero": [0, 0]}
        res = wf.run(x=2)
        assert res.tasklog == {"return_zero": [0]}

    def test_task_result_input(self):
        @workflow
        def my_wf():
            res = return_zero()
            with if_(res == 0):
                return_zero()
            with if_(res == 2):
                return_zero()

        wf = my_wf.create()
        res = wf.run()
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

        wf = my_wf.create()
        res = wf.run(x=True)
        assert res.tasklog == {"return_zero": [0, 0], "addition": [1, 2, 5]}

    def test_task_dependency_outside_of_assigned_block_flat_executed(self):
        @workflow
        def my_wf(x):
            with if_(x):
                a = addition(1, 1)
            with if_(x):
                addition(a, 1)

        wf = my_wf.create()
        res = wf.run(x=True)
        assert res.tasklog == {"addition": [2, 3]}

    def test_task_dependency_outside_of_assigned_block_conditional_not_executed(self):
        @workflow
        def my_wf(x):
            with if_(x == 0):
                a = addition(1, 1)
            with if_(x == 1):
                addition(a, 1)

        wf = my_wf.create()
        with pytest.raises(
            exceptions.WorkflowError,
            match=re.escape("Result for 'Task(name=addition)' is not resolved."),
        ):
            wf.run(x=1)


def test_constant_defined_in_workflow():
    @workflow
    def my_wf(x):
        res = 1
        with if_(x):
            addition(res, 1)

    wf = my_wf.create()
    res = wf.run(x=True)
    assert res.tasklog == {"addition": [2]}


class TestForLoopExpression:
    def test_at_root(self):
        @workflow
        def my_wf(x):
            with for_(x) as val:
                addition(1, val)

        wf = my_wf.create()
        res = wf.run(x=[1, 2])
        assert res.tasklog == {"addition": [2, 3]}

    def test_nested(self):
        @workflow
        def my_wf(x):
            with for_(x) as first:
                with for_(x) as second:
                    addition(first, second)

        wf = my_wf.create()
        res = wf.run(x=[1, 2])
        assert res.tasklog == {"addition": [2, 3, 3, 4]}

    def test_nested_loop_to_loop(self):
        @workflow
        def my_wf(x):
            with for_(x) as first:
                with for_(first) as second:
                    addition(1, second)

        wf = my_wf.create()
        res = wf.run(x=[[1, 2], [3, 4]])
        assert res.tasklog == {"addition": [2, 3, 4, 5]}
