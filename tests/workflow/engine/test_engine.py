from __future__ import annotations

import re
import textwrap
from typing import TYPE_CHECKING, Optional, Union
from unittest.mock import Mock

import pytest

from laboneq_applications.core.options import BaseExperimentOptions
from laboneq_applications.workflow import exceptions, task
from laboneq_applications.workflow.engine import (
    Workflow,
    WorkflowResult,
    for_,
    if_,
    workflow,
)
from laboneq_applications.workflow.engine.options import WorkflowOptions
from laboneq_applications.workflow.task import Task

if TYPE_CHECKING:
    from laboneq_applications.workflow.engine.core import WorkflowBuilder


@task
def addition(x, y) -> float:
    return x + y


@task
def substraction(x, y) -> float:
    return x - y


class TestWorkflowResult:
    def test_add_task(self):
        obj = WorkflowResult()
        assert len(obj.tasks) == 0
        t = Task(addition, output=1)
        obj.add_task(t)
        assert len(obj.tasks) == 1
        assert obj.tasks["addition"] == t


class TestWorkflowResultCollector:
    def test_add_task(self):
        obj = WorkflowResult()
        assert len(obj.tasks) == 0
        t = Task(addition, output=1)
        obj.add_task(t)
        assert len(obj.tasks) == 1
        assert obj.tasks["addition"] == t


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

        assert len(wf().run().tasks["addition", :]) == n_tasks

    def test_independent_tasks(self):
        @workflow
        def wf():
            addition(1, 1)
            substraction(3, 2)

        result = wf().run()
        assert result.tasks["addition"].output == 2
        assert result.tasks["substraction"].output == 1

    def test_context_passing_args(self):
        @workflow
        def wf():
            x = addition(1, 1)
            substraction(3, x)

        result = wf().run()
        assert result.tasks["addition"].output == 2
        assert result.tasks["substraction"].output == 1

    def test_context_passing_kwargs(self):
        @workflow
        def wf():
            y = addition(1, 1)
            substraction(3, y=y)

        result = wf().run()
        assert result.tasks["addition"].output == 2
        assert result.tasks["substraction"].output == 1

    def test_multiple_dependecy(self):
        @workflow
        def wf():
            x = addition(1, 1)
            substraction(3, x)
            addition(x, 5)

        result = wf().run()
        assert result.tasks["addition", 0].output == 2
        assert result.tasks["addition", 1].output == 7
        assert result.tasks["substraction", 0].output == 1

    def test_nested_calls(self):
        @workflow
        def wf():
            substraction(addition(1, addition(0, 1)), addition(1, substraction(2, 1)))

        result = wf().run()
        assert result.tasks["addition", 0].output == 1
        assert result.tasks["addition", 1].output == 2
        assert result.tasks["addition", 2].output == 2
        assert result.tasks["substraction", 0].output == 1
        assert result.tasks["substraction", 1].output == 0

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

        result = wf().run()
        assert result.tasks["return_mapping"].output == {"a": 123, "b": [1, 2]}
        assert result.tasks["act_on_mapping_value"].output == (123, 2)


class TestWorkFlowDecorator:
    @pytest.fixture()
    def builder(self) -> WorkflowBuilder:
        @workflow
        def my_wf(x: int, y: int):
            addition(x, y)

        return my_wf

    def test_call_arguments(self, builder: WorkflowBuilder):
        result = builder(x=1, y=2).run()
        assert result.tasks["addition"].output == 3

        result = builder(1, y=2).run()
        assert result.tasks["addition"].output == 3

        result = builder(1, 2).run()
        assert result.tasks["addition"].output == 3

        # No arguments
        @workflow
        def my_wf1():
            addition(1, 1)

        result = my_wf1().run()
        assert result.tasks["addition"].output == 2

        # Keyword argument with default
        @workflow
        def my_wf2(x: int, y: int = 1):
            addition(x, y)

        result = my_wf2(1).run()
        assert result.tasks["addition"].output == 2

        # Only default arguments
        @workflow
        def my_wf3(x: int = 1, y: int = 1):
            addition(x, y)

        result = my_wf3().run()
        assert result.tasks["addition"].output == 2

        # Invalid arguments
        @workflow
        def my_wf4(x: int): ...

        with pytest.raises(TypeError, match="missing a required argument: 'x'"):
            my_wf4()
        with pytest.raises(TypeError, match="got an unexpected keyword argument 'z'"):
            my_wf4(x=1, z=5)

        # Keyword argument with default overwritten
        @workflow
        def my_wf2(x: int, y: int = 1):
            addition(x, y)

        result = my_wf2(1, 5).run()
        assert result.tasks["addition"].output == 6

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

    def test_output_task_reference(self):
        @workflow
        def my_wf(x: int, y: int):
            return addition(x, y)

        wf = my_wf(1, 1)
        assert wf.run().output == 2

    def test_output_input_reference(self):
        @workflow
        def my_wf(y: int):
            return y

        wf = my_wf(1)
        assert wf.run().output == 1

    def test_output_not_reference(self):
        @workflow
        def my_wf():
            return 5

        wf = my_wf()
        assert wf.run().output == 5

    def test_output_no_return(self):
        @workflow
        def my_wf(): ...

        wf = my_wf()
        assert wf.run().output is None

    def test_output_if_clause(self):
        @workflow
        def my_wf(y):
            x = 1
            with if_(y == 2):
                x = 2
            return x  # noqa: RET504

        wf = my_wf(2)
        assert wf.run().output == 2

        # Due to the dry run, constants within the workflow
        # gets overwritten immediately.
        wf = my_wf(1)
        assert wf.run().output == 2


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
        assert len(res.tasks) == 2
        res = my_wf(2).run()
        assert len(res.tasks) == 1

    def test_nested(self):
        @workflow
        def my_wf(x):
            return_zero()
            with if_(x == 1):
                with if_(x == 1):
                    return_zero()

        res = my_wf(1).run()
        assert len(res.tasks) == 2
        res = my_wf(2).run()
        assert len(res.tasks) == 1

    def test_task_result_input(self):
        @workflow
        def my_wf():
            res = return_zero()
            with if_(res == 0):
                return_zero()
            with if_(res == 2):
                return_zero()

        res = my_wf().run()
        assert len(res.tasks) == 2


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
        assert [x.output for x in res.tasks["return_zero", :]] == [0, 0]
        assert [x.output for x in res.tasks["addition", :]] == [1, 2, 5]

    def test_task_dependency_outside_of_assigned_block_flat_executed(self):
        @workflow
        def my_wf(x):
            with if_(x):
                a = addition(1, 1)
            with if_(x):
                addition(a, 1)

        res = my_wf(x=True).run()
        assert [x.output for x in res.tasks["addition", :]] == [2, 3]

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
    assert res.tasks["addition"].output == 2


class TestForLoopExpression:
    def test_at_root(self):
        @workflow
        def my_wf(x):
            with for_(x) as val:
                addition(1, val)

        res = my_wf([1, 2]).run()
        assert [x.output for x in res.tasks["addition", :]] == [2, 3]

    def test_nested(self):
        @workflow
        def my_wf(x):
            with for_(x) as first:
                with for_(x) as second:
                    addition(first, second)

        res = my_wf([1, 2]).run()
        assert [x.output for x in res.tasks["addition", :]] == [2, 3, 3, 4]

    def test_nested_loop_to_loop(self):
        @workflow
        def my_wf(x):
            with for_(x) as first:
                with for_(first) as second:
                    addition(1, second)

        res = my_wf(x=[[1, 2], [3, 4]]).run()
        assert [x.output for x in res.tasks["addition", :]] == [2, 3, 4, 5]


class WfOptions(WorkflowOptions):
    task_with_opts: BaseExperimentOptions = BaseExperimentOptions()


class TestWorkflowValidOptions:
    @pytest.fixture()
    def task_with_opts(self):
        @task
        def task_with_opts(options: BaseExperimentOptions | None = None):
            return options

        return task_with_opts

    def test_task_indirect_assignment(self, task_with_opts):
        @workflow
        def my_wf(options: WfOptions | None = None):
            task_with_opts()

        opts = WfOptions()
        wf = my_wf(opts)
        result = wf.run()
        assert result.tasks["task_with_opts"].output == opts.task_with_opts

    def test_task_direct_assignment(self, task_with_opts):
        @workflow
        def my_wf(options: WfOptions | None = None):
            task_with_opts(options=options.task_with_opts)

        opts = WfOptions()
        wf = my_wf(opts)
        result = wf.run()
        assert result.tasks["task_with_opts"].output == opts.task_with_opts

    def test_workflow_options_not_provided_use_default(self, task_with_opts):
        @workflow
        def my_wf(options: WfOptions | None = None):
            task_with_opts(options=options)

        wf = my_wf()
        result = wf.run()
        assert result.tasks["task_with_opts"].output == WfOptions().task_with_opts

    def test_workflow_valid_options_invalid_input_type(self, task_with_opts):
        @workflow
        def my_wf(options: WfOptions | None = None):
            task_with_opts(options=options)

        with pytest.raises(
            TypeError,
            match="Workflow input options must be of type '<class 'test_engine.WfOptions'>' or 'None'",  # noqa: E501
        ):
            my_wf(options=123)


class ValidOptions(WorkflowOptions): ...


class TestWorkflowInvalidOptions:
    def test_invalid_options_type(self):
        with pytest.raises(
            TypeError,
            match="Workflow input options must be of type 'WorkflowOptions'",
        ):

            @workflow
            def my_wf_int(options: int | None = None): ...

        with pytest.raises(
            TypeError,
            match="Workflow input options must be of type 'WorkflowOptions'",
        ):

            @workflow
            def my_wf_no_typing(options): ...

        error_msg = (
            "It seems like you want to use the workflow feature of automatically "
            "passing options to the tasks, but the type provided is wrong. "
            "Please use either ValidOptions | None = None, "
            "Optional[ValidOptions] = None or "
            "Union[ValidOptions,None] "
            "to enable this feature. Use any other type if you don't want to use "
            "this feature but still want pass options manually to the workflow "
            "and its tasks."
        )
        with pytest.raises(TypeError, match=error_msg):

            @workflow
            def workflow_a(options: ValidOptions): ...

        with pytest.raises(TypeError, match=error_msg):

            @workflow
            def workflow_b(options: ValidOptions | int): ...

        with pytest.raises(TypeError, match=error_msg):

            @workflow
            def workflow_c(options: ValidOptions | None): ...

        with pytest.raises(TypeError, match=error_msg):

            @workflow
            def workflow_d(options: Union[ValidOptions, None]): ...  # noqa: UP007

        with pytest.raises(TypeError, match=error_msg):

            @workflow
            def workflow_e(options: Optional[ValidOptions]): ...  # noqa: UP007
