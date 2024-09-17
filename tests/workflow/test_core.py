from __future__ import annotations

import contextlib
import re
import textwrap
from typing import TYPE_CHECKING, Optional, Union
from unittest.mock import Mock

import pytest

from laboneq_applications.workflow import (
    TaskOptions,
    Workflow,
    WorkflowResult,
    exceptions,
    task,
    workflow,
)
from laboneq_applications.workflow.blocks.for_block import for_
from laboneq_applications.workflow.blocks.if_block import if_
from laboneq_applications.workflow.blocks.return_block import return_
from laboneq_applications.workflow.options import WorkflowOptions
from laboneq_applications.workflow.options_base import BaseOptions
from laboneq_applications.workflow.options_builder import OptionBuilder
from laboneq_applications.workflow.result import TaskResult
from laboneq_applications.workflow.taskview import TaskView

if TYPE_CHECKING:
    from laboneq_applications.workflow.core import WorkflowBuilder


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
    def test_name(self):
        wf = Workflow.from_callable(addition, name="test", input={"x": 1, "y": 2})
        assert wf.name == "test"

        wf = Workflow.from_callable(addition, input={"x": 1, "y": 2})
        assert wf.name == "addition"

    def test_input(self):
        wf = Workflow.from_callable(lambda: addition(1, 2))
        assert wf.input == {}

        wf = Workflow.from_callable(lambda x, y: addition(x, y), input={"x": 1, "y": 2})
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

    def test_recover(self):
        @workflow
        def wf():
            addition(1, 1)
            addition(None, 2)

        with pytest.raises(exceptions.WorkflowError) as err:
            wf.recover()
        assert str(err.value) == "Workflow has no result to recover."

        with pytest.raises(TypeError) as err:
            wf().run()

        result = wf.recover()
        task1, task2 = result.tasks
        assert task1.name == "addition"
        assert task1.output == 2

        assert task2.name == "addition"
        assert task2.output is None

        # check that the recovered result is removed:
        with pytest.raises(exceptions.WorkflowError) as err:
            wf.recover()
        assert str(err.value) == "Workflow has no result to recover."

    def test_run_until(self):
        @workflow
        def wf():
            addition(1, 1)
            substraction(1, 1)

        # Test stop at first
        flow = wf()
        result = flow.run(until="addition")
        assert len(result.tasks) == 1
        # Test run to last
        flow = wf()
        result_sub = flow.run(until="substraction")
        assert len(result_sub.tasks) == 2
        # Test run workflow until the end
        result_final = flow.resume()
        assert result_final == result_sub

    def test_run_until_nested_workflow_same_name(self):
        @workflow(name="wf")
        def also_wf():
            return_(1)

        @workflow
        def wf():
            also_wf()
            return_(0)

        flow = wf()
        result1 = flow.run(until="wf")
        assert len(result1.tasks) == 1
        assert result1.tasks["wf"].output == 1
        assert result1.output is None

        result2 = flow.resume()
        assert len(result2.tasks) == 1
        assert result2.tasks[0] == result1.tasks["wf"]
        assert result2.output == 0

    def test_run_until_invalid_input(self):
        @workflow
        def wf():
            addition(1, 1)
            substraction(1, 1)

        # Test stateless workflow
        flow = wf()
        with pytest.raises(
            ValueError, match="Task or workflow 'test' does not exist in the workflow."
        ):
            flow.run(until="test")

        # Test workflow in progress, but invalid task
        flow.run(until="addition")
        with pytest.raises(
            ValueError, match="Task or workflow 'test' does not exist in the workflow."
        ):
            flow.resume(until="test")

        # Test workflow cannot run until itself
        flow = wf()
        with pytest.raises(
            ValueError, match="Task or workflow 'wf' does not exist in the workflow."
        ):
            flow.run(until="wf")

    def test_run_until_nested_workflow(self):
        @task
        def a_task():
            return 0

        @workflow
        def wf_nested():
            return_(1)

        @workflow
        def wf_top():
            wf_nested()
            a_task()

        flow = wf_top()
        result = flow.run(until="wf_nested")
        assert len(result.tasks) == 1
        assert result.tasks["wf_nested"].output == 1

        result = flow.resume()
        assert len(result.tasks) == 2
        assert result.tasks["wf_nested"] == result.tasks["wf_nested"]
        assert result.tasks["a_task"].output == 0

    def test_run_until_if_expression(self):
        @workflow
        def wf(x):
            with if_(x == 1):
                addition(1, 1)
            substraction(1, 1)

        # IF condition true
        flow = wf(1)
        result = flow.run(until="addition")
        assert len(result.tasks) == 1
        result = flow.resume()
        assert len(result.tasks) == 2
        assert result.tasks[0].name == "addition"
        assert result.tasks[1].name == "substraction"

        # IF condition false
        flow = wf(0)
        result = flow.run(until="addition")
        assert len(result.tasks) == 1
        assert result.tasks[0].name == "substraction"

    def test_run_until_for_loop(self):
        @workflow
        def wf():
            with for_([1, 1]) as y:
                addition(y, 1)

        flow = wf()
        result = flow.run(until="addition")
        assert len(result.tasks) == 2

    def test_in_progress_state_reset_when_exception(self):
        @task
        def raise_error():
            raise RuntimeError("test")

        @workflow
        def wf():
            addition(1, 1)
            raise_error()

        flow = wf()
        with contextlib.suppress(RuntimeError):
            flow.run()
        assert flow._state is None

        with contextlib.suppress(RuntimeError):
            flow.run(until="raise_error")
        assert flow._state is None

    def test_reset_happens_on_run(self):
        mock_obj = Mock()

        @task
        def create_subject():
            return mock_obj()

        @workflow
        def wf():
            create_subject()

        flow = wf()
        flow.run(until="create_subject")
        mock_obj.assert_called_once()
        mock_obj.reset_mock()
        flow.run(until="create_subject")
        mock_obj.assert_called_once()
        mock_obj.reset_mock()
        flow.resume()
        mock_obj.assert_not_called()

    def test_resume_on_unexecuted_workflow(self):
        @workflow
        def wf(): ...

        flow = wf()
        with pytest.raises(
            exceptions.WorkflowError, match="Workflow is not in progress."
        ):
            flow.resume()

        flow = wf()
        flow.run()
        with pytest.raises(
            exceptions.WorkflowError, match="Workflow is not in progress."
        ):
            flow.resume()

    def test_result_input(self):
        @workflow
        def work(x, y, options: WorkflowOptions | None = None): ...

        wf = work(x=1, y=4)
        result = wf.run()
        assert result.input == {"options": WorkflowOptions(), "x": 1, "y": 4}

        class OptsTest(WorkflowOptions):
            foobar: int = 5

        opts = OptsTest()
        wf = work(x=1, y=4, options=opts)
        result = wf.run()
        assert result.input == {"options": opts, "x": 1, "y": 4}

    def test_result_has_correct_name(self):
        @workflow
        def noname2(): ...

        @workflow
        def noname():
            noname2()

        wf = noname()
        result = wf.run()
        assert result.name == "noname"
        assert result.tasks[0].name == "noname2"

        @workflow(name="test2")
        def a_name2(): ...

        @workflow(name="test")
        def a_name():
            a_name2()

        wf = a_name()
        result = wf.run()
        assert result.name == "test"
        assert result.tasks[0].name == "test2"


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

    def test_multiple_dependency(self):
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
    def test_workflow_definition_inside_workflow(self):
        @workflow
        def outer():
            addition(1, 1)

            @workflow
            def inner():
                addition(1, 1)

            inner()

        with pytest.raises(
            exceptions.WorkflowError,
            match="Defining a workflow inside a workflow is not allowed.",
        ):
            outer()

    def test_call_run_workflow_nested(self):
        @workflow
        def inner():
            addition(1, 1)

        inner_ = inner()

        @workflow
        def outer():
            inner_.run()

        with pytest.raises(
            exceptions.WorkflowError,
            match=re.escape(
                "Calling '.run()' within another workflow is " "not allowed.",
            ),
        ):
            outer()

    def test_task_output(self):
        @task
        def task1(y):
            return y

        @workflow
        def wf1(x: int = 1):
            task1(x)

        @workflow
        def wf2(x, y):
            wf1(x)
            wf1(y)
            wf1()

        wf = wf2(2, 3)
        res = wf.run()
        assert res.tasks[0].tasks[0].output == 2
        assert res.tasks[1].tasks[0].output == 3
        assert res.tasks[2].tasks[0].output == 1

    def test_nested_workflow_task_output(self):
        @task
        def task1():
            return 1

        @workflow
        def wf1():
            task1()

        @workflow
        def wf2():
            output = wf1()
            return_(output.tasks["task1"].output)

        wf = wf2()
        res = wf.run()
        assert res.output == 1

    def test_output(self):
        @task
        def task1(): ...

        @workflow
        def wf1(x: int):
            task1()
            return_(x)

        @workflow
        def wf2(x):
            output = wf1(x)
            return_(output.output)

        wf = wf2(2)
        res = wf.run()
        assert res.output == 2
        # Tested nested workflow result added to tasks
        assert len(res.tasks) == 1
        assert isinstance(res.tasks[0], WorkflowResult)
        assert res.tasks[0].output == 2

        # Test nested workflow tasks
        assert len(res.tasks[0].tasks) == 1
        assert res.tasks[0].tasks[0].name == "task1"

    def test_nested_workflow_runtime_error(self):
        # Test more than one level deep nesting of workflows
        @task
        def task1(): ...

        @task
        def task2():
            raise RuntimeError

        @workflow
        def wf1():
            task1()
            task2()

        @workflow
        def wf2():
            wf1()

        wf = wf2()
        try:
            wf.run()
        except RuntimeError:
            res = wf2.recover()
        assert len(res.tasks) == 1
        assert isinstance(res.tasks[0], WorkflowResult)

        # Test nested workflow tasks
        assert len(res.tasks[0].tasks) == 2
        assert res.tasks[0].tasks[0].name == "task1"
        assert res.tasks[0].tasks[1].name == "task2"

    def test_nested_nested_workflow_result(self):
        @task
        def task1(x):
            return x

        @workflow
        def wf3(x):
            task1(x)

        @workflow
        def wf2(x):
            wf3(x)

        @workflow
        def wf1(x):
            wf2(x)

        wf = wf1(3)
        result = wf.run()
        assert len(result.tasks) == 1
        assert len(result.tasks[0].tasks) == 1
        assert len(result.tasks[0].tasks[0].tasks) == 1
        assert result.tasks[0].tasks[0].tasks[0].output == 3

    def test_nested_workflow_result_inputs(self):
        @task
        def t():
            return 0

        @workflow
        def wf1(x, y, z: int = 5): ...

        @workflow
        def wf2(x, y):
            output_task = t()
            wf1(x, output_task)

        wf = wf2(1, 2)
        res = wf.run()
        # Top level workflow
        assert res.input == {"options": WorkflowOptions(), "x": 1, "y": 2}
        # Nested workflow
        assert res.tasks[1].input == {
            "options": WorkflowOptions(),
            "x": 1,
            "y": 0,
            "z": 5,
        }


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

    def test_decorator_name(self):
        @workflow(name="test")
        def wf_with_name(): ...

        obj = wf_with_name()
        assert obj.name == "test"

        @workflow
        def wf_no_name(): ...

        obj = wf_no_name()
        assert obj.name == "wf_no_name"

    def test_decorator_as_callable(self):
        def work(): ...

        obj = workflow(work, name="test")
        wf = obj()
        assert wf.name == "test"

        obj_partial = workflow(name="test")
        wf = obj_partial(work)
        assert wf().name == "test"

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

        assert my_wf.src == textwrap.dedent(
            """\
            @workflow
            def my_wf(x: int):
                res = 1
                addition(res, 1)
                addition(res, 1)
                with if_(x):
                    addition(res, 1)
                addition(res, 1)
        """,
        )


class TestReturnStatement:
    def test_return_stops_execution(self):
        @workflow
        def my_wf():
            return_(123)
            return_(addition(1, 2))

        wf = my_wf()
        result = wf.run()
        assert len(result.tasks) == 0
        assert wf.run().output == 123

    def test_return_task_reference(self):
        @workflow
        def my_wf(x: int, y: int):
            return_(addition(x, y))

        wf = my_wf(1, 1)
        assert wf.run().output == 2

    def test_return_input_reference(self):
        @workflow
        def my_wf(y: int):
            return_(y)

        wf = my_wf(1)
        assert wf.run().output == 1

    def test_return_not_reference(self):
        @workflow
        def my_wf():
            return_(5)

        wf = my_wf()
        assert wf.run().output == 5

    def test_return_no_return(self):
        @workflow
        def my_wf(): ...

        wf = my_wf()
        assert wf.run().output is None

    def test_return_default_value(self):
        @workflow
        def my_wf():
            return_()

        wf = my_wf()
        assert wf.run().output is None

    def test_return_branching_constant(self):
        @workflow
        def my_wf(y):
            x = 1
            with if_(y == 2):
                x = 2
            return_(x)

        wf = my_wf(2)
        assert wf.run().output == 2

        # Due to the dry run, constants within the workflow
        # gets overwritten immediately.
        wf = my_wf(1)
        assert wf.run().output == 2

    def test_return_if_block(self):
        @workflow
        def my_wf(y):
            with if_(y == 2):
                return_(0)
            return_(1)

        wf = my_wf(2)
        assert wf.run().output == 0

        wf = my_wf(1)
        assert wf.run().output == 1


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
            "Result for 'TaskBlock(task=task(name=addition), "
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


class TestForExpression:
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
    task_with_opts: TaskOptions = TaskOptions()


class TestWorkflowValidOptions:
    @pytest.fixture()
    def task_with_opts(self):
        @task
        def task_with_opts(options: TaskOptions | None = None):
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
        assert result.tasks["task_with_opts"].output == WfOptions()

    def test_workflow_valid_options_invalid_input_type(self, task_with_opts):
        @workflow
        def my_wf(options: WfOptions | None = None):
            task_with_opts(options=options)

        with pytest.raises(
            TypeError,
            match="Workflow input options must be of type 'WfOptions', 'dict' or 'None'",  # noqa: E501
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


class FooOpt(BaseOptions):
    foo: int = 1


class BarOpt(BaseOptions):
    bar: int = 2


class OptionFooBar(WorkflowOptions):
    task_foo: FooOpt = FooOpt()
    task_bar: BarOpt = BarOpt()


class OptionFooBarInvalid(WorkflowOptions):
    task_foo: FooOpt = FooOpt()
    task_no_opt: BarOpt = BarOpt()


class OptionNotExisting(OptionFooBar):
    task_not_existing: BarOpt = BarOpt()


class FooOptWorkFlow(WorkflowOptions):
    task_foo: FooOpt = FooOpt()


@task
def task_no_opt(foo):
    return foo


@task
def task_foo(foo, options: FooOpt | None = None):
    return foo


@task
def task_bar(bar, options: BarOpt | None = None):
    return bar


class TestWorkflowOptions:
    """
    Case 1: If a workflow has no options declared,
        it is users responsibility to handle it

    Case 2: If workflow has options declared, and options is not passed
        The default values of options, declared in the workflow, are used.

    Case 3: Workflow options is declared and options is provided to the workflow
        3.1: If the targeted task does not need options => ignore, raise warning
        3.2: If the targeted task does not exist => ignore, raise warning
        3.3: If the targeted task needs options => pass it in

    Case 4: Workflow options is declared, but got updated inside the workflow.
        Current implementation: options got updated. Use case: options could be
        used conditioned on results of previous tasks.
    """

    def test_run_workflow_with_invalid_options(self):
        @workflow
        def workflow_a(options: WorkflowOptions | None = None):
            task_foo(1)
            task_bar(2)

        with pytest.raises(
            TypeError,
            match="Workflow input options must be of type"
            " 'WorkflowOptions', 'dict' or 'None'",
        ):
            _ = workflow_a(options=1)

    def test_run_with_option_class(self):
        @workflow
        def workflow_a(options: OptionFooBar | None = None):
            task_foo(1)
            task_bar(2)

        @workflow
        def workflow_b(options: Optional[OptionFooBar] = None):  # noqa: UP007
            task_foo(1)
            task_bar(2)

        @workflow
        def workflow_c(options: Union[OptionFooBar, None] = None):  # noqa: UP007
            task_foo(1)
            task_bar(2)

        tbs = [workflow_a, workflow_b, workflow_c]

        opt = OptionFooBar()
        assert isinstance(opt, OptionFooBar)
        assert isinstance(opt.task_foo, FooOpt)
        assert isinstance(opt.task_bar, BarOpt)
        assert opt.task_foo.foo == 1
        assert opt.task_bar.bar == 2

        for tb in tbs:
            res = tb(options=opt).run()
            assert res.tasks[0] == TaskResult(
                task=task_foo,
                output=1,
                input={"foo": 1, "options": opt.task_foo},
            )
            assert res.tasks[1] == TaskResult(
                task=task_bar,
                output=2,
                input={"bar": 2, "options": opt.task_bar},
            )

        for tb in tbs:
            res = tb().run()
            assert res.tasks[0] == TaskResult(
                task=task_foo,
                output=1,
                input={"foo": 1, "options": opt.task_foo},
            )
            assert res.tasks[1] == TaskResult(
                task=task_bar,
                output=2,
                input={"bar": 2, "options": opt.task_bar},
            )

    def test_run_with_options(self):
        # Case 3.3
        @workflow
        def workflow_a(options: OptionFooBar | None = None):
            task_foo(1)
            task_bar(2)
            task_no_opt(3)

        opt1 = FooOpt()
        opt2 = BarOpt()
        opts = OptionFooBar(task_foo=opt1, task_bar=opt2)

        res = workflow_a(options=opts).run()
        assert res.tasks[0] == TaskResult(
            task=task_foo,
            output=1,
            input={"foo": 1, "options": opt1},
        )
        assert res.tasks[1] == TaskResult(
            task=task_bar,
            output=2,
            input={"bar": 2, "options": opt2},
        )

    def test_task_not_existing(self):
        # Case 3.2
        @workflow
        def workflow_a(options: OptionNotExisting | None = None):
            task_foo(1)
            task_bar(2)

        opt1 = FooOpt()
        opt2 = BarOpt()
        opts = OptionNotExisting(task_foo=opt1, task_not_existing=opt2)

        res = workflow_a(options=opts).run()
        assert res.tasks[0] == TaskResult(
            task=task_foo,
            output=1,
            input={"foo": 1, "options": opt1},
        )
        assert res.tasks[1] == TaskResult(
            task=task_bar,
            output=2,
            input={"bar": 2, "options": opt2},
        )

    def test_task_requires_options_but_not_provided(self):
        @workflow
        def workflow_a(options: FooOptWorkFlow | None = None):
            task_foo(1)
            task_bar(2)

        opts = FooOptWorkFlow(task_foo=FooOpt(foo=11))
        res = workflow_a(options=opts).run()
        assert res.tasks[0] == TaskResult(
            task=task_foo,
            output=1,
            input={"foo": 1, "options": FooOpt(foo=11)},
        )
        assert res.tasks[1] == TaskResult(
            task=task_bar,
            output=2,
            input={"bar": 2, "options": None},
        )

    def test_options_is_declared_but_not_provided(self):
        # Case 2
        @task
        def task_foo(foo, options: FooOpt | None = None):
            options = FooOpt() if options is None else options
            return foo, options.foo

        @task
        def task_bar(bar, options: BarOpt | None = None):
            options = BarOpt() if options is None else options
            return bar, options.bar

        @workflow
        def workflow_a(options: OptionFooBar | None = None):
            task_foo(1)
            task_bar(2)

        res = workflow_a().run()
        # default values of options are used.
        assert res.tasks == TaskView(
            [
                TaskResult(
                    task=task_foo,
                    output=(1, FooOpt().foo),
                    input={"foo": 1, "options": FooOpt()},
                ),
                TaskResult(
                    task=task_bar,
                    output=(2, BarOpt().bar),
                    input={"bar": 2, "options": BarOpt()},
                ),
            ],
        )

    def test_without_declaring_options(self):
        # Case 1
        @workflow
        def workflow_a():
            task_foo(1)
            task_bar(2)

        res = workflow_a().run()
        assert res.tasks == TaskView(
            [
                TaskResult(task=task_foo, output=1, input={"foo": 1, "options": None}),
                TaskResult(task=task_bar, output=2, input={"bar": 2, "options": None}),
            ],
        )

        @task
        def task_fed(options=0):
            return options

        @workflow
        def workflow_b():
            task_fed()

        res = workflow_b().run()
        assert res.tasks == TaskView(
            [
                TaskResult(task=task_fed, output=0, input={"options": 0}),
            ],
        )

    @pytest.mark.xfail(reason="Not implemented in workflow yet")
    def test_workflow_manual_handling_options(self):
        @workflow
        def workflow_a(options=None):
            task_foo(1, options[0])
            task_bar(2, options=options[1])

        options = [FooOpt(), BarOpt()]
        res = workflow_a(options)
        assert res.tasks == TaskView(
            [
                TaskResult(
                    task=task_foo, output=1, input={"foo": 1, "options": options[0]}
                ),
                TaskResult(
                    task=task_bar, output=2, input={"bar": 2, "options": options[1]}
                ),
            ],
        )

        @workflow
        def workflow_a(options: list | None = None):
            task_foo(1, options[0])
            task_bar(2, options=options[1])

        options = [FooOpt(), BarOpt()]
        res = workflow_a(options)
        assert res.tasks == TaskView(
            [
                TaskResult(
                    task=task_foo, output=1, input={"foo": 1, "options": options[0]}
                ),
                TaskResult(
                    task=task_bar, output=2, input={"bar": 2, "options": options[1]}
                ),
            ],
        )

    @pytest.mark.xfail(reason="Behaviour not finalized yet")
    def test_mid_update(self):
        # Case 4
        @workflow(options=OptionFooBar)
        def workflow_a(options=None):
            task_foo(1)
            options.task_foo.foo = 1234
            task_bar(2)

        opts = OptionFooBar(task_foo=FooOpt(), task_bar=BarOpt())

        res = workflow_a(options=opts)

        assert res.tasks[0] == TaskResult(
            task=task_foo,
            output=1,
            input={"foo": 1, "options": FooOpt()},
        )
        assert res.tasks[1] == TaskResult(
            task=task_bar,
            output=2,
            input={"bar": 2, "options": BarOpt()},
        )

    def test_options_as_task_options(self):
        @task
        def b_task(options: FooOpt | None = None): ...

        @task
        def a_task(options: FooOpt | None = None): ...

        @workflow
        def wf_inner(options: WorkflowOptions | None = None):
            a_task()
            b_task()

        @workflow
        def wf(options: WorkflowOptions | None = None):
            a_task()
            wf_inner()
            b_task()

        opts = WorkflowOptions(
            task_options={
                "a_task": FooOpt(foo=10),
                "wf_inner": WorkflowOptions(
                    task_options={"a_task": FooOpt(foo=20), "b_task": FooOpt(foo=30)}
                ),
            }
        )
        result = wf(opts).run()
        assert result.tasks["a_task"].input == {"options": FooOpt(foo=10)}
        assert result.tasks["wf_inner"].tasks["a_task"].input == {
            "options": FooOpt(foo=20)
        }
        assert result.tasks["wf_inner"].tasks["b_task"].input == {
            "options": FooOpt(foo=30)
        }
        assert result.tasks["b_task"].input == {"options": None}


class TaskOptionsTest(BaseOptions):
    foo: int = 1


class NestedOptions(WorkflowOptions):
    mytask: TaskOptionsTest = TaskOptionsTest()


class TopLevelOptions(WorkflowOptions):
    mytask: TaskOptionsTest = TaskOptionsTest()
    workflow_nested: NestedOptions = NestedOptions()


def test_nested_workflows_options():
    @task
    def mytask(options: TaskOptions | None = None): ...

    @workflow
    def workflow_nested(options: NestedOptions | None = None):
        mytask()
        return_(options)

    @workflow
    def workflow_b(options: TopLevelOptions | None = None):
        mytask()
        workflow_nested()
        mytask()
        return_(options)

    opts = TopLevelOptions(
        workflow_nested=NestedOptions(mytask=TaskOptionsTest(foo=1)),
        mytask=TaskOptionsTest(foo=2),
    )
    wf = workflow_b(options=opts)
    result = wf.run()
    assert len(result.tasks) == 3
    assert result.output == opts
    assert result.tasks[0].input == {"options": opts.mytask}
    assert result.tasks[2].input == {"options": opts.mytask}

    assert result.tasks[1].tasks[0].input == {"options": opts.workflow_nested.mytask}
    assert result.tasks[1].output == opts.workflow_nested


class BTaskOptions(BaseOptions):
    param: int = 1


class InnerWorkflowOptions(WorkflowOptions):
    param: int = 1


class TestWorkflowGeneratedOptions:
    @pytest.fixture()
    def tasks(self):
        @task
        def a_task(options: TaskOptions | None = None):
            return

        @task
        def b_task(options: BTaskOptions | None = None):
            return

        @task
        def c_task():
            return

        return a_task, b_task, c_task

    def test_tasks_with_options(self, tasks):
        a_task, b_task, c_task = tasks

        @workflow
        def wf_options_provided(options: WorkflowOptions | None = None):
            a_task()
            with if_(None):
                b_task()
            c_task()

        opts = wf_options_provided.options()
        assert opts == OptionBuilder(
            WorkflowOptions(
                task_options={
                    "a_task": TaskOptionsTest(),
                    "b_task": BTaskOptions(),
                }
            )
        )

        @workflow
        def wf_options_not_provided():
            a_task()
            with if_(None):
                b_task()
            c_task()

        opts = wf_options_not_provided.options()
        assert opts == OptionBuilder(
            WorkflowOptions(
                task_options={
                    "a_task": TaskOptionsTest(),
                    "b_task": BTaskOptions(),
                }
            )
        )

    def test_nested_workflows(self, tasks):
        a_task, b_task, _ = tasks

        @workflow
        def inner(options: InnerWorkflowOptions | None = None):
            b_task()
            a_task()

        @workflow
        def outer(options: WorkflowOptions | None = None):
            inner()
            a_task()

        opts = outer.options()

        assert opts == OptionBuilder(
            WorkflowOptions(
                task_options={
                    "inner": InnerWorkflowOptions(
                        task_options={
                            "a_task": TaskOptionsTest(),
                            "b_task": BTaskOptions(),
                        }
                    ),
                    "a_task": TaskOptionsTest(),
                }
            )
        )


class TestOption1(TaskOptions):
    t1: int = 1
    shared: int = 1


class TestOption2(TaskOptions):
    t2: int = 2
    shared: int = 2


class InnerOptions(WorkflowOptions):
    inner: int = 3


class OuterWorkflowOptions(WorkflowOptions):
    outer: int = 4
    shared: int = 4


@task
def task1(options: TestOption1 | None = None): ...


@task
def task2(options: TestOption2 | None = None): ...


@workflow
def inner_workflow(options: InnerOptions | None = None):
    task1()


@workflow
def outer_workflow(options: OuterWorkflowOptions | None = None):
    inner_workflow()
    task2()


class TestWorkFlowWithOptions:
    def test_run_with_right_options(self):
        opt = outer_workflow.options()
        opt.t1(123)
        opt.shared(321)
        wf = outer_workflow(options=opt)
        res = wf.run()
        assert res.tasks[0].input == {
            "options": InnerOptions(
                task_options={"task1": TestOption1(t1=1, shared=321)}
            )
        }
        assert res.tasks[1].input == {"options": TestOption2(t2=2, shared=321)}
