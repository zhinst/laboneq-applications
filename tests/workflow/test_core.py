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
    break_,
    elif_,
    else_,
    exceptions,
    for_,
    if_,
    options,
    return_,
    task,
    workflow,
)
from laboneq_applications.workflow.options import WorkflowOptions
from laboneq_applications.workflow.options_base import BaseOptions
from laboneq_applications.workflow.options_builder import OptionBuilder

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

    def test_run_until_if_to_else(self):
        @workflow
        def wf(x):
            with if_(x == 1):
                addition(1, 1)
            with else_():
                substraction(2, 1)
            substraction(1, 1)

        # IF condition true
        flow = wf(1)
        result = flow.run(until="substraction")
        assert len(result.tasks) == 2
        assert result.tasks[0].output == 2
        assert result.tasks[1].output == 0

        result = flow.resume()
        assert len(result.tasks) == 2
        assert result.tasks[0].output == 2
        assert result.tasks[1].output == 0

        # IF condition false, runs the whole thing
        flow = wf(0)
        result = flow.run(until="addition")
        assert len(result.tasks) == 2
        assert result.tasks[0].output == 1
        assert result.tasks[1].output == 0

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

    def test_state_reset(self):
        @workflow
        def work():
            addition(1, 1)
            task(addition, name="test")(1, 2)

        wf = work()
        assert wf._state is None
        wf.run()
        assert wf._state is None

        wf_until = work()
        wf_until.run(until="addition")
        assert wf_until._state is not None
        wf_until.run()
        assert wf_until._state is None

        @task
        def error():
            raise RuntimeError

        @workflow
        def work_error():
            error()

        wf_err = work_error()
        with pytest.raises(RuntimeError):
            wf_err.run()
        assert wf_err._state is None


class TestTasks:
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

    def test_task_kwargs(self):
        @task
        def a_task(a, **kwargs):
            return a, kwargs

        @workflow
        def wf():
            a_task(1, b=2, c=3)

        result = wf().run()
        assert result.tasks["a_task"].input == {"a": 1, "b": 2, "c": 3}
        assert result.tasks["a_task"].output == (1, {"b": 2, "c": 3})


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

    def test_return_namespace(self):
        @workflow
        def my_wf(a, b):
            return_(a=a, output=b)

        wf = my_wf(1, 2)
        out = wf.run().output
        assert out.a == 1
        assert out.output == 2

    def test_return_namespace_single_kw(self):
        @workflow
        def my_wf(a):
            return_(a=a)

        wf = my_wf(1)
        out = wf.run().output
        assert out.a == 1


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

    def test_chained_if_expression(self):
        @workflow
        def my_wf():
            res = return_zero()
            with if_(res == 0):
                return_zero()
            with if_(res == 2):
                return_zero()

        res = my_wf().run()
        assert len(res.tasks) == 2

    def test_overwrite_variable_simple_branch(self):
        @task
        def return_value(x):
            return x

        @workflow
        def my_wf(x):
            maybe_value = return_value(0)
            with if_(x):
                maybe_value = return_value(1)
            return_(maybe_value)

        result = my_wf(x=True).run()
        assert result.output == 1
        result = my_wf(x=False).run()
        assert result.output == 0

    def test_overwrite_variable_nested_stack(self):
        # Test that variables are resolved on nested stack.
        def define_task_call(x):
            def deep_stack(x):
                @task
                def return_value(x):
                    return x

                return return_value(x)

            return deep_stack(x)

        def define_task():
            def deep_stack():
                @task
                def return_value(x):
                    return x

                return return_value

            return deep_stack()

        @workflow
        def my_wf(x):
            maybe_value = define_task_call(0)
            with if_(x):
                maybe_value = define_task()(1)
                maybe_value = define_task()(2)
            return_(maybe_value)

        result = my_wf(x=True).run()
        assert result.output == 2
        result = my_wf(x=False).run()
        assert result.output == 0

    def test_overwrite_variable_nested_branch(self):
        @task
        def return_value(x):
            return x

        @workflow
        def my_wf(x, y):
            maybe_value = return_value(0)
            with if_(x):
                maybe_value = return_value(1)
                with if_(y):
                    maybe_value = return_value(2)
            return_(maybe_value)

        result = my_wf(x=False, y=False).run()
        assert result.output == 0

        result = my_wf(x=True, y=False).run()
        assert result.output == 1

        result = my_wf(x=True, y=True).run()
        assert result.output == 2

        result = my_wf(x=False, y=True).run()
        assert result.output == 0

    def test_overwrite_variable_workflow_constant(self):
        @task
        def return_value(x):
            return x

        @workflow
        def my_wf(x):
            constant = 0
            with if_(x):
                constant = 123
                constant = return_value(1)
            return_(constant)

        result = my_wf(x=True).run()
        assert result.output == 1

        result = my_wf(x=False).run()
        # Constant always overwrites constants,
        # overwrite works only on references
        assert result.output == 123

    def test_overwrite_variable_nested_workflow(self):
        @task
        def return_value(x):
            return x

        @workflow
        def inner(x):
            constant = return_value(5)
            with if_(x):
                constant = return_value(6)
                with if_(condition=False):
                    constant = return_value(10)
            return_(constant)

        @workflow
        def outer(x):
            constant = return_value(0)
            with if_(x):
                constant = inner(1)
            return_(constant)

        result = outer(x=True).run()
        # Run inner workflow
        assert result.output.output == 6
        # No inner workflow
        result = outer(x=False).run()
        assert result.output == 0

    def test_elif_else(self):
        @task
        def return_value(x):
            return x

        @workflow
        def my_wf(x):
            with if_(x == 1):
                maybe_value = return_value(1)
            with elif_(x == 2):
                maybe_value = return_value(2)
            with elif_(x == 2):  # Should never reach this
                maybe_value = return_value(3)
            with else_():
                maybe_value = return_value(4)
            return_(maybe_value)

        wf = my_wf(1)
        result = wf.run()
        assert len(result.tasks) == 1
        assert result.output == 1

        wf = my_wf(2)
        result = wf.run()
        assert len(result.tasks) == 1
        assert result.output == 2

        wf = my_wf(4)
        result = wf.run()
        assert len(result.tasks) == 1
        assert result.output == 4

    def test_invalid_elif_position(self):
        @task
        def return_value(x):
            return x

        @workflow
        def my_wf(x):
            with elif_(x == 2):
                return_value(2)

        with pytest.raises(
            exceptions.WorkflowError,
            match="An `elif_` expression may only follow an `if_` or an `elif_`",
        ):
            my_wf(1)

    def test_elif_position_after_else(self):
        @task
        def return_value(x):
            return x

        @workflow
        def my_wf():
            with if_(True):
                return_value(2)
            with else_():
                return_value(2)
            with elif_(False):
                return_value(2)

        with pytest.raises(
            exceptions.WorkflowError,
            match="An `elif_` expression may only follow an `if_` or an `elif_`",
        ):
            my_wf()

    def test_invalid_else_position_no_if(self):
        @task
        def return_value(x):
            return x

        @workflow
        def else_top():
            with else_():
                return_value(2)

        with pytest.raises(
            exceptions.WorkflowError,
            match="An `else_` expression may only follow an `if_` or an `elif_`",
        ):
            else_top()

    def test_invalid_else_position_on_else(self):
        @task
        def return_value(x):
            return x

        @workflow
        def else_duplicate():
            with if_(False):
                return_value(None)
            with else_():
                return_value(None)
            with else_():
                return_value(None)

        with pytest.raises(
            exceptions.WorkflowError,
            match="An `else_` expression may only follow an `if_` or an `elif_`",
        ):
            else_duplicate()

    def test_if_else_if(self):
        @task
        def return_value(x):
            return x

        @workflow
        def my_wf(x):
            with if_(x == 1):
                return_value(1)
            with else_():
                return_value(2)
            with if_(x == 1):
                return_value(3)
            with if_(x == 5):
                return_value(5)

        wf = my_wf(1)
        result = wf.run()
        assert len(result.tasks) == 2
        assert result.tasks[0].output == 1
        assert result.tasks[1].output == 3

        wf = my_wf(2)
        result = wf.run()
        assert len(result.tasks) == 1
        assert result.tasks[0].output == 2

        wf = my_wf(5)
        result = wf.run()
        assert len(result.tasks) == 2
        assert result.tasks[0].output == 2
        assert result.tasks[1].output == 5


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

        error_msg = "Result for 'task(name=addition)' is not resolved."
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


class TestBreakLoop:
    def test_break_breaks_loop(self):
        @workflow
        def my_wf(x):
            with for_(x) as _:
                addition(1, 1)  # 2
                with for_(x) as _:
                    addition(1, 1)  # 4
                    break_()
                    addition(1, 1)
                addition(1, 1)  # 2

        res = my_wf([None, None]).run()
        assert len(res.tasks) == 2 + 2 + 4

    def test_break_does_not_carry_over_workflows(self):
        @workflow
        def inner():
            break_()

        @workflow
        def outer(x):
            with for_(x) as _:
                inner()

        with pytest.raises(
            exceptions.WorkflowError,
            match="A `break_` statement may only occur within a `for_` loop",
        ):
            outer([])

    def test_break_outside_of_loop(self):
        @workflow
        def my_wf():
            break_()

        with pytest.raises(
            exceptions.WorkflowError,
            match="A `break_` statement may only occur within a `for_` loop",
        ):
            my_wf()


class TestForExpressionLoopIndex:
    def test_default_index(self):
        @task
        def a_task(): ...

        @workflow
        def single_loop(values):
            with for_(values):
                a_task()

        result = single_loop([1]).run()
        assert result.tasks[0].index == (0,)

        result = single_loop([1, 2, 3]).run()
        assert result.tasks[0].index == (0,)
        assert result.tasks[-1].index == (2,)

        @workflow
        def nested_loop(x, y):
            with for_(x):
                with for_(y):
                    a_task()

        result = nested_loop([1], [1]).run()
        assert result.tasks[0].index == (0, 0)

        result = nested_loop([1, 2], [2, 3, 4]).run()
        assert result.tasks[0].index == (0, 0)
        assert result.tasks[-1].index == (1, 2)

    def test_loop_indexer(self):
        @task
        def a_task(): ...

        @workflow
        def nested_loop(x, y):
            with for_(x, loop_indexer=lambda x: x):
                with for_(y):
                    a_task()

        result = nested_loop([1], [1]).run()
        assert result.tasks[0].index == (1, 0)

        result = nested_loop([1, 123], [2, 3, 4]).run()
        assert result.tasks[0].index == (1, 0)
        assert result.tasks[-1].index == (123, 2)

    def test_nested_workflows(self):
        @task
        def a_task(): ...

        @workflow
        def inner(values):
            with for_(values):
                a_task()

        @workflow
        def outer(outer_vals, inner_vals):
            with for_(outer_vals):
                a_task()
                inner(inner_vals)
                a_task()
            with for_(outer_vals):
                a_task()

        result = outer([1], [2, 3]).run()
        assert result.tasks[0].index == (0,)
        # Only nested workflow gets the parent indexes
        assert result.tasks[1].index == (0,)
        # Nested workflow tasks gets only current workflow indexes
        assert result.tasks[1].tasks[0].index == (0,)
        assert result.tasks[1].tasks[1].index == (1,)
        # Back to root workflow, out of nested index scope
        assert result.tasks[2].index == (0,)
        assert result.tasks[3].index == (0,)


class ValidOptions(WorkflowOptions): ...


class TestWorkflowInvalidOptions:
    def test_workflow_valid_options_invalid_input_type(self):
        @workflow
        def my_wf(options: ValidOptions | None = None): ...

        with pytest.raises(
            TypeError,
            match="Workflow input options must be of type 'ValidOptions', 'dict' or 'None'",  # noqa: E501
        ):
            my_wf(options=123)

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


@options
class FooOpt(BaseOptions):
    foo: int = 1


@options
class BarOpt(BaseOptions):
    bar: int = 2


@options
class OptionFooBar(WorkflowOptions): ...


@options
class OptionFooBarInvalid(WorkflowOptions):
    task_foo: FooOpt = FooOpt()
    task_no_opt: BarOpt = BarOpt()


@options
class OptionNotExisting(OptionFooBar):
    task_not_existing: BarOpt = BarOpt()


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

    def test_run_with_options(self):
        @workflow
        def workflow_a(options: OptionFooBar | None = None):
            task_foo(1)
            task_bar(2)
            task_no_opt(3)

        opts = workflow_a.options()

        res = workflow_a(options=opts).run()
        assert res.tasks[0].task == task_foo
        assert res.tasks[0].output == 1
        assert res.tasks[0].input == {
            "foo": 1,
            "options": opts._base._task_options["task_foo"],
        }

        assert res.tasks[1].task == task_bar
        assert res.tasks[1].output == 2
        assert res.tasks[1].input == {
            "bar": 2,
            "options": opts._base._task_options["task_bar"],
        }

    def test_run_workflow_with_default_options(self):
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
        # default values of options OptionFooBar are used.
        assert res.tasks[0].task == task_foo
        assert res.tasks[0].output == (1, FooOpt().foo)
        assert res.tasks[0].input == {"foo": 1, "options": None}

        assert res.tasks[1].task == task_bar
        assert res.tasks[1].output == (2, BarOpt().bar)
        assert res.tasks[1].input == {"bar": 2, "options": None}

    def test_without_declaring_options(self):
        @workflow
        def workflow_a():
            task_foo(1)
            task_bar(2)

        res = workflow_a().run()
        assert res.tasks[0].task == task_foo
        assert res.tasks[0].output == 1
        assert res.tasks[0].input == {"foo": 1, "options": None}

        assert res.tasks[1].task == task_bar
        assert res.tasks[1].output == 2
        assert res.tasks[1].input == {"bar": 2, "options": None}

        @task
        def task_fed(options=0):
            return options

        @workflow
        def workflow_b():
            task_fed()

        res = workflow_b().run()
        assert res.tasks[0].task == task_fed
        assert res.tasks[0].output == 0
        assert res.tasks[0].input == {"options": 0}

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

        wf_inner_opt = WorkflowOptions()
        wf_inner_opt._task_options = {
            "a_task": FooOpt(foo=20),
            "b_task": FooOpt(foo=30),
        }
        opts = WorkflowOptions()
        opts._task_options = {"a_task": FooOpt(foo=10), "wf_inner": wf_inner_opt}
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

    opts = workflow_b.options()
    wf = workflow_b(options=opts)
    result = wf.run()
    assert len(result.tasks) == 3
    assert result.output == opts._base
    assert result.tasks[0].input == {"options": opts._base._task_options["mytask"]}
    assert result.tasks[2].input == {"options": opts._base._task_options["mytask"]}

    assert result.tasks[1].tasks[0].input == {
        "options": opts._base._task_options["workflow_nested"]._task_options["mytask"]
    }
    assert result.tasks[1].output == opts._base._task_options["workflow_nested"]


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
        base_opt = WorkflowOptions()
        base_opt._task_options = {
            "a_task": TaskOptions(),
            "b_task": BTaskOptions(),
        }
        assert opts == OptionBuilder(base_opt)

        @workflow
        def wf_options_not_provided():
            a_task()
            with if_(None):
                b_task()
            c_task()

        opts = wf_options_not_provided.options()
        base_opt = WorkflowOptions()
        base_opt._task_options = {
            "a_task": TaskOptions(),
            "b_task": BTaskOptions(),
        }
        assert opts == OptionBuilder(base_opt)

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
        inner_wf_opt = InnerWorkflowOptions()
        inner_wf_opt._task_options = {
            "a_task": TaskOptions(),
            "b_task": BTaskOptions(),
        }
        base_opt = WorkflowOptions()
        base_opt._task_options = {
            "inner": inner_wf_opt,
            "a_task": TaskOptions(),
        }
        assert opts == OptionBuilder(base_opt)


@options
class TestOption1(TaskOptions):
    t1: int = 1
    shared: int = 1


@options
class TestOption2(TaskOptions):
    t2: int = 2
    shared: int = 2


@options
class InnerOptions(WorkflowOptions):
    inner: int = 3


@options
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
    @pytest.mark.parametrize(
        "call_func", [lambda wf, opts: wf(opts), lambda wf, opts: wf(options=opts)]
    )
    def test_run_with_right_options(self, call_func):
        # Test builder as args and kwargs
        opt = outer_workflow.options()
        opt.t1(123)
        opt.shared(321)
        wf = call_func(outer_workflow, opt)
        res = wf.run()
        inner_opt = InnerOptions()
        inner_opt._task_options = {"task1": TestOption1(t1=123, shared=321)}
        assert res.tasks[0].input == {"options": inner_opt}
        assert res.tasks[1].input == {"options": TestOption2(t2=2, shared=321)}


class TestDisplayGraph:
    def test_display_graph(self):
        @task
        def a_task(): ...

        @workflow
        def inner():
            a_task()
            with for_([]):
                a_task()
            a_task()

        @workflow
        def outer():
            a_task()
            inner()
            a_task()

        wf = outer()
        assert (
            str(wf.graph.tree)
            == """\
workflow(name=outer)
├─ task(name=a_task)
├─ workflow(name=inner)
│  ├─ task(name=a_task)
│  ├─ for_()
│  │  └─ task(name=a_task)
│  └─ task(name=a_task)
└─ task(name=a_task)\
"""
        )

    def test_display_graph_chained_if(self):
        @task
        def a_task(): ...

        @workflow
        def outer():
            a_task()
            with if_(False):
                a_task()
            with elif_(True):
                a_task()
            with else_():
                a_task()
            a_task()

        wf = outer()
        assert (
            str(wf.graph.tree)
            == """\
workflow(name=outer)
├─ task(name=a_task)
├─ conditional
│  ├─ if_()
│  │  └─ task(name=a_task)
│  ├─ elif_()
│  │  └─ task(name=a_task)
│  └─ else_()
│     └─ task(name=a_task)
└─ task(name=a_task)\
"""
        )
