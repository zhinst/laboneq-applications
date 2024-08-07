from __future__ import annotations

import re
import textwrap
from typing import TYPE_CHECKING, Optional, Union
from unittest.mock import Mock

import pytest

from laboneq_applications.core.options import BaseExperimentOptions, BaseOptions
from laboneq_applications.workflow import exceptions, task
from laboneq_applications.workflow.engine import (
    Workflow,
    WorkflowResult,
    for_,
    if_,
    return_,
    workflow,
)
from laboneq_applications.workflow.options import WorkflowOptions
from laboneq_applications.workflow.task import Task
from laboneq_applications.workflow.taskview import TaskView

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
        [task] = result.tasks
        assert task.name == "addition"
        assert task.output == 2

        # check that the recovered result is removed:
        with pytest.raises(exceptions.WorkflowError) as err:
            wf.recover()
        assert str(err.value) == "Workflow has no result to recover."


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
            " '<class 'laboneq_applications.workflow.options.WorkflowOptions'>'"
            " or 'None'",
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
            assert res.tasks[0] == Task(
                task=task_foo,
                output=1,
                input={"foo": 1, "options": opt.task_foo},
            )
            assert res.tasks[1] == Task(
                task=task_bar,
                output=2,
                input={"bar": 2, "options": opt.task_bar},
            )

        for tb in tbs:
            res = tb().run()
            assert res.tasks[0] == Task(
                task=task_foo,
                output=1,
                input={"foo": 1, "options": opt.task_foo},
            )
            assert res.tasks[1] == Task(
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
        assert res.tasks[0] == Task(
            task=task_foo,
            output=1,
            input={"foo": 1, "options": opt1},
        )
        assert res.tasks[1] == Task(
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
        assert res.tasks[0] == Task(
            task=task_foo,
            output=1,
            input={"foo": 1, "options": opt1},
        )
        assert res.tasks[1] == Task(
            task=task_bar,
            output=2,
            input={"bar": 2, "options": opt2},
        )

    @pytest.mark.xfail(reason="Do we allow to pass options as a position argument?")
    def test_options_passed_as_args(self):
        @workflow(options=OptionFooBar)
        def workflow_a(options: OptionFooBar | None = None):
            task_foo(1)
            task_bar(2)

        opt1 = FooOpt(foo=11)
        opt2 = BarOpt(bar=12)
        opts = OptionFooBar(task_foo=opt1, task_bar=opt2)

        res = workflow_a(opts)
        assert res.tasks == [
            Task(task=task_foo, output=1, input={"foo": 1, "options": opt1}),
            Task(task=task_bar, output=2, input={"bar": 2, "options": opt2}),
        ]

    def test_task_requires_options_but_not_provided(self):
        @workflow
        def workflow_a(options: FooOptWorkFlow | None = None):
            task_foo(1)
            task_bar(2)

        opts = FooOptWorkFlow(task_foo=FooOpt(foo=11))
        res = workflow_a(options=opts).run()
        assert res.tasks[0] == Task(
            task=task_foo,
            output=1,
            input={"foo": 1, "options": FooOpt(foo=11)},
        )
        assert res.tasks[1] == Task(
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
                Task(
                    task=task_foo,
                    output=(1, FooOpt().foo),
                    input={"foo": 1, "options": FooOpt()},
                ),
                Task(
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
                Task(task=task_foo, output=1, input={"foo": 1, "options": None}),
                Task(task=task_bar, output=2, input={"bar": 2, "options": None}),
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
                Task(task=task_fed, output=0, input={"options": 0}),
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
                Task(task=task_foo, output=1, input={"foo": 1, "options": options[0]}),
                Task(task=task_bar, output=2, input={"bar": 2, "options": options[1]}),
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
                Task(task=task_foo, output=1, input={"foo": 1, "options": options[0]}),
                Task(task=task_bar, output=2, input={"bar": 2, "options": options[1]}),
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

        assert res.tasks[0] == Task(
            task=task_foo,
            output=1,
            input={"foo": 1, "options": FooOpt()},
        )
        assert res.tasks[1] == Task(
            task=task_bar,
            output=2,
            input={"bar": 2, "options": BarOpt()},
        )
