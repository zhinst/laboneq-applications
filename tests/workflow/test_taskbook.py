from __future__ import annotations

import inspect
import textwrap

import pytest
from IPython.lib.pretty import pretty
from pydantic import ValidationError

from laboneq_applications.core.options import BaseOptions
from laboneq_applications.workflow.exceptions import WorkflowError
from laboneq_applications.workflow.task import Task, task
from laboneq_applications.workflow.taskbook import (
    TaskBook,
    TaskBookOptions,
    TasksView,
    taskbook,
)


@task
def task_a():
    return 1


@task
def task_b():
    task_a()
    return 2


class TestTaskbook:
    def test_init(self):
        book = TaskBook()
        assert book.output is None
        assert book.tasks == TasksView()

    def test_add_entry(self):
        book = TaskBook()
        entry_a = Task(task=task_a, output=1)
        entry_b = Task(task=task_a, output=5)
        book.add_entry(entry_a)
        book.add_entry(entry_b)
        assert book.tasks == [entry_a, entry_b]

    def test_task_cannot_be_attached_to_multiple_taskbooks(self):
        book = TaskBook()
        entry_c = Task(task=task_a, output=1)
        book.add_entry(entry_c)

        book2 = TaskBook()
        with pytest.raises(WorkflowError):
            book2.add_entry(entry_c)

    def test_repr(self):
        book = TaskBook()
        entry_a = Task(task=task_a, output=1)
        entry_b = Task(task=task_a, output=5)
        book.add_entry(entry_a)
        book.add_entry(entry_b)
        tasks_repr = repr(book.tasks)
        assert (
            repr(book) == f"TaskBook(output=None, parameters={{}}, tasks={tasks_repr})"
        )

    def test_str(self):
        book = TaskBook()
        entry_a = Task(task=task_a, output=1)
        entry_b = Task(task=task_a, output=5)
        book.add_entry(entry_a)
        book.add_entry(entry_b)
        assert str(book) == textwrap.dedent("""\
            Taskbook
            Tasks: Task(task_a), Task(task_a)""")

    def test_ipython_pretty(self):
        book = TaskBook()
        entry_a = Task(task=task_a, output=1)
        entry_b = Task(task=task_a, output=5)
        book.add_entry(entry_a)
        book.add_entry(entry_b)
        assert str(book) == textwrap.dedent("""\
            Taskbook
            Tasks: Task(task_a), Task(task_a)""")


class TestTasksView:
    @pytest.fixture()
    def view(self):
        return TasksView(
            [
                Task(task=task_a, output=1),
                Task(task=task_b, output=2),
                Task(task=task_b, output=3),
            ],
        )

    def test_no_copy(self):
        t = Task(task=task_a, output=1)
        tasks = [t]
        view = TasksView(tasks)
        assert view[0] is tasks[0]
        assert view["task_a"] is tasks[0]

    def test_unique(self, view):
        assert view.unique() == {"task_a", "task_b"}

    def test_repr(self, view):
        t = Task(task=task_a, output=1)
        view = TasksView([t])
        assert (
            repr(view)
            == f"[Task(name=task_a, output=1, parameters={{}}, func={task_a.func})]"
        )

    def test_str(self):
        t = Task(task=task_a, output=1)
        view = TasksView([t])
        assert str(view) == "Task(task_a)"

    def test_ipython_pretty(self):
        t = Task(task=task_a, output=1)
        view = TasksView([t])
        assert pretty(view) == "Task(task_a)"

    def test_len(self, view):
        assert len(view) == 3

    def test_getitem(self, view):
        # Test index
        assert view[1] == Task(task=task_b, output=2)
        # Test slice
        assert view[0:2] == [
            Task(task=task_a, output=1),
            Task(task=task_b, output=2),
        ]
        # Test string
        assert view["task_b"] == Task(task=task_b, output=2)
        # Test slice
        assert view["task_b", 0] == Task(task=task_b, output=2)
        assert view["task_b", 1] == Task(task=task_b, output=3)
        assert view["task_b", 0:4] == [
            Task(task=task_b, output=2),
            Task(task=task_b, output=3),
        ]
        assert view["task_b", :] == [
            Task(task=task_b, output=2),
            Task(task=task_b, output=3),
        ]

        with pytest.raises(IndexError):
            view[123]
        with pytest.raises(KeyError):
            view["i do not exist"]
        with pytest.raises(KeyError):
            view["i do not exist", 0]
        with pytest.raises(IndexError):
            view["task_b", 12345]

    def test_eq(self):
        assert TasksView(
            [
                Task(task=task_b, output=2),
                Task(task=task_b, output=3),
            ],
        ) == [
            Task(task=task_b, output=2),
            Task(task=task_b, output=3),
        ]
        assert TasksView() == []
        assert TasksView() != [1, 2]
        assert TasksView() == TasksView()
        assert TasksView([1]) != TasksView([5])


class TestTaskBookDecorator:
    def test_doc(self):
        @taskbook
        def book():
            """foobar"""

        assert book.__doc__ == "foobar"

    def test_parameters(self):
        @taskbook
        def testbook(a, b, options: dict, default=True): ...  # noqa: FBT002

        book = testbook(1, 2, options={"foo": 123})
        assert book.parameters == {
            "a": 1,
            "b": 2,
            "options": {"foo": 123},
            "default": True,
        }

        @taskbook
        def empty_testbook(): ...

        book = empty_testbook()
        assert book.parameters == {}

    def test_task_parameters(self):
        @task
        def myfunc(x: int, foobar=123) -> str:
            return str(x)

        @taskbook
        def testbook():
            myfunc(5)

        book = testbook()
        assert book.tasks[0].parameters == {"x": 5, "foobar": 123}

    def test_signature_matches_function(self):
        def myfunc(x: int) -> str:
            return str(x)

        assert inspect.signature(taskbook(myfunc)) == inspect.signature(myfunc)

    def test_src(self):
        @taskbook
        def book():
            """foobar"""
            return 123

        assert book.src == textwrap.dedent('''\
            @taskbook
            def book():
                """foobar"""
                return 123
        ''')

    def test_func(self):
        def foo(): ...

        tb = taskbook(foo)
        assert tb.func is foo

    def test_result(self):
        @taskbook
        def book():
            return 123

        res = book()
        assert res.output == 123

    def test_task_called_multiple_times(self):
        @taskbook
        def book():
            task_a()
            task_a()

        result = book()
        assert result.tasks == [
            Task(task=task_a, output=1),
            Task(task=task_a, output=1),
        ]

    def test_nested_task_recorded(self):
        @taskbook
        def book():
            task_b()

        result = book()
        assert result.tasks == TasksView(
            [
                Task(task=task_a, output=1),
                Task(task=task_b, output=2),
            ],
        )

    def test_nested_taskbooks(self):
        @taskbook
        def book_a():
            task_b()

        @taskbook
        def book_b():
            book_a()

        with pytest.raises(NotImplementedError, match="Taskbooks cannot be nested."):
            book_b()

    def test_task_return_value_is_returned(self):
        @taskbook
        def book():
            return task_a()

        res = book()
        assert res.output == 1

    def test_run_normal_function_as_task(self):
        def normal_function(x):
            return x + 2

        @taskbook
        def book():
            task(normal_function)(1)
            return normal_function(3)

        res = book()
        assert len(res.tasks) == 1
        assert res.tasks[0].output == 3
        assert res.output == 5

    def test_normal_function_with_task(self):
        @task
        def add_1(x):
            return x + 1

        def sum_2(x):
            return x + 2

        @taskbook
        def book(x, y=None):
            for value in [1, x, y]:
                latest_record = add_1(value)
            return sum_2(latest_record)

        res = book(5, y=10)
        assert [r.output for r in res.tasks] == [2, 6, 11]
        assert res.output == 10 + 1 + 2

    def test_task_rerun(self):
        @task
        def task_a(x, y: int = 1):
            return x + y

        @taskbook
        def book(x):
            task_a(x)

        result = book(1)
        assert result.tasks == [
            Task(task=task_a, output=2, parameters={"x": 1, "y": 1}),
        ]

        # Test update taskbook
        rerun_res = result.tasks[0].rerun(2)
        assert rerun_res == 3
        rerun_res = result.tasks[0].rerun(y=3)
        assert rerun_res == 4
        assert result.tasks == [
            Task(task=task_a, output=2, parameters={"x": 1, "y": 1}),
            Task(task=task_a, output=3, parameters={"x": 2, "y": 1}),
            Task(task=task_a, output=4, parameters={"x": 1, "y": 3}),
        ]
        assert result.tasks["task_a", :] == [
            Task(task=task_a, output=2, parameters={"x": 1, "y": 1}),
            Task(task=task_a, output=3, parameters={"x": 2, "y": 1}),
            Task(task=task_a, output=4, parameters={"x": 1, "y": 3}),
        ]

    def test_partial_results_stored_on_exception(self):
        @task
        def no_error():
            return 1

        def error(x):
            raise RuntimeError

        @taskbook
        def testbook(x):
            no_error()
            error(x)
            return 123

        try:
            testbook(5)
        except RuntimeError:
            book = testbook.recover()
        assert len(book.tasks) == 1
        assert book.output is None

    def test_latest_result_do_not_exists(self):
        @taskbook
        def testbook(): ...

        with pytest.raises(WorkflowError):
            testbook.recover()


class TestTaskBookRunUntil:
    @task
    def task_a():
        return 1

    @task
    def task_b():
        return 2

    @task
    def task_c():
        return 3

    task_a = task_a
    task_b = task_b
    task_c = task_c

    @staticmethod
    def make_run_until_option(task: str) -> dict:
        return TaskBookOptions(run_until=task)

    def test_with_arguments(self):
        @task
        def task_aa(x):
            return x

        @task
        def task_bb(x):
            return x

        @taskbook
        def bookk(a, b, options: TaskBookOptions | None = None):
            task_aa(a)
            task_bb(b)

        result = bookk(1, b=2, options=self.make_run_until_option("task_b"))
        assert result.tasks == [
            Task(task=task_aa, output=1, parameters={"x": 1}),
            Task(task=task_bb, output=2, parameters={"x": 2}),
        ]

    @pytest.fixture()
    def book(self):
        @taskbook
        def book(options: TaskBookOptions | None = None):
            self.task_a()
            self.task_b()
            self.task_c()

        return book

    def test_stop_during_execution(self, book):
        result = book(options=self.make_run_until_option("task_b"))
        assert result.tasks == [
            Task(task=self.task_a, output=1),
            Task(task=self.task_b, output=2),
        ]

    def test_stop_end_of_execution(self, book):
        result = book(options=self.make_run_until_option("task_c"))
        assert result.tasks == [
            Task(task=self.task_a, output=1),
            Task(task=self.task_b, output=2),
            Task(task=self.task_c, output=3),
        ]

    def test_until_task_does_not_exists(self, book):
        result = book(options=self.make_run_until_option("task_not_existing"))
        assert result.tasks == [
            Task(task=self.task_a, output=1),
            Task(task=self.task_b, output=2),
            Task(task=self.task_c, output=3),
        ]

    def test_duplicate_task_name(self):
        @taskbook
        def testbook(options: TaskBookOptions | None = None):
            self.task_a()
            self.task_b()
            self.task_c()
            self.task_b()

        # Exit after first task encounter
        result = testbook(options=self.make_run_until_option("task_b"))
        assert result.tasks == [
            Task(task=self.task_a, output=1),
            Task(task=self.task_b, output=2),
        ]

    def test_result_not_stored(self, book):
        # Ensure result not stored due to internal exception
        book(options=self.make_run_until_option("task_b"))
        with pytest.raises(WorkflowError):
            book.recover()

    def test_task_raises_exception(self):
        @task
        def task_a():
            return 123

        @task
        def task_b():
            raise Exception  # noqa: TRY002

        @taskbook
        def book(options: TaskBookOptions | None = None):
            task_a()
            task_b()

        with pytest.raises(Exception):  # noqa: B017
            book(options=self.make_run_until_option("task_b"))
        assert book.recover().tasks == [
            Task(task=task_a, output=123),
        ]

    def test_task_raises_system_exception(self):
        @task
        def task_a():
            return 123

        @task
        def task_b():
            raise SystemExit

        @taskbook
        def bookkkk(options: TaskBookOptions | None = None):
            task_a()
            task_b()

        with pytest.raises(SystemExit):
            bookkkk(options=self.make_run_until_option("task_b"))
        with pytest.raises(WorkflowError):
            bookkkk.recover()


class FooOpt(BaseOptions):
    foo: int = 1


class BarOpt(BaseOptions):
    bar: int = 2


class OptionFooBar(TaskBookOptions):
    task_foo: FooOpt = FooOpt()
    task_bar: BarOpt = BarOpt()


class OptionFooBarInvalid(TaskBookOptions):
    task_foo: FooOpt = FooOpt()
    task_no_opt: BarOpt = BarOpt()


class OptionNotExisting(OptionFooBar):
    task_not_existing: BarOpt = BarOpt()


class FooOptTaskBook(TaskBookOptions):
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


class TestTaskBookOption:
    """
    Case 1: If a taskbook has no options declared,
        it is users responsibility to handle it

    Case 2: If taskbook has options declared, and options is not passed
        The default values of options, declared in the taskbook, are used.

    Case 3: Taskbook options is declared and options is provided to the taskbook
        3.1: If the targeted task does not need options => ignore, raise warning
        3.2: If the targeted task does not exist => ignore, raise warning
        3.3: If the targeted task needs options => pass it in

    Case 4: Taskbook options is declared, but got updated inside the taskbook.
        Current implementation: options got updated. Use case: options could be
        used conditioned on results of previous tasks.
    """

    def test_create_options(self):
        # taskbook options is declared

        @taskbook
        def taskbook_a(options: OptionFooBar | None = None):
            task_foo(1)
            task_bar(2)

        opts = taskbook_a.options()
        assert isinstance(opts, OptionFooBar)
        assert isinstance(opts.task_foo, FooOpt)
        assert isinstance(opts.task_bar, BarOpt)
        assert opts.task_foo.foo == 1
        assert opts.task_bar.bar == 2

        # options can be updated
        opts.task_foo.foo = 2
        opts.task_bar.bar = 3
        assert opts.task_foo.foo == 2
        assert opts.task_bar.bar == 3

    def test_validate_options(self):
        # Initialize option with non-existing attributes will raises error
        @taskbook
        def taskbook_a(options: TaskBookOptions | None = None):
            task_foo(1)
            task_bar(2)

        with pytest.raises(ValidationError):
            _ = taskbook_a.options(non_existing=1)

    def test_run_taskbook_with_invalid_options(self):
        @taskbook
        def taskbook_a(options: TaskBookOptions | None = None):
            task_foo(1)
            task_bar(2)

        with pytest.raises(
            TypeError,
            match="Options must be a dictionary or an instance of TaskBookOptions.",
        ):
            _ = taskbook_a(options=1)

    def test_create_options_not_declare(self):
        # taskbook options is not declared
        @taskbook
        def taskbook_a(options: TaskBookOptions = None):
            task_foo(1)
            task_bar(2)

        with pytest.raises(AttributeError, match="Taskbook does not have options."):
            taskbook_a.options()

    def test_run_with_option_class(self):
        @taskbook
        def taskbook_a(options: OptionFooBar | None = None):
            task_foo(1)
            task_bar(2)

        opt = taskbook_a.options()
        assert isinstance(opt, OptionFooBar)
        assert isinstance(opt.task_foo, FooOpt)
        assert isinstance(opt.task_bar, BarOpt)
        assert opt.task_foo.foo == 1
        assert opt.task_bar.bar == 2

        res = taskbook_a(options=opt)
        assert res.tasks[0] == Task(
            task=task_foo,
            output=1,
            parameters={"foo": 1, "options": opt.task_foo},
        )
        assert res.tasks[1] == Task(
            task=task_bar,
            output=2,
            parameters={"bar": 2, "options": opt.task_bar},
        )

    def test_run_with_options(self):
        # Case 3.3
        @taskbook
        def taskbook_a(options: OptionFooBar | None = None):
            task_foo(1)
            task_bar(2)
            task_no_opt(3)

        opt1 = FooOpt()
        opt2 = BarOpt()
        opts = taskbook_a.options(task_foo=opt1, task_bar=opt2)

        res = taskbook_a(options=opts)
        assert res.tasks[0] == Task(
            task=task_foo,
            output=1,
            parameters={"foo": 1, "options": opt1},
        )
        assert res.tasks[1] == Task(
            task=task_bar,
            output=2,
            parameters={"bar": 2, "options": opt2},
        )

    def test_run_with_dict_options(self):
        @taskbook
        def taskbook_a(options: OptionFooBar | None = None):
            task_foo(1)
            task_bar(2)

        res = taskbook_a(options={"task_foo": FooOpt(), "task_bar": BarOpt()})
        assert res.tasks[0] == Task(
            task=task_foo,
            output=1,
            parameters={"foo": 1, "options": FooOpt()},
        )
        assert res.tasks[1] == Task(
            task=task_bar,
            output=2,
            parameters={"bar": 2, "options": BarOpt()},
        )

    def test_opts_not_needed(self):
        # Case 3.1
        @taskbook
        def taskbook_a(options: OptionFooBarInvalid | None = None):
            task_foo(1)
            task_no_opt(3)

        opt1 = FooOpt()
        opt2 = BarOpt()
        opts = taskbook_a.options(task_foo=opt1, task_no_opt=opt2)

        with pytest.raises(
            ValueError,
            match=f"Task {task_no_opt.name} does not require options.",
        ):
            taskbook_a(options=opts)

    def test_task_not_existing(self):
        # Case 3.2
        @taskbook
        def taskbook_a(options: OptionNotExisting | None = None):
            task_foo(1)
            task_bar(2)

        opt1 = FooOpt()
        opt2 = BarOpt()
        opts = taskbook_a.options(task_foo=opt1, task_not_existing=opt2)

        res = taskbook_a(options=opts)
        assert res.tasks[0] == Task(
            task=task_foo,
            output=1,
            parameters={"foo": 1, "options": opt1},
        )
        assert res.tasks[1] == Task(
            task=task_bar,
            output=2,
            parameters={"bar": 2, "options": opt2},
        )

    @pytest.mark.xfail(reason="Do we allow to pass options as a position argument?")
    def test_options_passed_as_args(self):
        @taskbook(options=OptionFooBar)
        def taskbook_a(options: OptionFooBar | None = None):
            task_foo(1)
            task_bar(2)

        opt1 = FooOpt(foo=11)
        opt2 = BarOpt(bar=12)
        opts = taskbook_a.options(task_foo=opt1, task_bar=opt2)

        res = taskbook_a(opts)
        assert res.tasks == [
            Task(task=task_foo, output=1, parameters={"foo": 1, "options": opt1}),
            Task(task=task_bar, output=2, parameters={"bar": 2, "options": opt2}),
        ]

    def test_task_requires_options_but_not_provided(self):
        @taskbook
        def taskbook_a(options: FooOptTaskBook | None = None):
            task_foo(1)
            task_bar(2)

        opts = taskbook_a.options(task_foo=FooOpt(foo=11))
        res = taskbook_a(options=opts)
        assert res.tasks[0] == Task(
            task=task_foo,
            output=1,
            parameters={"foo": 1, "options": FooOpt(foo=11)},
        )
        assert res.tasks[1] == Task(
            task=task_bar,
            output=2,
            parameters={"bar": 2, "options": None},
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

        @taskbook
        def taskbook_a(options: OptionFooBar | None = None):
            task_foo(1)
            task_bar(2)

        res = taskbook_a()
        # default values of options are used.
        assert res.tasks == [
            Task(
                task=task_foo,
                output=(1, FooOpt().foo),
                parameters={"foo": 1, "options": FooOpt()},
            ),
            Task(
                task=task_bar,
                output=(2, BarOpt().bar),
                parameters={"bar": 2, "options": BarOpt()},
            ),
        ]

    def test_without_declaring_options(self):
        # Case 1
        @taskbook
        def taskbook_a():
            task_foo(1)
            task_bar(2)

        res = taskbook_a()
        assert res.tasks == [
            Task(task=task_foo, output=1, parameters={"foo": 1, "options": None}),
            Task(task=task_bar, output=2, parameters={"bar": 2, "options": None}),
        ]

        @task
        def task_fed(options=0):
            return options

        @taskbook
        def taskbook_b():
            task_fed()

        res = taskbook_b()
        assert res.tasks == [
            Task(task=task_fed, output=0, parameters={"options": 0}),
        ]

    def test_taskbook_manual_handling_options(self):
        @taskbook
        def taskbook_a(options=None):
            task_foo(1, options[0])
            task_bar(2, options=options[1])

        options = [FooOpt(), BarOpt()]
        res = taskbook_a(options)
        assert res.tasks == [
            Task(task=task_foo, output=1, parameters={"foo": 1, "options": options[0]}),
            Task(task=task_bar, output=2, parameters={"bar": 2, "options": options[1]}),
        ]

        @taskbook
        def taskbook_a(options: list | None = None):
            task_foo(1, options[0])
            task_bar(2, options=options[1])

        options = [FooOpt(), BarOpt()]
        res = taskbook_a(options)
        assert res.tasks == [
            Task(task=task_foo, output=1, parameters={"foo": 1, "options": options[0]}),
            Task(task=task_bar, output=2, parameters={"bar": 2, "options": options[1]}),
        ]

    @pytest.mark.xfail(reason="Behaviour not finalized yet")
    def test_mid_update(self):
        # Case 4
        @taskbook(options=OptionFooBar)
        def taskbook_a(options=None):
            task_foo(1)
            options.task_foo.foo = 1234
            task_bar(2)

        opts = taskbook_a.options(task_foo=FooOpt(), task_bar=BarOpt())

        res = taskbook_a(options=opts)

        assert res.tasks[0] == Task(
            task=task_foo,
            output=1,
            parameters={"foo": 1, "options": FooOpt()},
        )
        assert res.tasks[1] == Task(
            task=task_bar,
            output=2,
            parameters={"bar": 2, "options": BarOpt()},
        )
