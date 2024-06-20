import inspect
import textwrap

import pytest

from laboneq_applications.workflow import task
from laboneq_applications.workflow.exceptions import WorkflowError
from laboneq_applications.workflow.taskbook import (
    Task,
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

    def task_cannot_be_attached_to_multiple_taskbooks(self):
        book = TaskBook()
        entry_c = Task(task=task_a, output=1)
        entry_c._taskbook = book

        with pytest.raises(WorkflowError):
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


class TestTask:
    def test_name(self):
        t = Task(task_a, 2)
        assert t.name == "task_a"

    def test_func(self):
        t = Task(task_a, 2)
        assert t.func == task_a.func

    def test_eq(self):
        e1 = Task(task_a, 2)
        e2 = Task(task_a, 2)
        assert e1 == e2

        e1 = Task(task_a, 2)
        e2 = Task(task_b, 2)
        assert e1 != e2

        e1 = Task(task_a, 2)
        assert e1 != 2
        assert e1 != "bar"

    def test_repr(self):
        t = Task(task_a, 2)
        assert repr(t) == f"Task(name=task_a, output=2, parameters={{}}, func={t.func})"

    def test_str(self):
        t = Task(task_a, 2)
        assert str(t) == "Task(task_a)"

    def test_src(self):
        @task
        def task_():
            return 1

        t = Task(task_, 2)
        assert t.src == textwrap.dedent("""\
            @task
            def task_():
                return 1
        """)


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


@task
def task_alice(bar, options=None):
    return bar


@task
def task_bob(foo, options=None):
    return foo


@task
def task_charles(fred):
    return fred


class TestTaskbookRunWithOptions:
    def test_empty_options(self):
        @task
        def task_print_options(options=None):
            return {"count": 1}

        @taskbook
        def book(options=None):
            task_alice(1)
            task_bob(2)
            task_charles(3)
            task_print_options()

        res = book()
        assert res.tasks == TasksView(
            [
                Task(task=task_alice, output=1, parameters={"bar": 1, "options": None}),
                Task(task=task_bob, output=2, parameters={"foo": 2, "options": None}),
                Task(task=task_charles, output=3, parameters={"fred": 3}),
                Task(
                    task=task_print_options,
                    output={"count": 1},
                    parameters={"options": None},
                ),
            ],
        )

    def test_broadcast_options(self):
        @taskbook
        def book(options=None):
            task_alice(1)
            task_bob(2)
            task_charles(3)

        # broadcast options
        options = {"count": 1}
        res = book(options=options)

        # count is broadcasted to all tasks requiring options
        assert res.tasks == TasksView(
            [
                Task(
                    task=task_alice,
                    output=1,
                    parameters={"bar": 1, "options": {"count": 1}},
                ),
                Task(
                    task=task_bob,
                    output=2,
                    parameters={"foo": 2, "options": {"count": 1}},
                ),
                Task(task=task_charles, output=3, parameters={"fred": 3}),
            ],
        )

    def test_options_not_required(self):
        @taskbook
        def book(options=None):
            task_alice(1)
            task_charles(3)

        options = {"count": 1, "task.task_charles": {"fred": 4}}
        res = book(options=options)

        # task_c does not require options
        assert res.tasks == TasksView(
            [
                Task(
                    task=task_alice,
                    output=1,
                    parameters={"bar": 1, "options": {"count": 1}},
                ),
                Task(task=task_charles, output=3, parameters={"fred": 3}),
            ],
        )

    def test_options_normal_func(self):
        # test that a normal function does not receive options
        def some_func(options=None):
            return options

        @taskbook
        def book(options=None):
            return some_func()

        options = {"foo": 1}
        res = book(options=options)
        assert res.output is None

    def test_override_options(self):
        @taskbook
        def taskbook_a(options=None):
            task_alice(1)
            task_bob(2)
            task_charles(3)

        options = {
            "count": 1,
            "task.task_alice": {"foo": "bar", "count": 3},
            "task.task_bob": {"bar": "foo"},
        }

        res = taskbook_a(options=options)
        assert res.tasks == [
            Task(
                task=task_alice,
                output=1,
                parameters={"bar": 1, "options": {"foo": "bar", "count": 3}},
            ),
            Task(
                task=task_bob,
                output=2,
                parameters={"foo": 2, "options": {"bar": "foo", "count": 1}},
            ),
            Task(task=task_charles, output=3, parameters={"fred": 3}),
        ]

    def test_options_called_explicitly_in_taskbook(self):
        # when options are explicitly called in the taskbook
        # it should be overridden only for keys that exist
        # in the options propagated from the taskbook
        @taskbook
        def taskbook_a(options=None):
            task_alice(1, options={"count": 1, "foo": "bar"})

        options = {
            "count": 2,
            "task.task_alice": {"count": 3},
        }

        res = taskbook_a(options=options)
        assert res.tasks == [
            Task(
                task=task_alice,
                output=1,
                parameters={"bar": 1, "options": {"foo": "bar", "count": 3}},
            ),
        ]

        res = taskbook_a()
        assert res.tasks == [
            Task(
                task=task_alice,
                output=1,
                parameters={"bar": 1, "options": {"foo": "bar", "count": 1}},
            ),
        ]

    def test_task_name_in_options(self):
        # one of the broadcasted options matches
        # to one of the task name
        @taskbook
        def taskbook_a(options=None):
            task_alice(1)
            task_bob(2)
            task_charles(3)

        options = {
            "count": 1,
            "task_alice": 1,
            "task.task_alice": {"foo": "bar", "count": 3},
            "task.task_bob": {"bar": "foo", "task_alice": 2},
        }

        res = taskbook_a(options=options)
        assert res.tasks == [
            Task(
                task=task_alice,
                output=1,
                parameters={
                    "bar": 1,
                    "options": {"foo": "bar", "count": 3, "task_alice": 1},
                },
            ),
            Task(
                task=task_bob,
                output=2,
                parameters={
                    "foo": 2,
                    "options": {"bar": "foo", "count": 1, "task_alice": 2},
                },
            ),
            Task(task=task_charles, output=3, parameters={"fred": 3}),
        ]

    def test_options_with_funky_name(self):
        @task(name="funky.wendy")
        def task_funky(arg, options=None):
            return arg

        @taskbook
        def taskbook_a(options=None):
            task_funky(1)

        options = {
            "count": 1,
            "task.funky.wendy": {"bar": "foo"},
            "foo.bar": 1,  # broadcast option
            "foo.bar.fred": 1,  # broadcast option
            "foo.": 1,  # broadcast option
        }

        res = taskbook_a(options=options)
        assert res.tasks == [
            Task(
                task=task_funky,
                output=1,
                parameters={
                    "arg": 1,
                    "options": {
                        "bar": "foo",
                        "count": 1,
                        "foo.bar.fred": 1,
                        "foo.bar": 1,
                        "foo.": 1,
                    },
                },
            ),
        ]

    def test_options_used_separately(self):
        opts = {"common": 1, "task.my_task": {"a": 5}}

        @taskbook
        def foobar(options=None):
            task_alice(options["task.my_task"])

        res = foobar(options=opts)
        assert res.tasks == [
            Task(
                task=task_alice,
                output={"a": 5},
                parameters={"bar": {"a": 5}, "options": {"common": 1}},
            ),
        ]

    @pytest.mark.xfail(reason="Not implemented")
    def test_update_options_between_calls(self):
        @taskbook
        def taskbook_invalid(options=None):
            task_alice(1)
            options["count"] = "updated"
            task_bob(2)
            task_charles(3)

        options = {
            "count": 1,
            "task.task_alice": {"foo": "bar", "count": 3},
            "task.task_bob": {"bar": "foo", "count": 2},
        }
        with pytest.raises(
            RuntimeError,
            match="Cannot update options during taskbook execution.",
        ):
            taskbook_invalid(options=options)

        # nested dict assignment
        @taskbook
        def taskbook_invalid(options=None):
            task_alice(1)
            options["task.task_alice"]["foo"] = 3
            task_bob(2)
            task_charles(3)

        options = {
            "count": 1,
            "task.task_alice": {"foo": "bar", "count": 3},
            "task.task_bob": {"bar": "foo", "count": 2},
        }
        with pytest.raises(
            RuntimeError,
            match="Cannot update options during taskbook execution.",
        ):
            taskbook_invalid(options=options)

        # test that options can be updated outside of the taskbook
        def taskbook_valid(options=None):
            task_alice(1)
            options["count"] = "updated"
            task_bob(2)
            task_charles(3)
            return options

        res = taskbook_valid(options=options)
        assert res["count"] == "updated"

    def test_priority_options(self):
        """Options are ranked in the following order of priority:
        1. task-specific option
        2. broadcast option
        3. option set in taskbook
        4. default option in definition of tasks
        """

        @task
        def task_alice(options=None):
            if options is None:
                options = {"count": 4}
            return options["count"]

        @taskbook
        def taskbook_a(options=None):
            task_alice(options={"count": 3})

        # test 1>2>3>4
        options = {
            "count": 2,
            "task.task_alice": {"count": 1},
        }
        res = taskbook_a(options=options)
        assert res.tasks == [
            Task(
                task=task_alice,
                output=1,
                parameters={"options": {"count": 1}},
            ),
        ]

        options = {
            "task.task_alice": {"count": 1},
        }
        res = taskbook_a(options=options)
        assert res.tasks == [
            Task(task=task_alice, output=1, parameters={"options": {"count": 1}}),
        ]

        # test 2>3>4
        options = {
            "count": 2,
        }
        res = taskbook_a(options=options)
        assert res.tasks == [
            Task(task=task_alice, output=2, parameters={"options": {"count": 2}}),
        ]

        # test 3>4
        res = taskbook_a()
        assert res.tasks == [
            Task(task=task_alice, output=3, parameters={"options": {"count": 3}}),
        ]

    def test_support_options_identical_tasks(self):
        def task_alice(bar, options=None):
            return bar

        @taskbook
        def taskbook_a(options):
            task(task_alice, name="task_alice_1")(1)
            task(task_alice, name="task_alice_2")(2)

        options = {
            "count": 0,
            "task.task_alice_1": {"foo": "bar", "count": 1},
            "task.task_alice_2": {"bar": "foo", "count": 2},
        }

        res = taskbook_a(options=options)

        assert res.tasks[0].output == 1
        assert res.tasks[0].parameters == {
            "bar": 1,
            "options": {"foo": "bar", "count": 1},
        }

        assert res.tasks[1].output == 2
        assert res.tasks[1].parameters == {
            "bar": 2,
            "options": {"bar": "foo", "count": 2},
        }

    def test_partial_update_default_options(self):
        @task
        def task_alice(bar, options=None):
            default_options = {"foo": 0, "not_supply": 0}
            if options is not None:
                options = default_options | options
            else:
                options = default_options

            return bar, options["foo"], options["not_supply"]

        @taskbook
        def taskbook_a(options=None):
            task_alice(1)

        # only foo is updated, "not_supply" remains default
        options = {"bar": 2, "task.task_alice": {"foo": 2}}
        res = taskbook_a(options=options)

        assert res.tasks[0] == Task(
            task=task_alice,
            output=(1, 2, 0),
            parameters={"bar": 1, "options": {"bar": 2, "foo": 2}},
        )

        # default options used
        res = taskbook_a()
        assert res.tasks[0] == Task(
            task=task_alice,
            output=(1, 0, 0),
            parameters={"bar": 1, "options": None},
        )


class TestTaskBookOptions:
    def test_init(self):
        task_options = {"foo": 1, "task.bar": {"bar1": 1, "bar2": 2}}
        options = TaskBookOptions(**task_options)
        assert options._broadcast == {"foo": 1}
        assert options._specific == {"bar": {"bar1": 1, "bar2": 2}}

    def test_init_empty(self):
        task_options = {}
        options = TaskBookOptions(**task_options)
        assert options._broadcast == {}
        assert options._specific == {}

    def test_process_options(self):
        task_options = {
            "foo": 1,
            "task.bar": {"bar1": 1, "bar2": 2},
            "task.foo": {"foo": "override"},
        }
        options = TaskBookOptions(**task_options)
        res_bar = options.task_options("bar")
        assert res_bar == {"bar1": 1, "bar2": 2, "foo": 1}

        res_foo = options.task_options("foo")
        assert res_foo == {"foo": "override"}

        res_baz = options.task_options("baz")
        assert res_baz == {"foo": 1}

    def test_process_funky_options(self):
        options = {
            "count": 1,
            "task.funky.wendy": {"bar": "foo", "foo.bar": 0},
            "foo.bar": 1,  # broadcast option
            "foo.bar.fred": 1,  # broadcast option
            "foo.": 1,  # broadcast option
        }
        options = TaskBookOptions(**options)
        res_funky_wendy = options.task_options("funky.wendy")
        assert res_funky_wendy == {
            "bar": "foo",
            "foo.bar": 0,
            "foo.bar.fred": 1,
            "foo.": 1,
            "count": 1,
        }

        res_funky_wendy = options.task_options("funky")
        assert res_funky_wendy == {
            "foo.bar": 1,
            "foo.bar.fred": 1,
            "foo.": 1,
            "count": 1,
        }

        res_funky_wendy = options.task_options("foo.bar")
        assert res_funky_wendy == {
            "foo.bar": 1,
            "foo.bar.fred": 1,
            "foo.": 1,
            "count": 1,
        }
