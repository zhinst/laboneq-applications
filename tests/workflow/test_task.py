from __future__ import annotations

import inspect
import textwrap

from IPython.lib.pretty import pretty

from laboneq_applications.workflow.options_base import BaseOptions
from laboneq_applications.workflow.task import TaskResult, task


class TestTaskDecorator:
    def test_name(self):
        @task(name="test")
        def foo1(): ...

        assert foo1.name == "test"

        @task
        def foo2(): ...

        assert foo2.name == "foo2"

    def test_doc(self):
        @task(name="test")
        def foo(x):
            """Best doc"""

        assert foo.__doc__ == "Best doc"

    def test_signature_matches_function(self):
        def myfunc(x: int) -> str:
            return str(x)

        assert inspect.signature(task(myfunc)) == inspect.signature(myfunc)

    def test_call(self):
        @task
        def a_task(x, y):
            return x + y

        assert a_task(1, 2) == 3

    def test_src(self):
        @task
        def a_task(x, y):
            return x + y

        assert a_task.src == textwrap.dedent("""\
            @task
            def a_task(x, y):
                return x + y
        """)

    def test_task_reinitialized(self):
        def foobar():
            return None

        task1 = task(foobar, "foobar")
        task2 = task(task1)
        assert task2.name == "foobar"
        assert task1._func == task2._func
        assert task1 is not task2

        task1 = task(foobar, "foobar")
        task2 = task(task1, name="foobar2")
        assert task2.name == "foobar2"
        assert task1._func == task2._func
        assert task1 is not task2

    def test_options(self):
        @task
        def a_task():
            return None

        assert a_task._options is None

    def test_repr(self):
        @task
        def a_task(x, y):
            return x + y

        assert repr(a_task) == f"task(func={a_task.func}, name=a_task)"

    def test_str(self):
        @task
        def a_task(x, y):
            return x + y

        assert str(a_task) == "task(name=a_task)"


class NotABaseOptionTypeOptions:
    def __init__(self) -> None:
        pass


class ProperOptions(BaseOptions): ...


class TestTaskOptions:
    def test_task_called_outside_of_workflow_with_options(self):
        @task
        def task_a(a, options=None):
            return a, options["count"]

        assert task_a(1, options={"count": 2}) == (1, 2)

    def test_task_options_not_base_type(self):
        # TODO: Should this raise an exception instead of None?
        @task
        def task_a(options: NotABaseOptionTypeOptions | None = None): ...

        assert task_a._options is None

    def test_task_options_base_type(self):
        @task
        def task_a(options: ProperOptions | None = None): ...

        assert task_a._options == ProperOptions


class TestTaskResult:
    @task
    def task_a():
        return 1

    @task
    def task_b():
        TestTaskResult.task_a()
        return 2

    def test_name(self):
        t = TaskResult(self.task_a, 2)
        assert t.name == "task_a"

    def test_func(self):
        t = TaskResult(self.task_a, 2)
        assert t.func == self.task_a.func

    def test_eq(self):
        e1 = TaskResult(self.task_a, 2)
        e2 = TaskResult(self.task_a, 2)
        assert e1 == e2

        e1 = TaskResult(self.task_a, 2)
        e2 = TaskResult(self.task_b, 2)
        assert e1 != e2

        e1 = TaskResult(self.task_a, 2)
        assert e1 != 2
        assert e1 != "bar"

        assert TaskResult(self.task_a, 1) != TaskResult(self.task_a, 2)
        assert TaskResult(self.task_a, 1, {"param": 1}) != TaskResult(
            self.task_a, 1, {"param": 2}
        )

    def test_repr(self):
        t = TaskResult(self.task_a, 2)
        assert (
            repr(t) == f"TaskResult(name=task_a, output=2, input={{}}, func={t.func})"
        )

    def test_str(self):
        t = TaskResult(self.task_a, 2)
        assert str(t) == "TaskResult(task_a)"

    def test_ipython_pretty(self):
        t = TaskResult(self.task_a, 2)
        assert pretty(t) == "TaskResult(task_a)"

    def test_src(self):
        @task
        def task_():
            return 1

        t = TaskResult(task_, 2)
        assert t.src == textwrap.dedent("""\
            @task
            def task_():
                return 1
        """)
