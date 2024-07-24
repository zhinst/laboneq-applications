import inspect
import textwrap

import pytest
from IPython.lib.pretty import pretty

from laboneq_applications.workflow.task import Task, task, task_


def foobar(x, y):
    return x + y


class TestTask_:  # noqa: N801
    @pytest.fixture()
    def obj(self):
        def foobar(x, y):
            return x + y

        return task_(foobar, "foobar")

    def test_call(self, obj):
        assert obj(1, 2) == 3

    def test_src(self, obj):
        assert obj.src == textwrap.dedent("""\
            def foobar(x, y):
                return x + y
        """)

    def test_name(self, obj):
        assert obj.name == "foobar"

    def test_has_opts(self, obj):
        assert obj.has_opts is False


class TestTaskDecorator:
    def test_name(self):
        @task(name="test")
        def foo1(x): ...

        assert foo1.name == "test"

        @task
        def foo2(x): ...

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

    def test_task_reinitialized(self):
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


class TestTaskOptions:
    """Test tasks called with options outside of taskbook."""

    def test_task_options(self):
        @task
        def task_a(a, options=None):
            return a, options["count"]

        assert task_a(1, options={"count": 2}) == (1, 2)


class TestTask:
    @task
    def task_a():
        return 1

    @task
    def task_b():
        TestTask.task_a()
        return 2

    def test_name(self):
        t = Task(self.task_a, 2)
        assert t.name == "task_a"

    def test_func(self):
        t = Task(self.task_a, 2)
        assert t.func == self.task_a.func

    def test_eq(self):
        e1 = Task(self.task_a, 2)
        e2 = Task(self.task_a, 2)
        assert e1 == e2

        e1 = Task(self.task_a, 2)
        e2 = Task(self.task_b, 2)
        assert e1 != e2

        e1 = Task(self.task_a, 2)
        assert e1 != 2
        assert e1 != "bar"

        assert Task(self.task_a, 1) != Task(self.task_a, 2)
        assert Task(self.task_a, 1, {"param": 1}) != Task(self.task_a, 1, {"param": 2})

    def test_repr(self):
        t = Task(self.task_a, 2)
        assert repr(t) == f"Task(name=task_a, output=2, input={{}}, func={t.func})"

    def test_str(self):
        t = Task(self.task_a, 2)
        assert str(t) == "Task(task_a)"

    def test_ipython_pretty(self):
        t = Task(self.task_a, 2)
        assert pretty(t) == "Task(task_a)"

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
