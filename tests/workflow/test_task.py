import textwrap

from laboneq_applications.workflow.task import _BaseTask, task, task_


class MyTestTask(_BaseTask):
    def _run(self):
        return 123


class TestBaseTask:
    def test_repr(self):
        task = MyTestTask(name="test")
        assert str(task) == "Task(name=test)"

    def test_name(self):
        task_ = MyTestTask("foobar")
        assert task_.name == "foobar"

    def test_src(self):
        task_ = MyTestTask("foobar")
        assert task_.src == textwrap.dedent("""\
            def _run(self):
                return 123
        """)


def foobar(x, y):
    return x + y


class TestTaskWrapper:
    def test_result(self):
        t = task_(foobar, "foobar")
        assert t(1, 2) == 3

    def test_src(self):
        t = task_(foobar, "foobar")
        assert t.src == textwrap.dedent("""\
            def foobar(x, y):
                return x + y
        """)

    def test_doc(self):
        @task(name="test")
        def foo(x):
            """Best doc"""

        assert foo.__doc__ == "Best doc"

    def test_task_reinitialized(self):
        task1 = task_(foobar, "foobar")
        task2 = task(task1)
        assert task2.name == "foobar"
        assert task1._func == task2._func
        assert task1 is not task2

        task1 = task_(foobar, "foobar")
        task2 = task(task1, name="foobar2")
        assert task2.name == "foobar2"
        assert task1._func == task2._func
        assert task1 is not task2
