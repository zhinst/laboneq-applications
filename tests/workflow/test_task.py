import textwrap

from laboneq_applications.workflow.task import FunctionTask, Task, task


class MyTestTask(Task):
    def _run(self):
        return 123


class TestTask:
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


class TestFunctionTask:
    def test_result(self):
        task_ = FunctionTask(foobar, "foobar")
        assert task_(1, 2) == 3

    def test_src(self):
        task_ = FunctionTask(foobar, "foobar")
        assert task_.src == textwrap.dedent("""\
            def foobar(x, y):
                return x + y
        """)


def test_task_reinitialized():
    task1 = FunctionTask(foobar, "foobar")
    task2 = task(task1)
    assert task2.name == "foobar"
    assert task1._func == task2._func
    assert task1 is not task2

    task1 = FunctionTask(foobar, "foobar")
    task2 = task(task1, name="foobar2")
    assert task2.name == "foobar2"
    assert task1._func == task2._func
    assert task1 is not task2
