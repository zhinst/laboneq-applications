import textwrap

from laboneq_applications.workflow.task import FunctionTask, Task


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
