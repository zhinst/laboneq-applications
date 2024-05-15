import textwrap

from laboneq_applications.workflow.task import FunctionTask, Task, TaskBlock, task


class MyTestTask(Task):
    def run(self):
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
            def run(self):
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


@task
def addition(): ...


class TestTaskBlock:
    def test_name(self):
        blk = TaskBlock(addition)
        assert blk.name == "addition"

    def test_repr(self):
        blk = TaskBlock(addition)
        assert str(blk) == "Task(name=addition)"

    def test_execute(self):
        @task
        def addition(x, y):
            return x + y

        blk = TaskBlock(addition, 1, y=2)
        r = blk.execute()
        assert r.log == {"addition": [3]}

    def test_src(self):
        @task
        def addition(x, y):
            return x + y

        blk = TaskBlock(addition, 1, y=2)
        assert blk.src == textwrap.dedent("""\
            @task
            def addition(x, y):
                return x + y
        """)
