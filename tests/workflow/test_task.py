from laboneq_applications.workflow.task import FunctionTask, Task, TaskBlock, task


class MyTestTask(Task):
    def run(self): ...


class TestTask:
    def test_repr(self):
        task = MyTestTask(name="test")
        assert str(task) == "Task(name=test)"

    def test_name(self):
        task_ = MyTestTask("foobar")
        assert task_.name == "foobar"


class TestFunctionTask:
    def test_result(self):
        def foobar(x, y):
            return x + y

        task_ = FunctionTask(foobar, "foobar")
        assert task_(1, 2) == 3


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
