from laboneq_applications.workflow.task import FunctionTask, Task


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
