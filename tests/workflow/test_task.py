from laboneq_applications.workflow._context import LocalContext
from laboneq_applications.workflow.task import Task, task


class MyTestTask(Task):
    def run(self):
        ...


class TestTask:
    def test_repr(self):
        task = MyTestTask(name="test")
        assert str(task) == "Task(name=test)"

    def test_task_name(self):
        @task
        def foobar():
            ...
        with LocalContext():
            task_ = foobar()
        assert task_._ref.task._name == "foobar"

        @task(name="test")
        def foobar():
            ...
        with LocalContext():
            task_ = foobar()
        assert task_._ref.task._name == "test"

    def test_task_result_query(self):
        @task
        def foobar():
            return 123
        with LocalContext():
            event = foobar()
        event._ref.execute()
        assert event.result() == 123
