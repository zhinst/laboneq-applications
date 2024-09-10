import textwrap

from IPython.lib.pretty import pretty

from laboneq_applications.workflow.result import TaskResult, WorkflowResult
from laboneq_applications.workflow.task import task, task_


class TestWorkflowResult:
    def test_name(self):
        obj = WorkflowResult("test")
        assert obj.name == "test"

    def test_input(self):
        obj = WorkflowResult("test")
        assert obj.input == {}

        inp = {"foo": 1, "bar": 2}
        obj = WorkflowResult("test", input=inp)
        assert obj.input == inp

    def test_add_task(self):
        def addition(): ...

        obj = WorkflowResult("test")
        assert len(obj.tasks) == 0
        t = TaskResult(task_(addition), output=1)
        obj._tasks.append(t)
        assert len(obj.tasks) == 1
        assert obj.tasks["addition"] == t

    def test_str(self):
        assert str(WorkflowResult("test")) == "WorkflowResult(test)"

    def test_ipython_pretty(self):
        assert pretty(WorkflowResult("test")) == "WorkflowResult(test)"

    def test_eq(self):
        assert WorkflowResult("test") == WorkflowResult("test")
        assert WorkflowResult("test") != WorkflowResult("test1")
        assert WorkflowResult("test", {"a": 1}) == WorkflowResult(
            "test", input={"a": 1}
        )
        assert WorkflowResult("test", {"a": 1}) != WorkflowResult(
            "test", input={"a": 2}
        )


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
