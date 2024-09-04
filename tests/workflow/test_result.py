from IPython.lib.pretty import pretty

from laboneq_applications.workflow.result import WorkflowResult
from laboneq_applications.workflow.task import TaskResult, task_


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
