from laboneq_applications.workflow.result import WorkflowResult
from laboneq_applications.workflow.task import TaskResult

from tests.workflow.engine.test_engine import addition


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
        obj = WorkflowResult("test")
        assert len(obj.tasks) == 0
        t = TaskResult(addition, output=1)
        obj._tasks.append(t)
        assert len(obj.tasks) == 1
        assert obj.tasks["addition"] == t
