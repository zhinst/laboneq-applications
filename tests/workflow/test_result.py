from laboneq_applications.workflow.result import WorkflowResult
from laboneq_applications.workflow.task import Task

from tests.workflow.engine.test_engine import addition


class TestWorkflowResult:
    def test_name(self):
        obj = WorkflowResult("test")
        assert obj.name == "test"

    def test_add_task(self):
        obj = WorkflowResult("test")
        assert len(obj.tasks) == 0
        t = Task(addition, output=1)
        obj._tasks.append(t)
        assert len(obj.tasks) == 1
        assert obj.tasks["addition"] == t
