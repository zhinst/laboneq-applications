import pytest
from IPython.lib.pretty import pretty

from laboneq_applications.workflow.result import TaskResult, WorkflowResult
from laboneq_applications.workflow.task import task
from laboneq_applications.workflow.taskview import TaskView


@task
def task_a():
    return 1


@task
def task_b():
    return 2


class TestTaskView:
    @pytest.fixture()
    def view(self):
        return TaskView(
            [
                TaskResult(task=task_a, output=1),
                TaskResult(task=task_b, output=2),
                TaskResult(task=task_b, output=3),
                WorkflowResult(name="wf_result"),
            ],
        )

    def test_no_copy(self):
        t = TaskResult(task=task_a, output=1)
        tasks = [t]
        view = TaskView(tasks)
        assert view[0] is tasks[0]
        assert view["task_a"] is tasks[0]

    def test_unique(self, view):
        assert view.unique() == ["task_a", "task_b", "wf_result"]

    def test_repr(self, view):
        t = TaskResult(task=task_a, output=1)
        view = TaskView([t])
        assert (
            repr(view)
            == f"[TaskResult(name=task_a, output=1, input={{}}, func={task_a.func})]"
        )

    def test_str(self):
        t = TaskResult(task=task_a, output=1)
        view = TaskView([t])
        assert str(view) == "TaskResult(task_a)"

    def test_ipython_pretty(self):
        t = TaskResult(task=task_a, output=1)
        view = TaskView([t])
        assert pretty(view) == "TaskResult(task_a)"

    def test_len(self, view):
        assert len(view) == 4

    def test_getitem(self, view):
        # Test index
        assert view[1] == TaskResult(task=task_b, output=2)
        assert view[3] == WorkflowResult("wf_result")

        # Test slice
        assert view[0:2] == [
            TaskResult(task=task_a, output=1),
            TaskResult(task=task_b, output=2),
        ]

        # Test string
        assert view["task_b"] == TaskResult(task=task_b, output=2)
        assert view["wf_result"] == WorkflowResult("wf_result")

        # Test slice by name
        assert view["task_b", 0] == TaskResult(task=task_b, output=2)
        assert view["task_b", 1] == TaskResult(task=task_b, output=3)
        assert view["task_b", 0:4] == [
            TaskResult(task=task_b, output=2),
            TaskResult(task=task_b, output=3),
        ]
        assert view["task_b", :] == [
            TaskResult(task=task_b, output=2),
            TaskResult(task=task_b, output=3),
        ]
        assert view["wf_result", :] == [WorkflowResult("wf_result")]

    def test_getitem_invalid_arguments(self, view):
        with pytest.raises(IndexError):
            view[123]
        with pytest.raises(KeyError):
            view["i do not exist"]
        with pytest.raises(KeyError):
            view["i do not exist", 0]
        with pytest.raises(IndexError):
            view["task_b", 12345]

    def test_eq(self):
        # TaskView to TaskView
        assert TaskView(
            [
                TaskResult(task=task_b, output=2),
                TaskResult(task=task_b, output=3),
            ],
        ) == TaskView(
            [
                TaskResult(task=task_b, output=2),
                TaskResult(task=task_b, output=3),
            ],
        )
        assert TaskView() == TaskView()
        assert TaskView([1]) != TaskView([5])

        # TestView to list
        assert TaskView(
            [
                TaskResult(task=task_b, output=2),
                TaskResult(task=task_b, output=3),
            ],
        ) != [
            TaskResult(task=task_b, output=2),
            TaskResult(task=task_b, output=3),
        ]
        assert TaskView() != []
        assert TaskView() != [1, 2]

        # TaskView as list to list
        assert list(
            TaskView(
                [
                    TaskResult(task=task_b, output=2),
                    TaskResult(task=task_b, output=3),
                ],
            ),
        ) == [
            TaskResult(task=task_b, output=2),
            TaskResult(task=task_b, output=3),
        ]
