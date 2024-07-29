import pytest
from IPython.lib.pretty import pretty

from laboneq_applications.workflow.task import Task, task
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
                Task(task=task_a, output=1),
                Task(task=task_b, output=2),
                Task(task=task_b, output=3),
            ],
        )

    def test_no_copy(self):
        t = Task(task=task_a, output=1)
        tasks = [t]
        view = TaskView(tasks)
        assert view[0] is tasks[0]
        assert view["task_a"] is tasks[0]

    def test_unique(self, view):
        assert view.unique() == ["task_a", "task_b"]

    def test_repr(self, view):
        t = Task(task=task_a, output=1)
        view = TaskView([t])
        assert (
            repr(view)
            == f"[Task(name=task_a, output=1, input={{}}, func={task_a.func})]"
        )

    def test_str(self):
        t = Task(task=task_a, output=1)
        view = TaskView([t])
        assert str(view) == "Task(task_a)"

    def test_ipython_pretty(self):
        t = Task(task=task_a, output=1)
        view = TaskView([t])
        assert pretty(view) == "Task(task_a)"

    def test_len(self, view):
        assert len(view) == 3

    def test_getitem(self, view):
        # Test index
        assert view[1] == Task(task=task_b, output=2)
        # Test slice
        assert view[0:2] == [
            Task(task=task_a, output=1),
            Task(task=task_b, output=2),
        ]
        # Test string
        assert view["task_b"] == Task(task=task_b, output=2)
        # Test slice
        assert view["task_b", 0] == Task(task=task_b, output=2)
        assert view["task_b", 1] == Task(task=task_b, output=3)
        assert view["task_b", 0:4] == [
            Task(task=task_b, output=2),
            Task(task=task_b, output=3),
        ]
        assert view["task_b", :] == [
            Task(task=task_b, output=2),
            Task(task=task_b, output=3),
        ]

        with pytest.raises(IndexError):
            view[123]
        with pytest.raises(KeyError):
            view["i do not exist"]
        with pytest.raises(KeyError):
            view["i do not exist", 0]
        with pytest.raises(IndexError):
            view["task_b", 12345]

    def test_eq(self):
        assert TaskView(
            [
                Task(task=task_b, output=2),
                Task(task=task_b, output=3),
            ],
        ) == TaskView(
            [
                Task(task=task_b, output=2),
                Task(task=task_b, output=3),
            ],
        )

        assert TaskView(
            [
                Task(task=task_b, output=2),
                Task(task=task_b, output=3),
            ],
        ) != [
            Task(task=task_b, output=2),
            Task(task=task_b, output=3),
        ]
        assert list(
            TaskView(
                [
                    Task(task=task_b, output=2),
                    Task(task=task_b, output=3),
                ],
            ),
        ) == [
            Task(task=task_b, output=2),
            Task(task=task_b, output=3),
        ]
        assert TaskView() != []
        assert TaskView() != [1, 2]
        assert TaskView() == TaskView()
        assert TaskView([1]) != TaskView([5])
