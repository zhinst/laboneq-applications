"""Workflow result objects."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from laboneq_applications.workflow.taskview import TaskView

if TYPE_CHECKING:
    from datetime import datetime

    from laboneq_applications.workflow.task import task_


class TaskResult:
    """Task result.

    The instance holds execution information of an task.
    """

    def __init__(
        self,
        task: task_,
        output: object,
        input: dict | None = None,  # noqa: A002
    ) -> None:
        self._task = task
        self._output = output
        self._input = input or {}
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None

    @property
    def name(self) -> str:
        """Task name."""
        return self._task.name

    @property
    def func(self) -> Callable:
        """Underlying function."""
        return self._task.func

    @property
    def src(self) -> str:
        """Source code of the task."""
        return self._task.src

    @property
    def output(self) -> object:
        """Output of the task."""
        return self._output

    @property
    def input(self) -> dict:
        """Input parameters of the task."""
        return self._input

    @property
    def start_time(self) -> datetime | None:
        """Time when the task has started."""
        return self._start_time

    @property
    def end_time(self) -> datetime | None:
        """Time when the task has ended regularly or failed."""
        return self._end_time

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, TaskResult):
            return NotImplemented
        return (
            self._task == value._task
            and self.output == value.output
            and self.input == value.input
        )

    def __repr__(self) -> str:
        attrs = ", ".join(
            [
                f"name={self.name}",
                f"output={self.output}",
                f"input={self.input}",
                f"func={self.func}",
            ],
        )
        return f"TaskResult({attrs})"

    def __str__(self) -> str:
        return f"TaskResult({self.name})"

    def _repr_pretty_(self, p, cycle):  # noqa: ANN001, ANN202, ARG002
        # For Notebooks
        p.text(str(self))


class WorkflowResult:
    """Workflow result."""

    def __init__(self, name: str, input: dict | None = None):  # noqa: A002
        self._name = name
        self._input = input or {}
        self._tasks: list[TaskResult | WorkflowResult] = []
        self._output = None
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None

    @property
    def name(self) -> str:
        """Name of the workflow producing the results."""
        return self._name

    @property
    def input(self) -> dict:
        """Input of the workflow."""
        return self._input

    @property
    def output(self) -> Any:  # noqa: ANN401
        """Output of the workflow."""
        return self._output

    @property
    def tasks(self) -> TaskView:
        """Task entries of the workflow.

        The ordering of the tasks is the order of the execution.

        Tasks is a `Sequence` of tasks, however item lookup
        is modified to support the following cases:

        Example:
            ```python
            wf = my_workflow.run()
            wf.tasks["run_experiment"]  # First task of name 'run_experiment'
            wf.tasks["run_experiment", :]  # All tasks named 'run_experiment'
            wf.tasks["run_experiment", 1:5]  # Slice tasks named 'run_experiment'
            wf.tasks[0]  # First executed task
            wf.tasks[0:5]  # Slicing

            wf.tasks.unique() # Unique task names
            ```
        """
        return TaskView(self._tasks)

    @property
    def start_time(self) -> datetime | None:
        """The time when the workflow execution has started."""
        return self._start_time

    @property
    def end_time(self) -> datetime | None:
        """The time when the workflow execution has ended regularly or failed."""
        return self._end_time

    def __str__(self) -> str:
        return f"WorkflowResult({self.name})"

    def _repr_pretty_(self, p, cycle):  # noqa: ANN001, ANN202, ARG002
        # For Notebooks
        p.text(str(self))

    def __eq__(self, other: object):
        if not isinstance(other, WorkflowResult):
            return NotImplemented
        return (
            self.name == other.name
            and self.input == other.input
            and self.output == other.output
            and self.tasks == other.tasks
            and self.end_time == other.end_time
            and self.start_time == other.start_time
        )
