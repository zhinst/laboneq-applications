"""Workflow result objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from laboneq_applications.workflow.taskview import TaskView

if TYPE_CHECKING:
    from datetime import datetime

    from laboneq_applications.workflow.task import TaskResult


class WorkflowResult:
    """Workflow result."""

    def __init__(self, name: str, input: dict | None = None):  # noqa: A002
        self._name = name
        self._input = input or {}
        self._tasks: list[TaskResult] = []
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
