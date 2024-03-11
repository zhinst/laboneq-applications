"""Core workflow objects."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from typing_extensions import Self  # in `typing` from Python 3.11

from laboneq_library.workflow import exceptions
from laboneq_library.workflow._context import LocalContext
from laboneq_library.workflow.orchestrator import sort_task_graph
from laboneq_library.workflow.promise import Promise
from laboneq_library.workflow.task import TaskEvent


class WorkflowResult:
    """Workflow result."""

    def __init__(self, tasklog: dict[str, list[Any]]):
        # TODO: Reserve `artifacts` property for the explicitly (important) saved items
        self._tasklog = tasklog

    @property
    def tasklog(self) -> dict[str, list[Any]]:
        """Log of executed tasks and their results.

        Returns:
            A mapping for each task and its results.
        """
        # NOTE: Currently values are a list of task return values,
        #       However it will be a `TaskResult` object, which can have more
        #       information e.g runtime, errors, etc.
        return self._tasklog


class WorkflowInput:
    """Workflow input.

    Creates promises of the inputs used, which are then resolved
    once the workflow is ran.

    Attributes:
        args: Arguments of the input.
        kwargs: Keyword arguments of the input
    """
    def __init__(self):
        self.args: tuple = Promise()
        self.kwargs: dict = Promise()


class Workflow:
    """Workflow for Task execution."""

    # TODO: Should Workflow be serializable?
    def __init__(self) -> None:
        self._tasks: list[TaskEvent] = []
        self._dag: list[TaskEvent] = []
        self._input = WorkflowInput()

    @property
    def input(self) -> WorkflowInput:
        """Workflow input."""
        return self._input

    @property
    def tasks(self) -> list[TaskEvent]:
        """Tasks of the workflow."""
        return self._tasks

    def __enter__(self) -> Self:
        """Enter the Workflow building context."""
        if LocalContext.is_active():
            msg = "Nesting Workflows is not allowed."
            raise exceptions.WorkflowError(msg)
        LocalContext.enter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        """Exit the workflow building context.

        Resolves the topology of the Workflow.
        """
        self._tasks = LocalContext.exit().instances
        self._dag = self._resolve_topology(self._tasks)

    @staticmethod
    def _resolve_topology(tasks: list[TaskEvent]) -> list[TaskEvent]:
        """Resolve the topology of the tasks.

        Arguments:
            tasks: List of tasks whose topology is to be resolved.

        Returns:
            Execution order of the tasks.
        """
        graph = {}
        event_map = {}
        for task in tasks:
            graph[task.event_id] = [
                t.event_id for t in task.requires if isinstance(t, TaskEvent)
            ]
            event_map[task.event_id] = task
        return [event_map[idd] for idd in sort_task_graph(graph)]

    def run(self, *args, **kwargs) -> WorkflowResult:
        """Run workflow.

        Arguments:
            *args: Arguments of the workflow.
            **kwargs: Keyword arguments of the workflow.

        Returns:
            Result of the workflow execution.
        """
        if LocalContext.is_active():
            msg = "Nesting Workflows is not allowed."
            raise exceptions.WorkflowError(msg)
        # Resolve Workflow input args and kwargs immediately, they do not have
        # dependencies
        # TODO: Resolve workflow input promises more intelligently
        self.input.args.set_result(args)
        self.input.kwargs.set_result(kwargs)
        # TODO: We might not want to save each task result, memory overload?
        #       Kept for POC purposes
        tasklog = defaultdict(list)
        for event in self._dag:
            result = event.execute()
            tasklog[event.task.name].append(result)
        return WorkflowResult(tasklog=dict(tasklog))
