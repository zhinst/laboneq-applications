"""Core workflow objects."""

from __future__ import annotations

from collections import defaultdict
from inspect import signature
from typing import Any, Callable

from typing_extensions import Self  # in `typing` from Python 3.11

from laboneq_applications.workflow import exceptions
from laboneq_applications.workflow._context import LocalContext
from laboneq_applications.workflow.orchestrator import sort_task_graph
from laboneq_applications.workflow.promise import Promise
from laboneq_applications.workflow.task import TaskEvent


class WorkflowResult:
    """Workflow result."""

    def __init__(self, tasklog: dict[str, list[Any]]):
        # TODO: Reserve `artifacts` property for the explicitly (important) saved items
        self._tasklog = tasklog

    @property
    def tasklog(self) -> dict[str, list[Any]]:
        """Log of executed tasks and their results.

        Returns:
            A mapping for each task and its' results.
        """
        # NOTE: Currently values are a list of task return values,
        #       However it will be a `TaskResult` object, which can have more
        #       information e.g runtime, errors, etc.
        return self._tasklog


class WorkflowInput:
    """Workflow input.

    Holds promises for workflow inputs during the construction of the workflow.
    The promises are resolved when the workflow is ran and the input values are
    provided.
    """

    def __init__(self):
        self._input: dict[str, Promise] = {}

    def __getitem__(self, key: str) -> Promise:
        if key not in self._input:
            self._input[key] = Promise()
        return self._input[key]

    def resolve(self, **kwargs: object) -> None:
        """Resolve input parameters.

        Arguments:
            **kwargs: The values for the workflow inputs.

        Raises:
            TypeError: Invalid or missing parameters.
        """
        undefined = set(kwargs.keys()) - set(self._input.keys())
        if undefined:
            raise TypeError(
                f"Workflow got undefined input parameter(s): {', '.join(undefined)}",
            )
        required_keys = set(self._input.keys()) - set(kwargs.keys())
        if required_keys:
            raise TypeError(
                f"Workflow missing input parameter(s): {', '.join(required_keys)}",
            )
        for k, v in kwargs.items():
            self._input[k].set_result(v)


class Workflow:
    """Workflow for task execution."""

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
        if self._tasks:
            self._tasks.extend(LocalContext.exit().instances)
        else:
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
            task_id = id(task)
            graph[task_id] = [id(t) for t in task.requires if isinstance(t, TaskEvent)]
            event_map[task_id] = task
        return [event_map[idd] for idd in sort_task_graph(graph)]

    def run(self, **kwargs: object) -> WorkflowResult:
        """Run workflow.

        Arguments:
            **kwargs: Keyword arguments of the workflow.

        Returns:
            Result of the workflow execution.
        """
        if LocalContext.is_active():
            msg = "Nesting Workflows is not allowed."
            raise exceptions.WorkflowError(msg)
        # Resolve Workflow input immediately, they do not have dependencies
        self.input.resolve(**kwargs)
        # TODO: We might not want to save each task result, memory overload?
        #       Kept for POC purposes
        tasklog = defaultdict(list)
        for event in self._dag:
            result = event.execute()
            tasklog[event.task.name].append(result)
        return WorkflowResult(tasklog=dict(tasklog))


class WorkflowBuilder:
    """A workflow builder.

    Builds a workflow out of the given Python function.

    Arguments:
        func: A python function, which acts as the core of the workflow.
            The docstring of the class is replaced with the `func` docstring
            if it has one.
    """

    def __init__(self, func: Callable) -> None:
        self._func = func
        if func.__doc__:
            msg = (
                "This function is a `WorkflowBuilder` and has additional "
                "functionality described in the `WorkflowBuilder` documentation."
            )
            self.__doc__ = "\n\n".join([func.__doc__, msg])

    def create(self) -> Workflow:
        """Create a workflow."""
        with Workflow() as wf:
            self._func(**{x: wf.input[x] for x in signature(self._func).parameters})
        return wf

    def __call__(self, **kw: object) -> WorkflowResult:
        """Create and execute a workflow."""
        wf = self.create()
        return wf.run(**kw)


def workflow(func: Callable) -> WorkflowBuilder:
    """A decorator to mark a function as workflow.

    The arguments of the function will be the input values for the `Workflow`.

    Returns:
        A Workflow builder, which can be used to generate a workflow out of the
        wrapped function.

    Example:
        ```python
        from laboneq_applications.workflow import workflow

        @workflow
        def my_workflow(x: int):
            ...

        wf = my_workflow.create()
        results = wf.run(x=123)
        ```
    """
    return WorkflowBuilder(func)
