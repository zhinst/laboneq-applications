"""Core workflow objects."""

from __future__ import annotations

import inspect
import textwrap
from collections import defaultdict
from functools import update_wrapper
from typing import TYPE_CHECKING, Any, Callable, Generic, cast

from typing_extensions import ParamSpec

from laboneq_applications.workflow import _utils, exceptions
from laboneq_applications.workflow._context import TaskExecutorContext
from laboneq_applications.workflow.engine.block import (
    WorkflowBlockBuilder,
)
from laboneq_applications.workflow.engine.executor import ExecutorState
from laboneq_applications.workflow.engine.graph import WorkflowGraph

if TYPE_CHECKING:
    from laboneq_applications.workflow.task import Task


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
        #       However it will be a `Task` object, which can have more
        #       information e.g runtime, errors, etc.
        return self._tasklog


class ResultCollector:
    """Workflow result collector."""

    def __init__(self) -> None:
        self._tasks = defaultdict(list)

    def on_task_end(self, task: Task) -> None:
        """Register task end."""
        self._tasks[task.name].append(task.output)


Parameters = ParamSpec("Parameters")


class Workflow(Generic[Parameters]):
    """Workflow for task execution.

    Arguments:
        graph: A workflow graph.
        **parameters: Parameters of the graph.
    """

    # TODO: Should Workflow be serializable?
    def __init__(
        self,
        graph: WorkflowGraph,
        **parameters: object,
    ) -> None:
        self._graph = graph
        self._input = parameters

    @classmethod
    def from_callable(cls, func: Callable, *args: object, **kwargs: object) -> Workflow:
        """Create a workflow from a callable.

        Arguments:
            func: A callable defining the workflow
            *args: Arguments of the callable
            **kwargs: Keyword arguments of the callable
        """
        return cls(
            WorkflowGraph.from_callable(func),
            **_utils.create_argument_map(func, *args, **kwargs),
        )

    @property
    def input(self) -> dict:
        """Input parameters of the workflow."""
        return self._input

    def run(self) -> WorkflowResult:
        """Run the workflow.

        Returns:
            Result of the workflow execution.

        Raises:
            WorkflowError: An error occurred during workflow execution.
        """
        if isinstance(
            TaskExecutorContext.get_active(),
            WorkflowBlockBuilder,
        ):
            msg = "Nesting Workflows is not allowed."
            raise exceptions.WorkflowError(msg)
        state = ExecutorState()
        # TODO: Result collector should be injected from the outside. E.g a logbook
        results = ResultCollector()
        state.set_result_callback(results)
        self._graph.execute(state, **self._input)
        return WorkflowResult(tasklog=dict(results._tasks))


class WorkflowBuilder(Generic[Parameters]):
    """A workflow builder.

    Builds a workflow out of the given Python function.

    Arguments:
        func: A python function, which acts as the core of the workflow.
    """

    def __init__(self, func: Callable[Parameters]) -> None:
        self._func = func

    @property
    def src(self) -> str:
        """Source code of the workflow."""
        src = inspect.getsource(self._func)
        return textwrap.dedent(src)

    def __call__(  #  noqa: D102
        self,
        *args: Parameters.args,
        **kwargs: Parameters.kwargs,
    ) -> Workflow[Parameters]:
        return Workflow.from_callable(self._func, *args, **kwargs)


def workflow(func: Callable[Parameters]) -> WorkflowBuilder[Parameters]:
    """A decorator to mark a function as workflow.

    The arguments of the function will be the input values for the wrapped function.

    Returns:
        A Workflow builder, which can be used to generate a workflow out of the
        wrapped function.

    Example:
        ```python
        from laboneq_applications.workflow.engine import workflow

        @workflow
        def my_workflow(x: int):
            ...

        wf = my_workflow(x=123)
        results = wf.run()
        ```
    """
    return cast(
        WorkflowBuilder[Parameters],
        update_wrapper(
            WorkflowBuilder(func),
            func,
        ),
    )
