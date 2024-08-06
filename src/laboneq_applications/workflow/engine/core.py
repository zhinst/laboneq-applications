"""Core workflow objects."""

from __future__ import annotations

import inspect
import textwrap
from functools import update_wrapper
from typing import TYPE_CHECKING, Any, Callable, Generic, cast

from typing_extensions import ParamSpec

from laboneq_applications.logbook import LoggingStore
from laboneq_applications.workflow import _utils, exceptions
from laboneq_applications.workflow._context import (
    ExecutorStateContext,
    TaskExecutorContext,
)
from laboneq_applications.workflow.engine.block import (
    WorkflowBlockBuilder,
)
from laboneq_applications.workflow.engine.executor import ExecutorState
from laboneq_applications.workflow.engine.graph import WorkflowGraph
from laboneq_applications.workflow.engine.options import WorkflowOptions
from laboneq_applications.workflow.options import get_and_validate_param_type
from laboneq_applications.workflow.taskview import TaskView

if TYPE_CHECKING:
    from laboneq_applications.logbook import LogbookStore
    from laboneq_applications.workflow.task import Task


class WorkflowResult:
    """Workflow result."""

    def __init__(self):
        self._tasks: list[Task] = []
        self._output = None

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

    def add_task(self, task: Task) -> None:
        """Add a task result."""
        self._tasks.append(task)


class _ResultCollector:
    """Workflow result collector."""

    def __init__(self, result: WorkflowResult) -> None:
        self._result = result

    def on_task_end(self, task: Task) -> None:
        """Register task end."""
        self._result.add_task(task)

    def on_workflow_end(self, output: Any) -> None:  # noqa: ANN401
        """Register workflow end."""
        self._result._output = output


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
        self._graph.validate_input(**parameters)
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
    def name(self) -> str:
        """Workflow name."""
        return self._graph.name

    @property
    def input(self) -> dict:
        """Input parameters of the workflow."""
        return self._input

    def run(
        self,
        logstore: LogbookStore | None = None,  # TODO: read logbook from options
    ) -> WorkflowResult:
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

        if logstore is None:
            logstore = LoggingStore()
        logbook = logstore.create_logbook(self)

        state = ExecutorState(logbook=logbook)
        # TODO: Result collector should be injected from the outside. E.g a logbook
        results = WorkflowResult()
        collector = _ResultCollector(results)
        state.set_result_callback(collector)
        with ExecutorStateContext.scoped(state):
            self._graph.execute(state, **self._input)
        return results


class WorkflowBuilder(Generic[Parameters]):
    """A workflow builder.

    Builds a workflow out of the given Python function.

    Arguments:
        func: A python function, which acts as the core of the workflow.
    """

    def __init__(self, func: Callable[Parameters]) -> None:
        self._func = func
        if "options" in inspect.signature(func).parameters:
            opt_type = get_and_validate_param_type(
                func,
                parameter="options",
                type_check=WorkflowOptions,
            )
            if opt_type is None:
                msg = "Workflow input options must be of type 'WorkflowOptions'"
                raise TypeError(msg)

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

    Arguments:
        func: A function that acts as the core of the workflow.

            The arguments of `func` can be freely defined, except for an optional
            argument `options`, which must have a type hint that indicates it is of type
            `WorkflowOptions` or its' subclass, otherwise an error is raised.

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
