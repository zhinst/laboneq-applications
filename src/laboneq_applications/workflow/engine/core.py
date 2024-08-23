"""Core workflow objects."""

from __future__ import annotations

import inspect
import textwrap
from functools import update_wrapper
from typing import TYPE_CHECKING, Any, Callable, Generic, cast

from typing_extensions import ParamSpec

from laboneq_applications.core import now
from laboneq_applications.logbook import (
    LogbookStore,
    LoggingStore,
    active_logbook_store,
)
from laboneq_applications.workflow import _utils, exceptions
from laboneq_applications.workflow._context import (
    ExecutorStateContext,
    TaskExecutorContext,
)
from laboneq_applications.workflow.engine.block import (
    WorkflowBlockBuilder,
)
from laboneq_applications.workflow.engine.executor import ExecutionStatus, ExecutorState
from laboneq_applications.workflow.engine.graph import WorkflowGraph
from laboneq_applications.workflow.options import (
    WorkflowOptions,
    get_and_validate_param_type,
)
from laboneq_applications.workflow.taskview import TaskView

if TYPE_CHECKING:
    from datetime import datetime

    from laboneq_applications.workflow.task import Task


class WorkflowResult:
    """Workflow result."""

    def __init__(self):
        self._tasks: list[Task] = []
        self._output = None
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None

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

    @property
    def start_time(self) -> datetime | None:
        """The time when the workflow execution has started."""
        return self._start_time

    @property
    def end_time(self) -> datetime | None:
        """The time when the workflow execution has ended regularly or failed."""
        return self._end_time


Parameters = ParamSpec("Parameters")


class WorkflowRecovery:
    """A layer of indirection for storing workflow recovery results."""

    def __init__(self):
        self.results = None


class Workflow(Generic[Parameters]):
    """Workflow for task execution.

    Arguments:
        graph: A workflow graph.
        **parameters: Parameters of the graph.
    """

    def __init__(
        self,
        graph: WorkflowGraph,
        **parameters: object,
    ) -> None:
        self._graph = graph
        self._graph.validate_input(**parameters)
        self._input = parameters
        self._recovery = (
            None  # WorkflowRecovery (unused if left as None, set by WorkflowBuilder)
        )
        self._result: WorkflowResult | None = None
        self._state: ExecutorState | None = None

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

    def _options(self) -> WorkflowOptions:
        """Return the workflow options passed."""
        options = self._input.get("options", None)
        # TODO: Expose functionality on the graph so that
        #       we don't have to grub about for _root.options
        if options is None:
            options = self._graph._root.options()
        elif isinstance(options, dict):
            options = self._graph._root.options.from_dict(options)
        return options

    def _logstore(self, options_logstore: LogbookStore | None) -> LogbookStore:
        """Return the appropriate logbook store."""
        logstore = options_logstore
        if logstore is None:
            logstore = active_logbook_store()
        if logstore is None:
            logstore = LoggingStore()
        return logstore

    def _reset(self) -> None:
        """Reset workflow execution state."""
        self._state = None
        self._result = None

    def _execute(self, state: ExecutorState, result: WorkflowResult) -> WorkflowResult:
        with ExecutorStateContext.scoped(state):
            try:
                result._start_time = now()
                state.recorder.on_start(result)
                # TODO: Must be moved to WorkflowGraph for nested workflows.
                with state.set_active_result(result):
                    self._graph.execute(state, **self._input)
                result._end_time = now()
                state.recorder.on_end(result)
            except Exception as error:
                result._end_time = now()
                if self._recovery is not None:
                    self._recovery.results = result
                if not getattr(
                    error,
                    "_logged_by_task",
                    False,
                ):  # TODO: better mechanism
                    state.recorder.on_error(result, error)
                state.recorder.on_end(result)
                self._reset()
                raise
        if state.get_block_status(self._graph._root) == ExecutionStatus.IN_PROGRESS:
            self._result = result
            self._state = state
        return result

    def resume(self, until: str | None = None) -> WorkflowResult:
        """Resume workflow execution.

        Resumes the workflow execution from the previous state.

        Arguments:
            until: Run until a first task with the given name.
                `None` will fully execute the workflow.

        Returns:
            Result of the workflow execution.

            if `until` is used, returns the results up to the selected task.

        Raises:
            WorkflowError: An error occurred during workflow execution or
                workflow is not in progress.
        """
        if not self._state or not self._result:
            raise exceptions.WorkflowError("Workflow is not in progress.")
        self._state.settings.run_until = until
        return self._execute(self._state, self._result)

    def run(
        self,
        until: str | None = None,
    ) -> WorkflowResult:
        """Run the workflow.

        Resets the state of an workflow before execution.

        Arguments:
            until: Run until the first task with the given name.
                `None` will fully execute the workflow.

                If `until` is used, the workflow execution can be resumed with
                `.resume()`.

        Returns:
            Result of the workflow execution.

            if `until` is used, returns the results up to the selected task.

        Raises:
            WorkflowError: An error occurred during workflow execution.
        """
        if isinstance(
            TaskExecutorContext.get_active(),
            WorkflowBlockBuilder,
        ):
            msg = "Nesting Workflows is not allowed."
            raise exceptions.WorkflowError(msg)
        self._reset()
        result = WorkflowResult()
        options = self._options()
        logstore = self._logstore(options.logstore)
        logbook = logstore.create_logbook(self)
        state = ExecutorState(options=options)
        state.add_recorder(logbook)
        state.settings.run_until = until
        return self._execute(state, result)


class WorkflowBuilder(Generic[Parameters]):
    """A workflow builder.

    Builds a workflow out of the given Python function.

    Arguments:
        func: A python function, which acts as the core of the workflow.
    """

    def __init__(self, func: Callable[Parameters]) -> None:
        self._func = func
        self._recovery = WorkflowRecovery()
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

    def recover(self) -> WorkflowResult:
        """Recover the result of the last run to raise an exception.

        Returns the result of the last failed run of a workflow created from
        this workflow builder. In no run has failed, an exception is raised.

        After a result is recovered, the result is cleared and further calls
        to `.recover` will raise an exception.

        Returns:
            Latest workflow that raised an exception.

        Raises:
            WorkflowError:
                Raised if no previous run failed.
        """
        if self._recovery.results is None:
            raise exceptions.WorkflowError("Workflow has no result to recover.")
        result = self._recovery.results
        self._recovery.results = None
        return result

    def __call__(  #  noqa: D102
        self,
        *args: Parameters.args,
        **kwargs: Parameters.kwargs,
    ) -> Workflow[Parameters]:
        wf = Workflow.from_callable(self._func, *args, **kwargs)
        wf._recovery = self._recovery
        return wf


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
