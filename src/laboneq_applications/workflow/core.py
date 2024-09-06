"""Core workflow objects."""

from __future__ import annotations

import inspect
import textwrap
from functools import partial, update_wrapper
from typing import TYPE_CHECKING, Callable, Generic, cast, overload

from typing_extensions import ParamSpec

from laboneq_applications.logbook import (
    LogbookStore,
    LoggingStore,
    active_logbook_store,
)
from laboneq_applications.workflow import _utils, exceptions
from laboneq_applications.workflow._context import (
    TaskExecutorContext,
)
from laboneq_applications.workflow.engine.block import (
    WorkflowBlockBuilder,
)
from laboneq_applications.workflow.engine.graph import WorkflowBlock, WorkflowGraph
from laboneq_applications.workflow.executor import (
    ExecutionStatus,
    ExecutorState,
    ExecutorStateContext,
)
from laboneq_applications.workflow.options import (
    WorkflowOptions,
)
from laboneq_applications.workflow.options_parser import (
    get_and_validate_param_type,
)

if TYPE_CHECKING:
    from laboneq_applications.workflow.result import WorkflowResult

Parameters = ParamSpec("Parameters")


class WorkflowRecovery:
    """A layer of indirection for storing workflow recovery results."""

    def __init__(self):
        self.results = None


class Workflow(Generic[Parameters]):
    """Workflow for task execution.

    Arguments:
        graph: A workflow graph.
        input: Input parameters of the workflow.
    """

    def __init__(
        self,
        graph: WorkflowGraph,
        input: dict | None = None,  # noqa: A002
    ) -> None:
        self._graph = graph
        self._input = input or {}
        self._graph.validate_input(**self._input)
        self._recovery = (
            None  # WorkflowRecovery (unused if left as None, set by WorkflowBuilder)
        )
        self._state: ExecutorState | None = None

    @classmethod
    def from_callable(
        cls,
        func: Callable,
        name: str | None = None,
        input: dict | None = None,  # noqa: A002
    ) -> Workflow:
        """Create a workflow from a callable.

        Arguments:
            func: A callable defining the workflow
            name: Name of the workflow
            input: Input parameters of the workflow
        """
        params = input or {}
        return cls(
            WorkflowGraph.from_callable(func, name),
            _utils.create_argument_map(func, **params),
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
        if options is None:
            # TODO: Replace with create_options() when new options are in
            options = self._graph.options_type()
        elif isinstance(options, dict):
            options = self._graph.options_type.from_dict(options)
        return cast(WorkflowOptions, options)

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

    def _execute(self, state: ExecutorState) -> WorkflowResult:
        with ExecutorStateContext.scoped(state):
            try:
                self._graph.execute(state, **self._input)
                result = state.get_state(self._graph._root)
            except Exception:
                if self._recovery is not None:
                    result = state.get_state(self._graph._root)
                    self._recovery.results = result
                raise
        if state.get_block_status(self._graph._root) == ExecutionStatus.IN_PROGRESS:
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
        if not self._state:
            raise exceptions.WorkflowError("Workflow is not in progress.")
        self._state.settings.run_until = until
        return self._execute(self._state)

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
            msg = "Calling '.run()' within another workflow is not allowed."
            raise exceptions.WorkflowError(msg)
        self._reset()
        options = self._options()
        logstore = self._logstore(options.logstore)
        logbook = logstore.create_logbook(self)
        state = ExecutorState()
        state.add_recorder(logbook)
        state.settings.run_until = until
        return self._execute(state)


class WorkflowBuilder(Generic[Parameters]):
    """A workflow builder.

    Builds a workflow out of the given Python function.

    Arguments:
        func: A python function, which acts as the core of the workflow.
        name: Name of the workflow.
            Defaults to wrapped function name.
    """

    def __init__(self, func: Callable[Parameters], name: str | None = None) -> None:
        self._func = func
        self._name = name or self._func.__name__
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
        active_ctx = TaskExecutorContext.get_active()
        if isinstance(active_ctx, WorkflowBlockBuilder):
            blk = WorkflowBlock.from_callable(
                self._name,
                self._func,
                **_utils.create_argument_map(self._func, *args, **kwargs),
            )
            return blk.ref
        wf = Workflow.from_callable(
            self._func,
            name=self._name,
            input=_utils.create_argument_map(self._func, *args, **kwargs),
        )
        wf._recovery = self._recovery
        return wf

    def options(self) -> WorkflowOptions:
        """Create default options for the workflow.

        The option attribute `tasks` is populated with all the sub-task
        and sub-workflow options within this workflow.
        """
        params = {}
        for key in inspect.signature(self._func).parameters:
            params[key] = None
        wf = self(**params)
        return wf._graph.create_options()


@overload
def workflow(func: Callable[Parameters], name: str) -> WorkflowBuilder[Parameters]: ...


@overload
def workflow(func: Callable[Parameters]) -> WorkflowBuilder[Parameters]: ...


@overload
def workflow(
    func: None = ...,
    name: str | None = ...,
) -> Callable[[Callable[Parameters]], WorkflowBuilder[Parameters]]: ...


def workflow(
    func: Callable[Parameters] | None = None, name: str | None = None
) -> (
    WorkflowBuilder[Parameters]
    | Callable[[Callable[Parameters]], WorkflowBuilder[Parameters]]
):
    """A decorator to mark a function as workflow.

    The arguments of the function will be the input values for the wrapped function.

    If `workflow` decorated function is called within another workflow definition,
    it adds a sub-graph to the workflow being built.

    Arguments:
        func: A function that defines the workflow structure.

            The arguments of `func` can be freely defined, except for an optional
            argument `options`, which must have a type hint that indicates it is of type
            `WorkflowOptions` or its' subclass, otherwise an error is raised.
        name: Name of the workflow.
            Defaults to wrapped function name.

    Returns:
        A wrapper which returns a `Workflow` instance if called outside of
            another workflow definition.
        A wrapper which returns a `WorkflowResult` if called within
            another workflow definition.

    Example:
        ```python
        from laboneq_applications import workflow

        @workflow.workflow
        def my_workflow(x: int):
            ...

        wf = my_workflow(x=123)
        results = wf.run()
        ```
    """
    if isinstance(
        TaskExecutorContext.get_active(),
        WorkflowBlockBuilder,
    ):
        msg = "Defining a workflow inside a workflow is not allowed."
        raise exceptions.WorkflowError(msg)

    if func is None:
        return cast(WorkflowBuilder[Parameters], partial(WorkflowBuilder, name=name))
    return cast(
        WorkflowBuilder[Parameters],
        update_wrapper(
            WorkflowBuilder(func, name=name),
            func,
        ),
    )
