"""Core workflow objects."""

from __future__ import annotations

import inspect
import textwrap
from inspect import signature
from typing import Any, Callable

from typing_extensions import Self  # in `typing` from Python 3.11

from laboneq_applications.workflow import exceptions
from laboneq_applications.workflow._context import get_active_context
from laboneq_applications.workflow.engine.block import (
    Block,
    BlockResult,
    WorkflowBlockExecutorContext,
)
from laboneq_applications.workflow.engine.promise import Promise


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


class WorkflowBlock(Block):
    """Workflow block."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def execute(self) -> BlockResult:
        """Execute the block."""
        r = BlockResult()
        for block in self._body:
            r.merge(self._run_block(block))
        return r


class Workflow:
    """Workflow for task execution."""

    # TODO: Should Workflow be serializable?
    def __init__(self) -> None:
        self._input = WorkflowInput()
        self._block = WorkflowBlock()

    @property
    def input(self) -> WorkflowInput:
        """Workflow input."""
        return self._input

    def __enter__(self) -> Self:
        """Enter the Workflow building context."""
        if isinstance(get_active_context(), WorkflowBlockExecutorContext):
            msg = "Nesting Workflows is not allowed."
            raise exceptions.WorkflowError(msg)
        self._block.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        """Exit the workflow building context."""
        self._block.__exit__(exc_type, exc_value, traceback)

    def run(self, **kwargs: object) -> WorkflowResult:
        """Run the workflow.

        Arguments:
            **kwargs: Keyword arguments of the workflow.

        Returns:
            Result of the workflow execution.

        Raises:
            TypeError: Workflow arguments do not match defined inputs.
            WorkflowError: An error occurred during workflow execution.
        """
        if isinstance(get_active_context(), WorkflowBlockExecutorContext):
            msg = "Nesting Workflows is not allowed."
            raise exceptions.WorkflowError(msg)
        # Resolve Workflow input immediately, they do not have dependencies
        self.input.resolve(**kwargs)
        # TODO: We might not want to save each task result, memory overload?
        #       Kept for POC purposes
        tasklog = self._block.execute()
        return WorkflowResult(tasklog=tasklog.log)


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

    @property
    def src(self) -> str:
        """Source code of the workflow."""
        src = inspect.getsource(self._func)
        return textwrap.dedent(src)

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

    The arguments of the function will be the input values for the wrapped function and
    must be supplied as keyword arguments.

    Returns:
        A Workflow builder, which can be used to generate a workflow out of the
        wrapped function.

    Example:
        ```python
        from laboneq_applications.workflow.engine import workflow

        @workflow
        def my_workflow(x: int):
            ...

        wf = my_workflow.create()
        results = wf.run(x=123)
        ```

        Executing the workflow without an instance:

        ```python
        results = my_workflow(x=123)
        ```
    """
    return WorkflowBuilder(func)
