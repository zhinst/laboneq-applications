"""A module for expressions supported in Workflow."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeVar, cast

from laboneq_applications.workflow._context import TaskExecutorContext
from laboneq_applications.workflow.engine.block import (
    Block,
    WorkflowBlockBuilder,
)
from laboneq_applications.workflow.executor import ExecutionStatus
from laboneq_applications.workflow.reference import Reference

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from laboneq_applications.workflow.executor import ExecutorState


class IFExpression(Block):
    """If expression.

    A block that is executed if a given `condition` is true.

    Arguments:
        condition: A condition that has to be `True` for block to be
            executed.
    """

    def __init__(self, condition: Any) -> None:  # noqa: ANN401
        super().__init__(condition=condition)

    def execute(self, executor: ExecutorState) -> None:
        """Execute the block."""
        arg = executor.resolve_inputs(self)["condition"]
        if bool(arg):
            executor.set_block_status(self, ExecutionStatus.IN_PROGRESS)
            for block in self.body:
                if executor.get_block_status(block) == ExecutionStatus.FINISHED:
                    continue
                block.execute(executor)
            executor.set_block_status(self, ExecutionStatus.FINISHED)


@contextmanager
def if_(condition: Any) -> Generator[None, None, None]:  # noqa: ANN401
    """Workflow if statement.

    The equivalent of Python's if-statement.

    Arguments:
        condition: A condition that has to be `True` for code block to be
            executed.

    Example:
        ```python
       with if_(x == 1):
            ...
        ```
    """
    with IFExpression(condition=condition):
        yield


T = TypeVar("T")


class ForExpression(Block):
    """For expression.

    A block that iterates workflow blocks over the given values.

    The expression will always fully execute regardless if the workflow
    is partially executed or not.

    Arguments:
        values: An iterable.
            Iterable can contain workflow objects.
    """

    def __init__(self, values: Iterable | Reference) -> None:
        super().__init__(values=values)
        self._ref = Reference(self)

    @property
    def ref(self) -> Reference:
        """Reference to the object."""
        return self._ref

    def __enter__(self) -> Reference:
        """Enter the loop context.

        Returns:
            Individual values of the given iterable.
        """
        super().__enter__()
        return self.ref

    def execute(self, executor: ExecutorState) -> None:
        """Execute the block."""
        vals = executor.resolve_inputs(self)["values"]
        executor.set_block_status(self, ExecutionStatus.IN_PROGRESS)
        # Disable run until within the loop
        # TODO: Add support if seen necessary
        run_until = executor.settings.run_until
        executor.settings.run_until = None
        try:
            for val in vals:
                executor.set_state(self, val)
                for block in self.body:
                    block.execute(executor)
        finally:
            executor.settings.run_until = run_until
        executor.set_block_status(self, ExecutionStatus.FINISHED)


@contextmanager
def for_(values: Iterable[T]) -> Generator[T, None, None]:
    """For expression to iterate over the values within a code block.

    The equivalent of Python's for loop.

    Arguments:
        values: An iterable.

    Example:
        ```python
       with for_([1, 2, 3]) as x:
            ...
        ```
    """
    with ForExpression(values=values) as x:
        yield cast(T, x)


class ReturnStatement(Block):
    """Return statement for a workflow.

    Sets the active workflow block output to the given `value` and interrupts
    the current workflow execution.

    Arguments:
        value: Value to be set for workflow output.
    """

    def __init__(self, value: Any) -> None:  # noqa: ANN401
        super().__init__(value=value)

    def execute(self, executor: ExecutorState) -> None:
        """Execute the block."""
        executor.set_block_status(self, ExecutionStatus.IN_PROGRESS)
        value = executor.resolve_inputs(self).get("value")
        executor.set_execution_output(value)
        executor.set_block_status(self, ExecutionStatus.FINISHED)
        executor.interrupt()


def return_(value: Any | None = None) -> None:  # noqa: ANN401
    """Return statement of an workflow.

    Sets the active workflow output value to the given `value` and interrupts
    the current workflow execution. Comparative to Python's `return` statement.

    The equivalent of Python's `return` statement.

    Arguments:
        value: Value to be set for workflow output.
    """
    active_ctx = TaskExecutorContext.get_active()
    if isinstance(active_ctx, WorkflowBlockBuilder):
        active_ctx.register(ReturnStatement(value=value))
