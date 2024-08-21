"""A module for expressions supported in Workflow."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from laboneq_applications.workflow._context import TaskExecutorContext
from laboneq_applications.workflow.engine.block import (
    Block,
    WorkflowBlockBuilder,
)
from laboneq_applications.workflow.engine.reference import Reference

if TYPE_CHECKING:
    from collections.abc import Iterable

    from laboneq_applications.workflow.engine.executor import ExecutorState


class IFExpression(Block):
    """IF expression.

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
            for block in self.body:
                block.execute(executor)


T = TypeVar("T")


class ForExpression(Block, Generic[T]):
    """For expression.

    A block that iterates workflow blocks over the given values.

    Arguments:
        values: An iterable.
            Iterable can contain workflow objects.
    """

    def __init__(self, values: Iterable[T]) -> None:
        super().__init__(values=values)
        self._ref = Reference(self)

    def __enter__(self) -> T:
        """Enter the loop context.

        Returns:
            Individual values of the given iterable.
        """
        super().__enter__()
        return self._ref

    def execute(self, executor: ExecutorState) -> None:
        """Execute the block."""
        vals = executor.resolve_inputs(self)["values"]
        for val in vals:
            executor.set_state(self, val)
            for block in self.body:
                block.execute(executor)


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
        value = executor.resolve_inputs(self).get("value")
        executor.set_execution_output(value)
        executor.interrupt()


def return_(value: Any | None = None) -> None:  # noqa: ANN401
    """Return statement of an workflow.

    Sets the active workflow output value to the given `value` and interrupts
    the current workflow execution. Comparative to Python's `return` statement.

    Arguments:
        value: Value to be set for workflow output.
    """
    active_ctx = TaskExecutorContext.get_active()
    if isinstance(active_ctx, WorkflowBlockBuilder):
        active_ctx.register(ReturnStatement(value=value))
