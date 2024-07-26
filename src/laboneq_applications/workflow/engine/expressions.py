"""A module for expressions supported in Workflow."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from laboneq_applications.workflow.engine.block import Block
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
