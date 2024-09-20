"""Workflow for block."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, TypeVar, cast

from laboneq_applications.workflow.blocks.block import Block
from laboneq_applications.workflow.executor import ExecutionStatus, ExecutorState
from laboneq_applications.workflow.reference import Reference

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable


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
        super().__init__(parameters={"values": values})
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
                executor.set_variable(self, val)
                for block in self.body:
                    block.execute(executor)
        finally:
            executor.settings.run_until = run_until
        executor.set_block_status(self, ExecutionStatus.FINISHED)


T = TypeVar("T")


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
