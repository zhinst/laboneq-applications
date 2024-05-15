"""A module for expressions supported in Workflow."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from laboneq_applications.workflow.block import Block, BlockResult
from laboneq_applications.workflow.promise import Promise

if TYPE_CHECKING:
    from collections.abc import Iterable


class IFExpression(Block):
    """IF expression.

    A block that is executed if a given `condition` is true.

    Arguments:
        condition: A condition that has to be `True` for block to be
            executed.
    """

    def __init__(self, condition: Any) -> None:  # noqa: ANN401
        super().__init__(condition)

    def _should_execute(self) -> bool:
        [arg], _ = self._resolver.resolve()
        return bool(arg)

    def execute(self) -> BlockResult:
        """Execute the block."""
        r = BlockResult()
        if self._should_execute():
            for block in self.body:
                r.merge(self._run_block(block))
        return r


T = TypeVar("T")


class ForExpression(Block, Generic[T]):
    """For expression.

    A block that iterates workflow blocks over the given values.

    Arguments:
        values: An iterable.
            Iterable can contain workflow objects.
    """

    def __init__(self, values: Iterable[T]) -> None:
        super().__init__(values)
        self._promise = Promise()

    def __enter__(self) -> T:
        """Enter the loop context.

        Returns:
            Individual values of the given iterable.
        """
        super().__enter__()
        return self._promise

    def execute(self) -> BlockResult:
        """Execute the block."""
        r = BlockResult()
        [vals], _ = self._resolver.resolve()
        for val in vals:
            if isinstance(val, Promise):
                val = val.result()  # noqa: PLW2901
            self._promise.set_result(val)
            for block in self.body:
                r.merge(self._run_block(block))
        return r
