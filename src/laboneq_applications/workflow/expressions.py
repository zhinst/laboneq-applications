"""A module for expressions supported in Workflow."""

from __future__ import annotations

from typing import Any

from laboneq_applications.workflow.block import Block, BlockResult


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
