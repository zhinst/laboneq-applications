"""Workflow if block."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from laboneq_applications.workflow import variable_tracker
from laboneq_applications.workflow.blocks.block import Block
from laboneq_applications.workflow.executor import ExecutionStatus, ExecutorState

if TYPE_CHECKING:
    from collections.abc import Generator


class IFExpression(Block):
    """If expression.

    A block that is executed if a given `condition` is true.

    Arguments:
        condition: A condition that has to be `True` for block to be
            executed.
    """

    def __init__(self, condition: Any) -> None:  # noqa: ANN401
        super().__init__(parameters={"condition": condition})

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


@variable_tracker.track
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
