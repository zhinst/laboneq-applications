"""Workflow return block."""

from __future__ import annotations

from typing import Any

from laboneq_applications.workflow import variable_tracker
from laboneq_applications.workflow.blocks.block import Block, BlockBuilderContext
from laboneq_applications.workflow.executor import ExecutionStatus, ExecutorState


class ReturnStatement(Block):
    """Return statement for a workflow.

    Sets the active workflow block output to the given `value` and interrupts
    the current workflow execution.

    Arguments:
        value: Value to be set for workflow output.
    """

    def __init__(self, value: Any) -> None:  # noqa: ANN401
        super().__init__(parameters={"value": value})

    def execute(self, executor: ExecutorState) -> None:
        """Execute the block."""
        executor.set_block_status(self, ExecutionStatus.IN_PROGRESS)
        value = executor.resolve_inputs(self).get("value")
        executor.set_execution_output(value)
        executor.set_block_status(self, ExecutionStatus.FINISHED)
        executor.interrupt()

    def __str__(self):
        return "return_()"


@variable_tracker.track
def return_(value: Any | None = None) -> None:  # noqa: ANN401
    """Return statement of an workflow.

    Sets the active workflow output value to the given `value` and interrupts
    the current workflow execution. Comparative to Python's `return` statement.

    The equivalent of Python's `return` statement.

    Arguments:
        value: Value to be set for workflow output.
    """
    root = BlockBuilderContext.get_active()
    if root:
        root.extend(ReturnStatement(value=value))
