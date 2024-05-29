"""Tasks used within Workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq_applications.workflow.engine.block import Block, BlockResult
from laboneq_applications.workflow.engine.promise import ReferencePromise

if TYPE_CHECKING:
    from laboneq_applications.workflow.task import Task


class TaskBlock(Block):
    """Task block.

    `TaskBlock` is an workflow executor for a task.

    Arguments:
        task: A task this block contains.
        *args: Arguments of the task.
        **kwargs: Keyword arguments of the task.
    """

    def __init__(self, task: Task, *args: object, **kwargs: object):
        super().__init__(*args, **kwargs)
        self._promise = ReferencePromise(self)
        self.task = task

    def __repr__(self):
        return repr(self.task)

    @property
    def src(self) -> str:
        """Source code of the task."""
        return self.task.src

    @property
    def name(self) -> str:
        """Name of the task."""
        return self.task.name

    def execute(self) -> BlockResult:
        """Execute the task.

        Returns:
            Task block result.
        """
        args, kwargs = self._resolver.resolve()
        result = self.task.run(*args, **kwargs)
        self._promise.set_result(result)
        r = BlockResult()
        r.add_result(self.name, result)
        return r
