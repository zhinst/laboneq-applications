"""A block of work in a workflow."""

from __future__ import annotations

import abc
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

from laboneq_applications.workflow._context import (
    TaskExecutor,
    TaskExecutorContext,
)
from laboneq_applications.workflow.engine.promise import (
    PromiseResultNotResolvedError,
    ReferencePromise,
)
from laboneq_applications.workflow.engine.resolver import ArgumentResolver
from laboneq_applications.workflow.exceptions import WorkflowError

if TYPE_CHECKING:
    from laboneq_applications.workflow.engine.block import Block
    from laboneq_applications.workflow.engine.promise import Promise
    from laboneq_applications.workflow.task import _BaseTask


class BlockResult:
    """A class representing block result.

    A collection of results recorded within the block.
    """

    def __init__(self) -> None:
        self._log: dict[str, list] = defaultdict(list)

    @property
    def log(self) -> dict[str, list]:
        """Log of the block."""
        return dict(self._log)

    def merge(self, other: BlockResult) -> None:
        """Merge block results."""
        for key, value in other.log.items():
            self._log[key].extend(value)

    def add_result(self, key: str, result: Any) -> None:  # noqa: ANN401
        """Add result to the log."""
        self._log[key].append(result)


class Block(abc.ABC):
    """A base class for workflow blocks.

    A block can be an individual task or a collection of other blocks.

    Classes inheriting from `Block` must define the following methods:

        - `execute()`: A method that executes the block and it's children defined
            in `Block.body`.

    Arguments:
        *args: Arguments of the block.
        **kwargs: Keyword arguments of the block.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self._body: list[Block] = []
        self._resolver = ArgumentResolver(*self.args, **self.kwargs)

    @property
    def name(self) -> str:
        """Name of the block."""
        return self.__class__.__name__

    @property
    def body(self) -> list[Block]:
        """Body of the block.

        A list of other blocks that are defined within this block.
        """
        return self._body

    def __enter__(self):
        TaskExecutorContext.enter(WorkflowBlockExecutorContext())

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        register = cast(WorkflowBlockExecutorContext, TaskExecutorContext.exit())
        self._body.extend(register.blocks)
        active_ctx = TaskExecutorContext.get_active()
        if isinstance(active_ctx, WorkflowBlockExecutorContext):
            active_ctx.register(self)

    def _run_block(self, block: Block) -> BlockResult:
        """Run a block belonging to this block.

        Argument:
            block: Child block of this block.
        """
        try:
            return block.execute()
        except PromiseResultNotResolvedError as error:
            raise WorkflowError(error) from error

    @abc.abstractmethod
    def execute(self) -> BlockResult:
        """Execute the block.

        Classes implementing this method should run any child block
        via `_run_block()` method.

        Returns:
            Block result.
        """


class TaskBlock(Block):
    """Task block.

    `TaskBlock` is an workflow executor for a task.

    Arguments:
        task: A task this block contains.
        *args: Arguments of the task.
        **kwargs: Keyword arguments of the task.
    """

    def __init__(self, task: _BaseTask, *args: object, **kwargs: object):
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


class WorkflowBlockExecutorContext(TaskExecutor):
    """Workflow context executor for blocks."""

    def __init__(self):
        self._blocks: list[Block] = []

    @property
    def blocks(self) -> list[Block]:
        """Blocks registered within the context."""
        return self._blocks

    def execute_task(
        self,
        task: _BaseTask,
        *args: object,
        **kwargs: object,
    ) -> Promise:
        """Run a task.

        Arguments:
            task: The task instance.
            *args: `task` arguments.
            **kwargs: `task` keyword arguments.
        """
        block = TaskBlock(task, *args, **kwargs)
        self.register(block)
        return block._promise

    def register(self, block: Block) -> None:
        """Register a block."""
        self._blocks.append(block)
