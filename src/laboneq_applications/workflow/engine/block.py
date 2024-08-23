"""A block of work in a workflow."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, cast

from laboneq_applications.core import now
from laboneq_applications.workflow import _utils
from laboneq_applications.workflow._context import (
    TaskExecutor,
    TaskExecutorContext,
)
from laboneq_applications.workflow.engine.reference import (
    Reference,
)
from laboneq_applications.workflow.task import Task

if TYPE_CHECKING:
    from collections.abc import Iterable

    from laboneq_applications.workflow.engine.block import Block
    from laboneq_applications.workflow.engine.executor import ExecutorState
    from laboneq_applications.workflow.task import task_


class Block(abc.ABC):
    """A base class for workflow blocks.

    A block can be an individual task or a collection of other blocks.

    Classes inheriting from `Block` must define the following methods:

        - `execute()`: A method that executes the block and it's children defined
            in `Block.body`.

    Arguments:
        **parameters: Input parameters of the block.
    """

    def __init__(self, **parameters) -> None:
        self._parameters = parameters
        self._body: list[Block] = []

    @property
    def parameters(self) -> dict:
        """Input parameters of the block."""
        return self._parameters

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

    def extend(self, blocks: Block | Iterable[Block]) -> None:
        """Extend the body of the block."""
        if isinstance(blocks, Block):
            self._body.append(blocks)
        else:
            self._body.extend(blocks)

    def find(
        self,
        by: type[Block],
        *,
        recursive: bool = False,
    ) -> list[Block]:
        """Search blocks within this block.

        Arguments:
            by: Block type to be searched.
            recursive: Searches recursively and returns a flat list of all
                the results.

        Returns:
            List of blocks that matches the search criteria.
            Empty list if no matches are found.
        """
        if not recursive:
            return [t for t in self.body if isinstance(t, by)]
        objs = []
        for x in self.body:
            if isinstance(x, by):
                objs.append(x)
            objs.extend(x.find(by=by, recursive=True))
        return objs

    def __enter__(self):
        TaskExecutorContext.enter(WorkflowBlockBuilder())

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        register = cast(WorkflowBlockBuilder, TaskExecutorContext.exit())
        self.extend(register.blocks)
        active_ctx = TaskExecutorContext.get_active()
        if isinstance(active_ctx, WorkflowBlockBuilder):
            active_ctx.register(self)

    @abc.abstractmethod
    def execute(self, executor: ExecutorState) -> None:
        """Execute the block."""


class TaskBlock(Block):
    """Task block.

    `TaskBlock` is an workflow executor for a task.

    Arguments:
        task: A task this block contains.
        **parameters: Input parameters of the task.
    """

    def __init__(self, task: task_, **parameters):
        super().__init__(**parameters)
        self.task = task
        # TODO: Should this be by object ID? Still bound to the object
        self._ref = Reference(self)

    @property
    def src(self) -> str:
        """Source code of the task."""
        return self.task.src

    @property
    def name(self) -> str:
        """Name of the task."""
        return self.task.name

    def execute(self, executor: ExecutorState) -> None:
        """Execute the task."""
        params = {}
        if self.parameters:
            params = executor.resolve_inputs(self)
            if (
                self.task.has_opts
                and executor.options
                and params.get("options") is None
            ):
                task_opts = getattr(executor.options, self.name, None)
                if task_opts:
                    params["options"] = task_opts

        task = Task(
            task=self.task,
            output=None,
            input=_utils.create_argument_map(self.task.func, **params),
        )

        task._start_time = now()
        executor.recorder.on_task_start(task)
        try:
            task._output = self.task.func(**params)
            task._end_time = now()
        except Exception as error:
            task._end_time = now()
            executor.recorder.on_task_error(task, error)
            error._logged_by_task = True  # TODO: Nicer mechanism
            raise
        finally:
            executor.add_task_result(task)
            executor.recorder.on_task_end(task)
        executor.set_state(self, task.output)

    def __repr__(self):
        return f"TaskBlock(task={self.task}, parameters={self.parameters})"


class WorkflowBlockBuilder(TaskExecutor):
    """Workflow block builder."""

    def __init__(self):
        self._blocks: list[Block] = []

    @property
    def blocks(self) -> list[Block]:
        """Workflow blocks."""
        return self._blocks

    def execute_task(
        self,
        task: task_,
        *args: object,
        **kwargs: object,
    ) -> Reference:
        """Run a task.

        Arguments:
            task: The task instance.
            *args: `task` arguments.
            **kwargs: `task` keyword arguments.
        """
        block = TaskBlock(
            task,
            **_utils.create_argument_map(task.func, *args, **kwargs),
        )
        self.register(block)
        return block._ref

    def register(self, block: Block) -> None:
        """Register a block."""
        self._blocks.append(block)
