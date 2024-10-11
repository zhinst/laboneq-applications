"""Workflow block for tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq_applications.core import utc_now
from laboneq_applications.core.utils import pygmentize
from laboneq_applications.workflow import _utils
from laboneq_applications.workflow.blocks.block import Block
from laboneq_applications.workflow.executor import ExecutionStatus, ExecutorState
from laboneq_applications.workflow.reference import Reference
from laboneq_applications.workflow.result import TaskResult

if TYPE_CHECKING:
    from laboneq_applications.workflow.options_base import BaseOptions
    from laboneq_applications.workflow.task_wrapper import task_


class TaskBlock(Block):
    """Task block.

    `TaskBlock` is an workflow executor for a task.

    Arguments:
        task: A task this block contains.
        parameters: Input parameters of the task.
    """

    def __init__(self, task: task_, parameters: dict | None = None):
        super().__init__(parameters=parameters)
        self.task = task
        # TODO: Should this be by object ID? Still bound to the object
        self._ref = Reference(self)

    @property
    def ref(self) -> Reference:
        """Reference to the object."""
        return self._ref

    @property
    def options_type(self) -> type[BaseOptions] | None:
        """Type of block options."""
        return self.task._options

    @property
    @pygmentize
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
            if self.options_type and params.get("options") is None:
                params["options"] = executor.get_options(self.name)
        task = TaskResult(
            task=self.task,
            output=None,
            input=_utils.create_argument_map(self.task.func, **params),
            index=executor.get_index(),
        )
        task._start_time = utc_now()
        executor.recorder.on_task_start(task)
        try:
            executor.set_block_status(self, ExecutionStatus.IN_PROGRESS)
            task._output = self.task.func(**params)
            task._end_time = utc_now()
        except Exception as error:
            task._end_time = utc_now()
            executor.recorder.on_task_error(task, error)
            raise
        finally:
            executor.add_task_result(task)
            executor.recorder.on_task_end(task)
            executor.set_block_status(self, ExecutionStatus.FINISHED)
        executor.set_variable(self, task.output)
        if executor.settings.run_until == self.name:
            executor.interrupt()

    def __repr__(self):
        return f"TaskBlock(task={self.task}, parameters={self.parameters})"

    def __str__(self):
        return f"task(name={self.name})"
