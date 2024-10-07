"""A module that defines workflow graph executor."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from laboneq_applications.core.utils import utc_now
from laboneq_applications.workflow import reference
from laboneq_applications.workflow._context import LocalContext
from laboneq_applications.workflow.exceptions import WorkflowError
from laboneq_applications.workflow.options import WorkflowOptions
from laboneq_applications.workflow.recorder import (
    ExecutionRecorder,
    ExecutionRecorderManager,
)
from laboneq_applications.workflow.reference import resolve_to_value

if TYPE_CHECKING:
    from collections.abc import Generator
    from datetime import datetime

    from laboneq_applications.workflow.blocks import Block
    from laboneq_applications.workflow.options_base import BaseOptions
    from laboneq_applications.workflow.result import TaskResult, WorkflowResult


class _ExecutorInterrupt(Exception):  # noqa: N818
    """Executor interrupt signal."""


@dataclass
class ExecutorSettings:
    """A class that defines the settings for the executor.

    Attributes:
        run_until: Execute until a task with given name was executed and exit.
            If `None`, the execution will continue until the end.
    """

    run_until: str | None = None


class ExecutionStatus(Enum):
    """Execution status of an block."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    FINISHED = "finished"
    SKIPPED = "skipped"


class ExecutorState:
    """A class that holds the graph execution state."""

    def __init__(
        self,
        settings: ExecutorSettings | None = None,
    ) -> None:
        self._settings = settings or ExecutorSettings()
        self._recorder_manager = ExecutionRecorderManager()
        self._options = WorkflowOptions()
        self._results: list[WorkflowResult] = []
        # Block variables, either workflow inputs or outputs of any block
        self._block_variables = {}
        self._block_status: dict[Block, ExecutionStatus] = {}
        self._context_depth = 0
        self._start_time: datetime = utc_now()

    @property
    def has_active_context(self) -> bool:
        """Return true if executor has an active context."""
        return self._context_depth != 0

    def get_options(self, name: str) -> BaseOptions | None:
        """Get options by block name."""
        if name in self._options._task_options:
            return self._options._task_options.get(name)
        # TODO: Remove when WorkflowOptions are not required to have
        # task names defined on upper level
        return getattr(self._options, name, None)

    @contextmanager
    def set_active_workflow_settings(
        self, result: WorkflowResult, options: WorkflowOptions | None = None
    ) -> Generator[None]:
        """Set an active workflow settings for the duration of the context.

        Given settings are then used and are available for sub-blocks executed
        within the context.
        """
        self._results.append(result)
        opts_old = self._options
        self._options = options or WorkflowOptions()
        try:
            yield
        finally:
            self._results.pop()
            self._options = opts_old

    def add_workflow_result(self, result: WorkflowResult) -> None:
        """Add executed workflow result.

        If workflow is within another workflow, the result is added
        to the parent workflow, otherwise not action is done.
        """
        # self._results is a list of active workflows and self._results[-1]
        # is the parent of the workflow producing the 'result' and therefore
        # we append it into the parent workflow results.
        # If self._results is empty, the result is from the active workflow itself
        # and nothing should be done.
        if self._results:
            self._results[-1]._tasks.append(result)

    def add_task_result(self, task: TaskResult) -> None:
        """Add executed task result."""
        self._results[-1]._tasks.append(task)

    def set_execution_output(self, output: Any) -> None:  # noqa: ANN401
        """Set an output for the workflow being executed."""
        self._results[-1]._output = output

    @property
    def settings(self) -> ExecutorSettings:
        """Executor settings."""
        return self._settings

    @property
    def block_variables(self) -> dict:
        """Block variables."""
        return self._block_variables

    def set_block_status(self, block: Block, status: ExecutionStatus) -> None:
        """Set block status."""
        # TODO: Move to executor blocks once a proper executor is ready.
        self._block_status[block] = status

    def get_block_status(self, block: Block) -> ExecutionStatus:
        """Get block status."""
        # TODO: Move to executor blocks once a proper executor is ready.
        return self._block_status.get(block, ExecutionStatus.NOT_STARTED)

    def __enter__(self):
        """Enter an execution context.

        When execution context is active, it can be interrupted
        with '.interrupt()'. This will either exit the execution
        or continue the upper context in case of nested context.
        """
        self._context_depth += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        """Exit the execution context."""
        self._context_depth -= 1
        return isinstance(exc_value, _ExecutorInterrupt)

    def interrupt(self) -> None:
        """Interrupt the current active execution context.

        Must be called while an executor context is active, otherwise
        raises an `WorkflowError`.
        """
        if self.has_active_context:
            raise _ExecutorInterrupt
        raise WorkflowError(
            "interrupt() cannot be called outside of active executor context."
        )

    @property
    def recorder(self) -> ExecutionRecorderManager:
        """Execution recorder."""
        return self._recorder_manager

    def add_recorder(self, recorder: ExecutionRecorder) -> None:
        """Add a recorder to the execution.

        Arguments:
            recorder: A recorder that records the execution information.
        """
        self._recorder_manager.add_recorder(recorder)

    def resolve_inputs(self, block: Block) -> dict:
        """Resolve the inputs of the block."""
        inp = {}
        for k, v in block.parameters.items():
            if isinstance(v, reference.Reference):
                value = resolve_to_value(v, self._block_variables)
                inp[k] = reference.unwrap(v, value)
            else:
                inp[k] = v
        return inp

    def set_variable(self, block: object, value: object) -> None:
        """Set the block variable."""
        # TODO: Move to executor blocks once a proper executor is ready.
        self._block_variables[block] = value

    def get_variable(self, block: object) -> Any:  # noqa: ANN401
        """Get block_variable."""
        # TODO: Move to executor blocks once a proper executor is ready.
        return self._block_variables[block]

    def results(self) -> list[WorkflowResult]:
        """Return the results of the execution."""
        return self._results

    @property
    def start_time(self) -> datetime:
        """Return the start time of the execution."""
        return self._start_time


class ExecutorStateContext(LocalContext[ExecutorState]):
    """Context for workflow execution state."""

    _scope = "workflow_executor"


class WorkflowExecutionInfoView:
    """A view to query properties of the workflow execution."""

    def __init__(self, state: ExecutorState) -> None:
        self._state = state

    @property
    def workflows(self) -> list[str]:
        """Return the names of the workflows which are currently executed.

        The list is ordered from the outermost workflow to the
        innermost (active) workflow.
        """
        return [result.name for result in self._state.results()]

    @property
    def start_time(self) -> datetime | None:
        """Return the timestamp of the workflow execution start."""
        return self._state.start_time


def execution_info() -> WorkflowExecutionInfoView | None:
    """Return a view of the workflow information."""
    active_context = ExecutorStateContext.get_active()
    if not active_context:
        return None
    return WorkflowExecutionInfoView(active_context)
