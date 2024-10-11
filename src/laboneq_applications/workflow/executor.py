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


class _RunTimeIndexer:
    """Runtime indexer for blocks."""

    def __init__(self) -> None:
        self._index: list = []

    def push(self, index: object) -> None:
        """Push an active index."""
        self._index.append(index)

    def pop(self) -> object:
        """Pop an active index."""
        return self._index.pop()

    def format(self) -> tuple[object]:
        """Format the index."""
        return tuple(self._index)


class _WorkflowExecutionContext:
    """Workflow execution context.

    Independent scope for each workflow that is being executed.
    """

    def __init__(
        self, result: WorkflowResult, options: WorkflowOptions, indexer: _RunTimeIndexer
    ) -> None:
        self.result = result
        self.options = options
        self.indexer = indexer


class ExecutorState:
    """A class that holds the graph execution state."""

    def __init__(
        self,
        settings: ExecutorSettings | None = None,
    ) -> None:
        self._settings = settings or ExecutorSettings()
        self._recorder_manager = ExecutionRecorderManager()
        self._workflow_contexts: list[_WorkflowExecutionContext] = []
        # Block variables, either workflow inputs or outputs of any block
        self._block_variables = {}
        self._block_status: dict[Block, ExecutionStatus] = {}
        self._context_depth = 0
        self._start_time: datetime = utc_now()

    @property
    def has_active_context(self) -> bool:
        """Return true if executor has an active context."""
        return len(self._workflow_contexts) != 0

    @contextmanager
    def scoped_index(self, index: object) -> Generator[None, None, None]:
        """Add a scoped index for the given context.

        Must be called within an active workflow.
        """
        self._workflow_contexts[-1].indexer.push(index)
        try:
            yield
        finally:
            self._workflow_contexts[-1].indexer.pop()

    def get_index(self) -> tuple:
        """Get execution index."""
        if not self.has_active_context:
            return ()
        return self._workflow_contexts[-1].indexer.format()

    def get_options(self, name: str) -> BaseOptions | None:
        """Get options by block name."""
        if not self._workflow_contexts:
            return None
        opts = self._workflow_contexts[-1].options
        if name in opts._task_options:
            return opts._task_options.get(name)
        # TODO: Remove when WorkflowOptions are not required to have
        # task names defined on upper level
        return getattr(opts, name, None)

    @contextmanager
    def enter_workflow(
        self, result: WorkflowResult, options: WorkflowOptions | None = None
    ) -> Generator[None]:
        """Enter an workflow execution context.

        Sets given settings for the duration of the context.

        Given settings are then used and are available for sub-blocks executed
        within the context.

        When execution context is active, it can be interrupted
        with '.interrupt()'. This will either exit the execution
        or continue the upper context in case of nested context.
        """
        ctx = _WorkflowExecutionContext(
            result=result,
            options=options or WorkflowOptions(),
            indexer=_RunTimeIndexer(),
        )
        self._workflow_contexts.append(ctx)
        try:
            yield
        except _ExecutorInterrupt:
            return
        finally:
            self._workflow_contexts.pop()

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
        if self.has_active_context:
            self._workflow_contexts[-1].result._tasks.append(result)

    def add_task_result(self, task: TaskResult) -> None:
        """Add executed task result."""
        self._workflow_contexts[-1].result._tasks.append(task)

    def set_execution_output(self, output: Any) -> None:  # noqa: ANN401
        """Set an output for the workflow being executed."""
        self._workflow_contexts[-1].result._output = output

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
        return [x.result for x in self._workflow_contexts]

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
        # We store the workflow names and start time here because the
        # execution state is (highly) mutable.
        self._workflows = [result.name for result in state.results()]
        self._start_time = state.start_time

    @property
    def workflows(self) -> list[str]:
        """Return the names of the workflows which are currently executed.

        The list is ordered from the outermost workflow to the
        innermost (active) workflow.
        """
        return self._workflows

    @property
    def start_time(self) -> str | None:
        """Return the timestamp of the workflow execution start."""
        return self._start_time


def execution_info() -> WorkflowExecutionInfoView | None:
    """Return a view of the workflow information."""
    active_context = ExecutorStateContext.get_active()
    if not active_context:
        return None
    return WorkflowExecutionInfoView(active_context)
