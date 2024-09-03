"""A module that defines workflow graph executor."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from laboneq_applications.workflow import reference
from laboneq_applications.workflow.exceptions import WorkflowError
from laboneq_applications.workflow.options import WorkflowOptions
from laboneq_applications.workflow.recorder import (
    ExecutionRecorder,
    ExecutionRecorderManager,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from laboneq_applications.workflow.engine.block import Block
    from laboneq_applications.workflow.result import WorkflowResult
    from laboneq_applications.workflow.task import TaskResult


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


class ExecutorState:
    """A class that holds the graph execution state."""

    def __init__(
        self,
        options: WorkflowOptions | None = None,
        settings: ExecutorSettings | None = None,
    ) -> None:
        if options is None:
            options = WorkflowOptions()
        self._settings = settings or ExecutorSettings()
        self._graph_variable_states = {}
        self._recorder_manager = ExecutionRecorderManager()
        self._block_states: dict[Block, ExecutionStatus] = {}
        self.options = options
        self._results: list[WorkflowResult] = []

    @contextmanager
    def set_active_workflow_settings(
        self, result: WorkflowResult, options: WorkflowOptions | None = None
    ) -> Generator[None]:
        """Set an active workflow settings for the duration of the context.

        Given settings are then used and are available for sub-blocks executed
        within the context.
        """
        self._results.append(result)
        opts_old = self.options
        self.options = options
        try:
            yield
        finally:
            self._results.pop()
            self.options = opts_old

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
    def states(self) -> dict:
        """States of the graph."""
        return self._graph_variable_states

    def set_block_status(self, block: Block, status: ExecutionStatus) -> None:
        """Set block status."""
        # TODO: Move to executor blocks once a proper executor is ready.
        self._block_states[block] = status

    def get_block_status(self, block: Block) -> ExecutionStatus:
        """Get block status."""
        # TODO: Move to executor blocks once a proper executor is ready.
        return self._block_states.get(block, ExecutionStatus.NOT_STARTED)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        return isinstance(exc_value, _ExecutorInterrupt)

    def interrupt(self) -> None:
        """Interrupt the current workflow block execution."""
        raise _ExecutorInterrupt

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
                try:
                    ref = reference.get_ref(v)
                    value = self._graph_variable_states[ref]
                except KeyError as error:
                    default = reference.get_default(v)
                    if default != reference.notset:
                        value = default
                    else:
                        # Reference was never executed.
                        # TODO: Validate at graph definition time for
                        #       branching statements.
                        raise WorkflowError(
                            f"Result for '{ref}' is not resolved.",
                        ) from error
                inp[k] = reference.unwrap(v, value)
            else:
                inp[k] = v
        return inp

    def set_state(self, block: Block | str, state) -> None:  # noqa: ANN001
        """Set the block state."""
        # TODO: Move to executor blocks once a proper executor is ready.
        self._graph_variable_states[block] = state

    def get_state(self, item: Block | str) -> Any:  # noqa: ANN401
        """Get state of an item."""
        # TODO: Move to executor blocks once a proper executor is ready.
        return self._graph_variable_states[item]
