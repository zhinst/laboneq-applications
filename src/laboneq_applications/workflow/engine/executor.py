"""A module that defines workflow graph executor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from laboneq_applications.logbook import LoggingStore
from laboneq_applications.workflow.engine.reference import Reference
from laboneq_applications.workflow.exceptions import WorkflowError
from laboneq_applications.workflow.options import WorkflowOptions

if TYPE_CHECKING:
    from laboneq_applications.logbook import Logbook
    from laboneq_applications.workflow.engine.block import Block
    from laboneq_applications.workflow.task import Task


class ExecutionRecorder(Protocol):
    """A class that defines interface for an execution recorder.

    The recorder provides an interface to record specific actions
    during the execution.
    """

    def on_task_end(self, task: Task) -> None:
        """Record a task result."""

    def on_workflow_end(self, result: Any) -> None:  # noqa: ANN401
        """Record the workflow result."""


class _ExecutionRecorderManager(ExecutionRecorder):
    """A class that manages multiple execution recorders."""

    def __init__(self) -> None:
        self._recorders: list[ExecutionRecorder] = []

    def add_recorder(self, recorder: ExecutionRecorder) -> None:
        """Add a recorder to the execution.

        Arguments:
            recorder: A recorder that records the execution information.
        """
        self._recorders.append(recorder)

    def on_task_end(self, task: Task) -> None:
        """Record a task result."""
        for recorder in self._recorders:
            recorder.on_task_end(task)

    def on_workflow_end(self, result: Any) -> None:  # noqa: ANN401
        """Record the workflow result."""
        for recorder in self._recorders:
            recorder.on_workflow_end(result)


class _ExecutorInterrupt(Exception):  # noqa: N818
    """Executor interrupt signal."""


class ExecutorState:
    """A class that holds the graph execution state."""

    def __init__(
        self,
        logbook: Logbook | None = None,
        options: WorkflowOptions | None = None,
    ) -> None:
        if logbook is None:
            logstore = LoggingStore()
            logbook = logstore.create_logbook("unknown")
        if options is None:
            options = WorkflowOptions()
        self._logbook = logbook
        self._graph_variable_states = {}
        self._recorder_manager = _ExecutionRecorderManager()
        self.options = options

    @property
    def states(self) -> dict:
        """States of the graph."""
        return self._graph_variable_states

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        return isinstance(exc_value, _ExecutorInterrupt)

    def interrupt(self) -> None:
        """Interrupt the current workflow block execution."""
        raise _ExecutorInterrupt

    @property
    def recorder(self) -> ExecutionRecorder:
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
            if isinstance(v, Reference):
                try:
                    ref = self._graph_variable_states[v.ref]
                except KeyError as error:
                    # Reference was never executed.
                    # TODO: Validate at graph definition time for branching statements.
                    raise WorkflowError(
                        f"Result for '{v.ref}' is not resolved.",
                    ) from error
                inp[k] = v.unwrap(ref)
            else:
                inp[k] = v
        return inp

    def set_state(self, block: Block | str, state) -> None:  # noqa: ANN001
        """Set the block state."""
        self._graph_variable_states[block] = state

    def get_state(self, item: Block | str) -> Any:  # noqa: ANN401
        """Get state of an item."""
        return self._graph_variable_states[item]
