"""A module that defines workflow graph executor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from laboneq_applications.logbook import LoggingStore
from laboneq_applications.workflow.engine.options import WorkflowOptions
from laboneq_applications.workflow.engine.reference import Reference
from laboneq_applications.workflow.exceptions import WorkflowError

if TYPE_CHECKING:
    from laboneq_applications.common.logbooks import Logbook
    from laboneq_applications.workflow.engine.block import Block
    from laboneq_applications.workflow.task import Task


class ResultHander(Protocol):
    """A result protocol for recording results."""

    def on_task_end(self, task: Task) -> None:
        """Add a task result."""


class ExecutorState:
    """A class that holds the graph execution state."""

    def __init__(self, logbook: Logbook | None = None) -> None:
        # TODO: require logbook here or create a better dummy
        if logbook is None:
            logstore = LoggingStore()
            logbook = logstore.create_logbook(None)  # TODO: remove hack

        self._logbook = logbook
        self._graph_variable_states = {}
        self._result_handler: ResultHander | None = None
        self.options: WorkflowOptions = WorkflowOptions()

    @property
    def states(self) -> dict:
        """States of the graph."""
        return self._graph_variable_states

    @property
    def result_handler(self) -> ResultHander | None:
        """Result handler for the executor."""
        return self._result_handler

    def set_result_callback(self, result_handler: ResultHander) -> None:
        """Set a result callback.

        Arguments:
            result_handler: A result handler that receives executed task information.
        """
        self._result_handler = result_handler

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
