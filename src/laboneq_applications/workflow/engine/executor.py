"""A module that defines workflow graph executor."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from laboneq_applications.workflow.engine.reference import Reference
from laboneq_applications.workflow.exceptions import WorkflowError
from laboneq_applications.workflow.options import WorkflowOptions
from laboneq_applications.workflow.recorder import (
    ExecutionRecorder,
    ExecutionRecorderManager,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from laboneq_applications.workflow.engine.block import Block
    from laboneq_applications.workflow.engine.core import WorkflowResult
    from laboneq_applications.workflow.task import Task


class _ExecutorInterrupt(Exception):  # noqa: N818
    """Executor interrupt signal."""


class ExecutorState:
    """A class that holds the graph execution state."""

    def __init__(
        self,
        options: WorkflowOptions | None = None,
    ) -> None:
        if options is None:
            options = WorkflowOptions()
        self._graph_variable_states = {}
        self._recorder_manager = ExecutionRecorderManager()
        self.options = options
        self._results: list[WorkflowResult] = []

    @contextmanager
    def set_active_result(self, result: WorkflowResult) -> Generator[None]:
        """Set an active result object for the duration of the context."""
        self._results.append(result)
        yield
        self._results.pop()

    def add_task_result(self, task: Task) -> None:
        """Add executed task result."""
        self._results[-1].add_task(task)

    def set_execution_output(self, output: Any) -> None:  # noqa: ANN401
        """Set an output for the workflow being executed."""
        self._results[-1]._output = output

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
