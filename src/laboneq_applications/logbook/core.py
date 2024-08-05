"""Basic logbook classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from laboneq_applications.workflow.engine.core import Workflow
    from laboneq_applications.workflow.task import Task


@runtime_checkable  # required to allow a protocol to be used with Pydantic
class LogbookStore(Protocol):
    """Protocol for storing a collection of records of workflow execution."""

    def create_logbook(self, workflow: Workflow) -> Logbook:
        """Create a logbook for recording a single workflow execution."""


class Logbook(Protocol):
    """Protocol for storing the record of a single worfklow exection."""

    def on_start(
        self,
    ) -> None:
        """Called when the workflow execution starts."""

    def on_end(
        self,
    ) -> None:
        """Called when the workflow execution ends."""

    def on_error(
        self,
        error: Exception,
    ) -> None:
        """Called when the workflow raises an exception."""

    def on_task_start(
        self,
        task: Task,
    ) -> None:
        """Called when a task begins execution."""

    def on_task_end(
        self,
        task: Task,
    ) -> None:
        """Called when a task ends execution."""

    def on_task_error(
        self,
        task: Task,
        error: Exception,
    ) -> None:
        """Called when a task raises an exception."""

    def comment(
        self,
        message: str,
    ) -> None:
        """Called to leave a comment."""

    def save_artifact(
        self,
        task: Task,
        name: str,
        artifact: object,
        metadata: dict[str, object] | None = None,
        # TODO: Better name / think about whether we want serializer options:
        serialization_options: dict[str, object] | None = None,
    ) -> str:
        """Called to save an artifact.

        Parameters:
            artifact: The artifact to be saved.
            name: Name hint for the filename of the artifact. A running number might be
                appended if the name is already taken.
            metadata: Additional metadata for the artifact (optional).
            serialization_options: Serialization options for the artifact (optional).

        Returns:
            The URL or filename to the artifact.
        """
        # TODO: What should be returned when no file is created? E.g. by the
        #       logging logbook?


def comment(message: str) -> None:
    """Add a comment to the current workflow logbook."""
    from laboneq_applications.workflow import _context

    ctx = _context.ExecutorStateContext.get_active()
    if ctx is not None:
        ctx._logbook.comment(message)
