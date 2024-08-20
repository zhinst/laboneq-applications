"""Basic logbook classes."""

from __future__ import annotations

import abc
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol

from laboneq_applications.core import now

if TYPE_CHECKING:
    from laboneq_applications.typing import SimpleDict
    from laboneq_applications.workflow.engine.core import Workflow, WorkflowResult
    from laboneq_applications.workflow.task import Task


class Artifact:
    """An artifact to record.

    An artifact consists of a Python object that a workflow wishes to
    record plus the additional information required to store and
    reference it, including the artifact's creation time.

    Arguments:
        name:
            A name hint for the artifact. Logbooks may use this to generate
            meaningful filenames for artifacts when they are saved to disk,
            for example.
        obj:
            The object to be recorded.
        metadata:
            Additional metadata for the artifact (optional).
        options:
            Serialization options for the artifact (optional).
    """

    def __init__(
        self,
        name: str,
        obj: object,
        metadata: SimpleDict | None = None,
        options: SimpleDict | None = None,
    ):
        self.name = name
        self.obj = obj
        self.metadata = metadata or {}
        self.options = options or {}
        self.timestamp = now()


class LogbookStore(abc.ABC):
    """Protocol for storing a collection of records of workflow execution."""

    @abc.abstractmethod
    def create_logbook(self, workflow: Workflow) -> Logbook:
        """Create a logbook for recording a single workflow execution."""

    def activate(self) -> None:
        """Activate this logbook store.

        Workflows write to all active logbook stores by default.
        """
        if self not in _active_logbook_stores:
            _active_logbook_stores.append(self)

    def deactivate(self) -> None:
        """Deactivate this logbook store.

        If this store is not active, this method does nothing.
        """
        if self in _active_logbook_stores:
            _active_logbook_stores.remove(self)


_active_logbook_stores = []


def active_logbook_store() -> LogbookStore | None:
    """Return the active logbook store."""
    from laboneq_applications.logbook.combined_store import CombinedStore

    if not _active_logbook_stores:
        return None
    return CombinedStore(_active_logbook_stores)


class Logbook(Protocol):
    """Protocol for storing the record of a single workflow execution."""

    def on_start(
        self,
        workflow_result: WorkflowResult,
    ) -> None:
        """Called when the workflow execution starts."""

    def on_end(
        self,
        workflow_result: WorkflowResult,
    ) -> None:
        """Called when the workflow execution ends."""

    def on_error(
        self,
        workflow_result: WorkflowResult,
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

    def save(
        self,
        artifact: Artifact,
    ) -> None:
        """Called to record an artifact.

        Arguments:
            artifact:
                The artifact to be saved.
        """


def comment(message: str) -> None:
    """Add a comment to the current workflow logbook.

    Arguments:
        message:
            The comment to record.
    """
    from laboneq_applications.workflow import _context

    ctx = _context.ExecutorStateContext.get_active()
    if ctx is not None:
        ctx._logbook.comment(message)
    else:
        raise RuntimeError(
            "Workflow comments are currently not supported outside of tasks.",
        )


def save_artifact(
    name: str,
    artifact: object,
    *,
    metadata: SimpleDict | None = None,
    options: SimpleDict | None = None,
) -> None:
    """Save an artifact to the current workflow logbook.

    Arguments:
        name:
            A name hint for the artifact. Logbooks may use this to generate
            meaningful filenames for artifacts when they are saved to disk,
            for example.
        artifact:
            The object to be recorded.
        metadata:
            Additional metadata for the artifact (optional).
        options:
            Serialization options for the artifact (optional).

    Returns:
        The URL or filename to the artifact.
    """
    from laboneq_applications.workflow import _context

    ctx = _context.ExecutorStateContext.get_active()
    if ctx is not None:
        artifact = Artifact(name, artifact, metadata=metadata, options=options)
        ctx._logbook.save(artifact)
    else:
        raise RuntimeError(
            "Workflow artifact saving is currently not supported outside of tasks.",
        )


def format_time(time: datetime) -> str:
    """Format a datetime object as a string."""
    return time.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%fZ")
