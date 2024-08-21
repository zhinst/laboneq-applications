"""Workflow recorder that records different events during execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from laboneq_applications.core import now

if TYPE_CHECKING:
    from laboneq_applications.typing import SimpleDict
    from laboneq_applications.workflow.engine.core import WorkflowResult
    from laboneq_applications.workflow.task import Task


class Artifact:
    """An artifact to record.

    An artifact consists of a Python object that a workflow wishes to
    record plus the additional information required to store and
    reference it.

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


class ExecutionRecorder(Protocol):
    """A class that defines interface for an execution recorder.

    The recorder provides an interface to record specific actions
    during the execution.
    """

    def on_start(self, workflow_result: WorkflowResult) -> None:
        """Called when the workflow execution starts."""

    def on_end(self, workflow_result: WorkflowResult) -> None:
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


class ExecutionRecorderManager(ExecutionRecorder):
    """A class that manages multiple execution recorders."""

    def __init__(self) -> None:
        self._recorders: list[ExecutionRecorder] = []

    def add_recorder(self, recorder: ExecutionRecorder) -> None:
        """Add a recorder to the execution.

        Arguments:
            recorder: A recorder that records the execution information.
        """
        self._recorders.append(recorder)

    def on_start(self, workflow_result: WorkflowResult) -> None:
        """Called when the workflow execution starts."""
        for recorder in self._recorders:
            recorder.on_start(workflow_result)

    def on_end(self, workflow_result: WorkflowResult) -> None:
        """Called when the workflow execution ends."""
        for recorder in self._recorders:
            recorder.on_end(workflow_result)

    def on_error(
        self,
        workflow_result: WorkflowResult,
        error: Exception,
    ) -> None:
        """Called when the workflow raises an exception."""
        for recorder in self._recorders:
            recorder.on_error(workflow_result, error)

    def on_task_start(
        self,
        task: Task,
    ) -> None:
        """Called when a task begins execution."""
        for recorder in self._recorders:
            recorder.on_task_start(task)

    def on_task_end(self, task: Task) -> None:
        """Add a task result."""
        for recorder in self._recorders:
            recorder.on_task_end(task)

    def on_task_error(
        self,
        task: Task,
        error: Exception,
    ) -> None:
        """Called when a task raises an exception."""
        for recorder in self._recorders:
            recorder.on_task_error(task, error)

    def comment(
        self,
        message: str,
    ) -> None:
        """Called to leave a comment."""
        for recorder in self._recorders:
            recorder.comment(message)

    def save(
        self,
        artifact: Artifact,
    ) -> None:
        """Called to record an artifact.

        Arguments:
            artifact:
                The artifact to be saved.
        """
        for recorder in self._recorders:
            recorder.save(artifact)


def comment(message: str) -> None:
    """Add a comment to the current workflow logbook.

    Arguments:
        message:
            The comment to record.
    """
    from laboneq_applications.workflow import _context

    ctx = _context.ExecutorStateContext.get_active()
    if ctx is not None:
        ctx.recorder.comment(message)
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
    """
    from laboneq_applications.workflow import _context

    ctx = _context.ExecutorStateContext.get_active()
    if ctx is not None:
        artifact = Artifact(name, artifact, metadata=metadata, options=options)
        ctx.recorder.save(artifact)
    else:
        raise RuntimeError(
            "Workflow artifact saving is currently not supported outside of tasks.",
        )
