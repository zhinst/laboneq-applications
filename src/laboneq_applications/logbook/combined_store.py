"""Logbook that outputs events to several other logstores."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq_applications.logbook import Logbook, LogbookStore

if TYPE_CHECKING:
    from laboneq_applications.logbook.core import Artifact
    from laboneq_applications.workflow.engine.core import Workflow, WorkflowResult
    from laboneq_applications.workflow.task import Task


class CombinedStore(LogbookStore):
    """A logging store that writes logs to several other logging stores."""

    def __init__(self, stores: list[LogbookStore]):
        self._stores = stores.copy()

    def create_logbook(self, workflow: Workflow) -> Logbook:
        """Create a new logbook for the given workflow."""
        logbooks = [store.create_logbook(workflow) for store in self._stores]
        return CombinedLogbook(logbooks)


class CombinedLogbook(Logbook):
    """A logbook that logs events to several other logbooks."""

    def __init__(
        self,
        logbooks: list[Logbook],
    ):
        self._logbooks = logbooks

    def on_start(self, workflow_result: WorkflowResult) -> None:
        """Called when the workflow execution starts."""
        for logbook in self._logbooks:
            logbook.on_start(workflow_result)

    def on_end(self, workflow_result: WorkflowResult) -> None:
        """Called when the workflow execution ends."""
        for logbook in self._logbooks:
            logbook.on_end(workflow_result)

    def on_error(
        self,
        workflow_result: WorkflowResult,
        error: Exception,
    ) -> None:
        """Called when the workflow raises an exception."""
        for logbook in self._logbooks:
            logbook.on_error(workflow_result, error)

    def on_task_start(self, task: Task) -> None:
        """Called when a task begins execution."""
        for logbook in self._logbooks:
            logbook.on_task_start(task)

    def on_task_end(self, task: Task) -> None:
        """Called when a task ends execution."""
        for logbook in self._logbooks:
            logbook.on_task_end(task)

    def on_task_error(
        self,
        task: Task,
        error: Exception,
    ) -> None:
        """Called when a task raises an exception."""
        for logbook in self._logbooks:
            logbook.on_task_error(task, error)

    def comment(self, message: str) -> None:
        """Called to leave a comment."""
        for logbook in self._logbooks:
            logbook.comment(message)

    def save(self, artifact: Artifact) -> None:
        """Called to save an artifact."""
        for logbook in self._logbooks:
            logbook.save(artifact)
