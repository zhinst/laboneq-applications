"""Logbook that outputs events to a Python logger."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from laboneq_applications.logbook import Logbook, LogbookStore

if TYPE_CHECKING:
    from laboneq_applications.workflow.engine.core import Workflow
    from laboneq_applications.workflow.task import Task


class LoggingStore(LogbookStore):
    """A logging store that writes logs to a Python logger."""

    DEFAULT_LOGGER = logging.getLogger("laboneq_applications.workflow")

    def __init__(self, logger: logging.Logger | None = None):
        if logger is None:
            logger = self.DEFAULT_LOGGER
        self._logger = logger

    def create_logbook(self, workflow: Workflow) -> Logbook:
        """Create a new logbook for the given workflow."""
        return LoggingLogbook(workflow, self._logger)


class LoggingLogbook(Logbook):
    """A logbook that logs events to a Python logger."""

    def __init__(self, workflow: Workflow, logger: logging.Logger):
        self._workflow = workflow
        self._logger = logger

    def on_start(self) -> None:
        """Called when the workflow execution starts."""
        self._logger.info("Workflow execution started")

    def on_end(self) -> None:
        """Called when the workflow execution ends."""
        self._logger.info("Workflow execution ended")

    def on_error(self, error: Exception) -> None:
        """Called when the workflow raises an exception."""
        self._logger.error("Workflow execution failed with: %r", error)

    def on_task_start(self, task: Task) -> None:
        """Called when a task begins execution."""
        self._logger.info("Task %s started", task.name)

    def on_task_end(self, task: Task) -> None:
        """Called when a task ends execution."""
        self._logger.info("Task %s ended", task.name)

    def on_task_error(self, task: Task, error: Exception) -> None:
        """Called when a task raises an exception."""
        self._logger.error("Task %s failed with: %r", task.name, error)

    def comment(self, message: str) -> None:
        """Called to leave a comment."""
        self._logger.info(message)

    def save_artifact(
        self,
        task: Task,
        name: str,
        artifact: object,
        metadata: dict[str, object] | None = None,
        serialization_options: dict[str, object] | None = None,
    ) -> str:
        """Called to save an artifact."""
        self._logger.info(
            "Task %s saving artifact %s of type '%s':"
            " [metadata: %r, serialization_options: %r]",
            task.name,
            name,
            artifact.type_,
            metadata,
            serialization_options,
        )
        # TODO: Return or not?
