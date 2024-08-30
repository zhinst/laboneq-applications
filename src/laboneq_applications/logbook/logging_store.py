"""Logbook that outputs events to a Python logger."""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from laboneq_applications.core import now
from laboneq_applications.logbook import Logbook, LogbookStore, format_time

if TYPE_CHECKING:
    from typing import Callable

    from laboneq_applications.workflow import Workflow, WorkflowResult
    from laboneq_applications.workflow.recorder import Artifact
    from laboneq_applications.workflow.task import TaskResult


class LoggingStore(LogbookStore):
    """A logging store that writes logs to a Python logger."""

    # Giving the logger a name that starts with "laboneq." means
    # that it is configured via LabOne Q's default logging configuration.
    DEFAULT_LOGGER = logging.getLogger("laboneq.applications")

    def __init__(self, logger: logging.Logger | None = None, *, rich: bool = True):
        if logger is None:
            logger = self.DEFAULT_LOGGER
        self._logger = logger
        self._rich = rich

    def create_logbook(self, workflow: Workflow) -> Logbook:
        """Create a new logbook for the given workflow."""
        return LoggingLogbook(workflow, self._logger, rich=self._rich)


class LoggingLogbook(Logbook):
    """A logbook that logs events to a Python logger."""

    def __init__(
        self,
        workflow: Workflow,
        logger: logging.Logger,
        *,
        rich: bool = True,
    ):
        self._workflow = workflow
        self._logger = logger
        self._rich = rich

    def _log_rich(self, log: Callable, obj: object) -> None:
        """Log a Rich object to the given Python logger."""
        with io.StringIO() as buf:
            console = Console(file=buf, force_jupyter=False, width=80)
            console.print(obj)
            lines = buf.getvalue().splitlines()

        for line in lines:
            log(line)

    def _log_in_rich_panel(self, log: Callable, msg: str, *args) -> None:
        """Log a message in a Rich panel if required."""
        if not self._rich:
            log(msg, *args)
            return

        txt = msg % args
        panel = Panel(txt, box=box.HORIZONTALS, style="bold")
        self._log_rich(log, panel)

    def _log_in_rich_bold(self, log: Callable, msg: str, *args) -> None:
        """Log a message in Rich bold style."""
        if not self._rich:
            log(msg, *args)
            return

        txt = msg % args
        rich_txt = Text(txt, style="bold")
        self._log_rich(log, rich_txt)

    def on_start(self, workflow_result: WorkflowResult) -> None:
        """Called when the workflow execution starts."""
        self._log_in_rich_panel(
            self._logger.info,
            "Workflow %r: execution started at %s",
            workflow_result.name,
            format_time(now(workflow_result.start_time)),
        )

    def on_end(self, workflow_result: WorkflowResult) -> None:
        """Called when the workflow execution ends."""
        self._log_in_rich_panel(
            self._logger.info,
            "Workflow %r: execution ended at %s",
            workflow_result.name,
            format_time(now(workflow_result.end_time)),
        )

    def on_error(self, workflow_result: WorkflowResult, error: Exception) -> None:
        """Called when the workflow raises an exception."""
        self._log_in_rich_bold(
            self._logger.error,
            "Workflow %r: execution failed at %s with: %r",
            workflow_result.name,
            format_time(now(workflow_result.end_time)),
            error,
        )

    def on_task_start(self, task: TaskResult) -> None:
        """Called when a task begins execution."""
        self._log_in_rich_bold(
            self._logger.info,
            "Task %r: started at %s",
            task.name,
            format_time(now(task.start_time)),
        )

    def on_task_end(self, task: TaskResult) -> None:
        """Called when a task ends execution."""
        self._log_in_rich_bold(
            self._logger.info,
            "Task %r: ended at %s",
            task.name,
            format_time(now(task.end_time)),
        )

    def on_task_error(
        self,
        task: TaskResult,
        error: Exception,
    ) -> None:
        """Called when a task raises an exception."""
        self._log_in_rich_bold(
            self._logger.error,
            "Task %r: failed at %s with: %r",
            task.name,
            format_time(now(task.end_time)),
            error,
        )

    def comment(self, message: str) -> None:
        """Called to leave a comment."""
        self._log_in_rich_bold(
            self._logger.info,
            "Comment: %s",
            message,
        )

    def save(self, artifact: Artifact) -> None:
        """Called to save an artifact."""
        self._log_in_rich_bold(
            self._logger.info,
            "Artifact: %r of type %r logged at %s",
            artifact.name,
            type(artifact.obj).__name__,
            format_time(now(artifact.timestamp)),
        )
