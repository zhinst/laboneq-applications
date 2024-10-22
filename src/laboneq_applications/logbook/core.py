"""Basic logbook classes."""

from __future__ import annotations

import abc
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from laboneq_applications.workflow.recorder import ExecutionRecorder

if TYPE_CHECKING:
    from laboneq_applications.workflow import Workflow


class LogbookStore(abc.ABC):
    """Protocol for storing a collection of records of workflow execution."""

    @abc.abstractmethod
    def create_logbook(self, workflow: Workflow, start_time: datetime) -> Logbook:
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


class Logbook(ExecutionRecorder):
    """Protocol for storing the record of a single workflow execution."""


def format_time(time: datetime) -> str:
    """Format a datetime object as a string."""
    return time.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%fZ")
