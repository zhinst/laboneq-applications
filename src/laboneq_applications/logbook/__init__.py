"""Logbooks for recording the operations of workflows."""

__all__ = [
    "LogbookStore",
    "Logbook",
    "FolderStore",
    "LoggingStore",
    "active_logbook_store",
    "comment",
    "format_time",
    "save_artifact",
    "DEFAULT_LOGGING_STORE",
]


from .core import (
    Logbook,
    LogbookStore,
    active_logbook_store,
    comment,
    format_time,
    save_artifact,
)
from .folder_store import FolderStore
from .logging_store import LoggingStore

# Add the default logging store:
DEFAULT_LOGGING_STORE = LoggingStore()
DEFAULT_LOGGING_STORE.activate()
