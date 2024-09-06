"""Options for workflows."""

from __future__ import annotations

import typing

from pydantic import Field

from laboneq_applications.logbook import LogbookStore  # noqa: TCH001
from laboneq_applications.workflow.options_base import BaseOptions


class TaskOptions(BaseOptions):
    """Base class for task options."""


class WorkflowOptions(BaseOptions):
    """Base options for a workflow.

    Attributes:
        logbook:
            The logbook to use. Not serialized/deserialized.
            Default: `None`.
        task_options:
            A mapping of sub-task and sub-workflow options.
            A task can have only one unique set of options per workflow.
    """

    logstore: LogbookStore | None = Field(default=None, repr=False, exclude=True)
    task_options: dict[str, BaseOptions] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        # Exclude fields from serialization by default
        exclude: typing.ClassVar[set[str]] = {"logbook"}
        arbitrary_types_allowed = True
