"""Options for workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq_applications.workflow.options_base import (
    BaseOptions,
    option_field,
    options,
)

if TYPE_CHECKING:
    from laboneq_applications.logbook import LogbookStore


@options
class TaskOptions(BaseOptions):
    """Base class for task options."""


@options
class WorkflowOptions(BaseOptions):
    """Base options for a workflow.

    Attributes:
        logstore:
            The logstore to use. Not serialized/deserialized.
            Default: `None`.
        task_options:
            A mapping of sub-task and sub-workflow options.
            A task can have only one unique set of options per workflow.
    """

    logstore: LogbookStore | None = option_field(
        None, description="The logstore to use.", exclude=True, repr=False
    )
    _task_options: dict[str, BaseOptions] = option_field(
        factory=dict, description="task options", alias="_task_options"
    )
