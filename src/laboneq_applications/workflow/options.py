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


def _logstore_converter(x: object) -> object:
    # To avoid circular import
    from laboneq_applications.logbook import LogbookStore

    return [x] if isinstance(x, LogbookStore) else x


@options
class WorkflowOptions(BaseOptions):
    """Base options for a workflow.

    Attributes:
        logstore:
            The logstore to use for the particular workflow.
            If left as `None`, uses the currently activated logstores.
            The field accepts either a single logstore or a list of logstores.
            Empty list results to no logging at all.

            This field is not serialized/deserialized.
        task_options:
            A mapping of sub-task and sub-workflow options.
            A task can have only one unique set of options per workflow.
    """

    logstore: list[LogbookStore] | LogbookStore | None = option_field(
        None,
        description="The logstores to use.",
        exclude=True,
        repr=False,
        converter=_logstore_converter,
    )
    _task_options: dict[str, BaseOptions] = option_field(
        factory=dict, description="task options", alias="_task_options"
    )
