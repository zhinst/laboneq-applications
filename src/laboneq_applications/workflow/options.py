"""Options for workflows."""

from __future__ import annotations

import typing

from pydantic import Field, PrivateAttr

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
    _task_options: dict[str, BaseOptions] = PrivateAttr(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        # Exclude fields from serialization by default
        exclude: typing.ClassVar[set[str]] = {"logbook"}
        arbitrary_types_allowed = True

    def to_dict(self) -> dict:
        """Generate a dictionary representation of the options."""
        data = super().to_dict()
        data["_task_options"] = {
            key: value.to_dict() for key, value in self._task_options.items()
        }
        return data

    def __rich_repr__(self):
        yield from super().__rich_repr__()
        yield "_task_options", self._task_options
