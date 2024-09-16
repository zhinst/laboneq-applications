"""A workflow graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

from laboneq_applications.workflow.blocks.task_block import TaskBlock
from laboneq_applications.workflow.blocks.workflow_block import WorkflowBlock

if TYPE_CHECKING:
    from laboneq_applications.workflow.options import (
        WorkflowOptions,
    )


class WorkflowGraph:
    """Workflow graph.

    A graph contains blocks which defines the flow of the workflow.

    Arguments:
        root: Root block of the Workflow.
    """

    def __init__(self, root: WorkflowBlock) -> None:
        self._root = root

    @property
    def root(self) -> WorkflowBlock:
        """Root workflow block the graph."""
        return self._root

    @property
    def name(self) -> str:
        """Name of the graph."""
        return self._root.name

    def create_options(self) -> WorkflowOptions:
        """Create options for the graph."""
        return self._root.create_options()

    @property
    def options_type(self) -> type[WorkflowOptions]:
        """Type of graph options."""
        return self._root.options_type

    @property
    def tasks(self) -> list[TaskBlock]:
        """A flat list of individual tasks within the graph."""
        return cast(list[TaskBlock], self._root.find(by=TaskBlock, recursive=True))

    @classmethod
    def from_callable(cls, func: Callable, name: str | None = None) -> WorkflowGraph:
        """Create the graph from a callable."""
        return cls(WorkflowBlock.from_callable(name or func.__name__, func))
