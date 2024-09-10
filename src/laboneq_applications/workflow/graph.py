"""A workflow graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

from laboneq_applications.workflow.blocks.task_block import TaskBlock
from laboneq_applications.workflow.blocks.workflow_block import WorkflowBlock

if TYPE_CHECKING:
    from laboneq_applications.workflow.executor import ExecutorState
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

    def validate_input(self, **kwargs: object) -> None:
        """Validate input parameters of the graph.

        Raises:
            TypeError: `options`-parameter is of wrong type.
        """
        if "options" in kwargs:
            opt_param = kwargs["options"]
            if opt_param is not None and not isinstance(
                opt_param,
                (self._root.options_type, dict),
            ):
                msg = (
                    "Workflow input options must be of "
                    f"type '{self._root.options_type.__name__}', 'dict' or 'None'"
                )
                raise TypeError(msg)

    def execute(self, executor: ExecutorState, **kwargs: object) -> None:
        """Execute the graph.

        Arguments:
            executor: Block executor.
            **kwargs: Input parameters of the workflow.
        """
        self._root.set_params(executor, **kwargs)
        self._root.execute(executor)
