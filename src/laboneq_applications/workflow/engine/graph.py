"""A workflow graph."""

from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Callable

from laboneq_applications.workflow import exceptions
from laboneq_applications.workflow._context import TaskExecutorContext
from laboneq_applications.workflow.engine.block import (
    Block,
    WorkflowBlockBuilder,
)
from laboneq_applications.workflow.engine.reference import Reference

if TYPE_CHECKING:
    from laboneq_applications.workflow.engine.executor import ExecutorState


class WorkflowBlock(Block):
    """Workflow block."""

    def __init__(self, **parameters: object) -> None:
        super().__init__(**parameters)

    def execute(self, executor: ExecutorState) -> None:
        """Execute the block."""
        for block in self._body:
            block.execute(executor)

    @classmethod
    def from_callable(cls, func: Callable) -> WorkflowBlock:
        """Create the block from a callable."""
        params = {}
        for arg in signature(func).parameters:
            # TODO: Improve reference system to unique reference,
            #       Otherwise blocks nesting workflows
            params[arg] = Reference(arg)
        cls = cls(**params)
        with cls:
            func(**cls.parameters)
        return cls


class WorkflowGraph:
    """Workflow graph.

    A graph contains blocks which defines the flow of the workflow.

    Arguments:
        root: Root block of the Workflow.
    """

    def __init__(self, root: WorkflowBlock) -> None:
        if isinstance(
            TaskExecutorContext.get_active(),
            WorkflowBlockBuilder,
        ):
            msg = "Nesting Workflows is not allowed."
            raise exceptions.WorkflowError(msg)
        self._root = root

    @classmethod
    def from_callable(cls, func: Callable) -> WorkflowGraph:
        """Create the graph from a callable."""
        return cls(WorkflowBlock.from_callable(func))

    def execute(self, executor: ExecutorState, **kwargs: object) -> None:
        """Execute the graph.

        Arguments:
            executor: Block executor.
            **kwargs: Input parameters of the workflow.
        """
        for k, v in kwargs.items():
            executor.set_state(k, v)
        self._root.execute(executor)
