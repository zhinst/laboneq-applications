"""A workflow graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq_applications.workflow.graph_tree import WorkflowGraphTree

if TYPE_CHECKING:
    from laboneq_applications.workflow import blocks


class WorkflowGraph:
    """Workflow graph.

    A graph contains blocks which defines the flow of the workflow.

    Arguments:
        root: Root block of the Workflow.
    """

    def __init__(self, root: blocks.WorkflowBlock) -> None:
        self._root = root
        self._tree = WorkflowGraphTree(self._root)

    @property
    def tree(self) -> WorkflowGraphTree:
        """Tree graph of the workflow."""
        return self._tree
