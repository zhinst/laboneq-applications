from __future__ import annotations

from laboneq_applications.workflow import WorkflowOptions
from laboneq_applications.workflow.blocks.workflow_block import WorkflowBlock
from laboneq_applications.workflow.graph import WorkflowGraph


class TestWorkflowGraph:
    def test_name(self):
        wf_block = WorkflowBlock(name="test")
        graph = WorkflowGraph(wf_block)
        assert graph.name == "test"

    def test_root(self):
        wf_block = WorkflowBlock(name="test")
        graph = WorkflowGraph(wf_block)
        assert graph.root == wf_block

    def test_options_type(self):
        wf_block = WorkflowBlock(name="test")
        graph = WorkflowGraph(wf_block)
        assert graph.options_type == WorkflowOptions

    def test_from_callable(self):
        def wf_block(): ...

        graph = WorkflowGraph.from_callable(wf_block)
        assert graph.root.name == "wf_block"
        graph = WorkflowGraph.from_callable(wf_block, name="test")
        assert graph.root.name == "test"
