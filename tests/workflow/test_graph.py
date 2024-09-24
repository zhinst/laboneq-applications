from __future__ import annotations

from IPython.lib.pretty import pretty

from laboneq_applications import workflow
from laboneq_applications.workflow import blocks
from laboneq_applications.workflow.graph import WorkflowGraph


class TestWorkflowGraph:
    def test_display_tree(self):
        # root
        root = blocks.WorkflowBlock(name="root")
        root.extend(blocks.TaskBlock(workflow.task(lambda: None, name="top")))

        if_ = blocks.IFExpression(None)
        root.extend(if_)
        branch = blocks.WorkflowBlock(name="branch")
        if_.extend(branch)

        branch.extend(blocks.TaskBlock(workflow.task(lambda: None, name="in_branch")))
        for_ = blocks.ForExpression([])
        branch.extend(for_)
        for_.extend(blocks.TaskBlock(workflow.task(lambda: None, name="in_loop")))

        root.extend(blocks.TaskBlock(workflow.task(lambda: None, name="bottom")))
        root.extend(blocks.ReturnStatement(None))

        graph = WorkflowGraph(root)
        expected = """\
workflow(name=root)
├─ task(name=top)
├─ if_()
│  └─ workflow(name=branch)
│     ├─ task(name=in_branch)
│     └─ for_()
│        └─ task(name=in_loop)
├─ task(name=bottom)
└─ return_()\
"""
        assert str(graph.tree) == expected
        assert pretty(graph.tree) == expected
