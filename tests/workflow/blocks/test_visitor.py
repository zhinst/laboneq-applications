import pytest

from laboneq_applications.workflow import task
from laboneq_applications.workflow.blocks import BlockVisitor, TaskBlock, WorkflowBlock


@pytest.fixture()
def a_workflow():
    wf = WorkflowBlock(name="root")
    wf.extend(TaskBlock(task=task(lambda x: x, name="a_task")))
    wf_node = WorkflowBlock(name="node")
    wf_node.extend(TaskBlock(task=task(lambda x: x, name="b_task")))
    wf_node.extend(TaskBlock(task=task(lambda x: x, name="c_task")))
    return wf


def test_custom_visitor(a_workflow):
    workflow_queue = ["root", "node"]
    task_queue = ["a_task", "b_task", "c_task"]

    class CustomVisitor(BlockVisitor):
        def visit_workflowblock(self, block):
            assert block.name == workflow_queue[0]
            workflow_queue.pop(0)
            self.generic_visit(block)

        def visit_taskblock(self, block):
            assert block.name == task_queue[0]
            task_queue.pop(0)
            self.generic_visit(block)

    visitor = CustomVisitor()
    visitor.visit(a_workflow)


def test_custom_visitor_no_generic_visit(a_workflow):
    workflow_queue = ["root", "node"]
    task_queue = ["a_task", "b_task", "c_task"]

    class CustomVisitor(BlockVisitor):
        def visit_workflowblock(self, block):
            assert block.name == workflow_queue[0]
            workflow_queue.pop(0)

        def visit_taskblock(self, _):
            task_queue.pop(0)

    visitor = CustomVisitor()
    visitor.visit(a_workflow)
    # Only parent visited
    assert workflow_queue == ["node"]
    # No children visited
    assert len(task_queue) == 3
