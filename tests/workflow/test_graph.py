from __future__ import annotations

from laboneq_applications.workflow import WorkflowOptions, task
from laboneq_applications.workflow.blocks.if_block import if_
from laboneq_applications.workflow.graph import WorkflowGraph


class TestWorkflowGraph:
    def test_name(self):
        def wf_block(): ...

        graph = WorkflowGraph.from_callable(wf_block)
        assert graph.name == "wf_block"

    def test_options_from_argument(self):
        def wf_block(options: WorkflowOptions | None = None): ...

        graph = WorkflowGraph.from_callable(wf_block)
        assert graph.options_type == WorkflowOptions

    def test_options_default(self):
        def wf_block(): ...

        graph = WorkflowGraph.from_callable(wf_block)
        assert graph.options_type == WorkflowOptions

    def test_tasks_flat(self):
        @task
        def task_a(): ...

        @task
        def task_b(): ...

        def wf_block():
            task_a()
            task_b()
            task_a()

        graph = WorkflowGraph.from_callable(wf_block)
        assert len(graph.tasks) == 3
        assert graph.tasks[0].task == task_a
        assert graph.tasks[1].task == task_b
        assert graph.tasks[2].task == task_a

    def test_tasks_nested(self):
        @task
        def task_a(): ...

        @task
        def task_b(): ...

        def wf_block():
            task_a()
            with if_(condition=True):
                task_b()
                task_a()
                with if_(condition=True):
                    task_b()

        graph = WorkflowGraph.from_callable(wf_block)
        assert len(graph.tasks) == 4
        assert graph.tasks[0].task == task_a
        assert graph.tasks[1].task == task_b
        assert graph.tasks[2].task == task_a
        assert graph.tasks[3].task == task_b
