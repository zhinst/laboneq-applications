from __future__ import annotations

from laboneq_applications.workflow import WorkflowOptions, task
from laboneq_applications.workflow.engine import if_
from laboneq_applications.workflow.engine.graph import WorkflowBlock, WorkflowGraph
from laboneq_applications.workflow.reference import Reference


class TestWorkflowGraph:
    def test_name(self):
        def wf_block(): ...

        graph = WorkflowGraph.from_callable(wf_block)
        assert graph.name == "wf_block"

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


class Opts(WorkflowOptions): ...


class TestWorkflowBlock:
    def test_name(self):
        block = WorkflowBlock(name="test")
        assert block.name == "test"

    def test_parameters(self):
        block = WorkflowBlock(name="test")
        assert block.parameters == {}

        block = WorkflowBlock(name="test", parameters={"foo": 1, "bar": 5})
        assert block.parameters == {"foo": 1, "bar": 5}

    def test_options(self):
        block = WorkflowBlock(name="test")
        assert block.options == WorkflowOptions

    def test_from_callable(self):
        def work(x, y=5): ...

        block = WorkflowBlock.from_callable("test", work)
        assert block.options == WorkflowOptions
        assert block.parameters == {"x": Reference("x"), "y": Reference("y")}

        def work_opts(x, options: Opts | None = None): ...

        block = WorkflowBlock.from_callable("test", work_opts)
        assert block.options == Opts
        assert block.parameters == {
            "x": Reference("x"),
            "options": Reference("options"),
        }
