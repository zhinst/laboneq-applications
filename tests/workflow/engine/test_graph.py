from laboneq_applications.workflow import task
from laboneq_applications.workflow.engine import if_
from laboneq_applications.workflow.engine.graph import WorkflowGraph


class TestWorkflowGraph:
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
