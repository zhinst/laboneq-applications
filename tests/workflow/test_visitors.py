from laboneq_applications import workflow
from laboneq_applications.workflow import blocks
from laboneq_applications.workflow.visitors import SpecificBlockTypeCollector


class TestSpecificBlockTypeCollector:
    def test_tasks_flat(self):
        @workflow.task
        def task_a(): ...

        @workflow.task
        def task_b(): ...

        @workflow.workflow
        def wf_block():
            task_a()
            task_b()
            task_a()

        wf = wf_block()

        collector = SpecificBlockTypeCollector(wf._graph._root)
        tasks = collector.collect([blocks.TaskBlock])
        assert len(tasks) == 3
        assert tasks[0].task == task_a
        assert tasks[1].task == task_b
        assert tasks[2].task == task_a

    def test_tasks_nested(self):
        @workflow.task
        def task_a(): ...

        @workflow.task
        def task_b(): ...

        @workflow.workflow
        def wf_block():
            task_a()
            with workflow.if_(condition=True):
                task_b()
                task_a()
                with workflow.if_(condition=True):
                    task_b()

        wf = wf_block()
        collector = SpecificBlockTypeCollector(wf._graph._root)
        tasks = collector.collect([blocks.TaskBlock])
        assert len(tasks) == 4
        assert tasks[0].task == task_a
        assert tasks[1].task == task_b
        assert tasks[2].task == task_a
        assert tasks[3].task == task_b

        collector = SpecificBlockTypeCollector(wf._graph._root)
        ifs = collector.collect([blocks.IFExpression])
        assert len(ifs) == 2

    def test_workflows(self):
        @workflow.task
        def task_a(): ...

        @workflow.task
        def task_b(): ...

        @workflow.workflow
        def wf_block_nested_2():
            task_b()

        @workflow.workflow
        def wf_block_nested():
            task_a()
            wf_block_nested_2()

        @workflow.workflow
        def wf_block():
            task_a()
            wf_block_nested()

        wf = wf_block()
        collector = SpecificBlockTypeCollector(wf._graph._root)
        tasks = collector.collect([blocks.WorkflowBlock])
        assert len(tasks) == 3
