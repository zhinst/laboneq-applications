from __future__ import annotations

import textwrap

from laboneq_applications.workflow import task
from laboneq_applications.workflow.blocks.task_block import TaskBlock
from laboneq_applications.workflow.executor import ExecutorState
from laboneq_applications.workflow.options_base import BaseOptions
from laboneq_applications.workflow.reference import Reference
from laboneq_applications.workflow.result import WorkflowResult


class TaskOptions(BaseOptions):
    param: int = 0


class TestTaskBlock:
    @task
    def no_args_callable(): ...

    def test_name(self):
        block = TaskBlock(self.no_args_callable)
        assert block.name == "no_args_callable"

    def test_repr(self):
        block = TaskBlock(self.no_args_callable)
        assert (
            repr(block) == "TaskBlock(task=task(name=no_args_callable), parameters={})"
        )

    def test_str(self):
        block = TaskBlock(self.no_args_callable)
        assert str(block) == "task(name=no_args_callable)"

    def test_parameters(self):
        block = TaskBlock(self.no_args_callable)
        assert block.parameters == {}

    def test_ref(self):
        block = TaskBlock(self.no_args_callable)
        assert block.ref == Reference(block)

    def test_src(self):
        @task
        def addition(x, y):
            return x + y

        blk = TaskBlock(addition)
        assert blk.src == textwrap.dedent("""\
            @task
            def addition(x, y):
                return x + y
        """)

    def test_options(self):
        @task
        def a_task(): ...

        blk = TaskBlock(a_task)
        assert blk.options_type is None

        @task
        def b_task(options: TaskOptions | None = None): ...

        blk = TaskBlock(b_task)
        assert blk.options_type == TaskOptions

    def test_execute(self):
        @task
        def addition(x, y):
            return x + y

        params = {"x": Reference("x"), "y": Reference("y")}
        block = TaskBlock(addition, parameters=params)
        state = ExecutorState()
        result = WorkflowResult("test")
        with state.set_active_workflow_settings(result):
            state.set_variable("x", 1)
            state.set_variable("y", 5)
            block.execute(state)
            assert state.get_variable(block) == 6
        assert result.tasks[0].output == 6
