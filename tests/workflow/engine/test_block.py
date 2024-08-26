"""Tests for laboneq_applications.workflow.engine.block."""

import textwrap

from laboneq_applications.workflow.engine.block import Block, TaskBlock
from laboneq_applications.workflow.engine.core import WorkflowResult
from laboneq_applications.workflow.engine.executor import ExecutorState
from laboneq_applications.workflow.reference import Reference
from laboneq_applications.workflow.task import task


class CustomBlock(Block):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def execute(self, executor: ExecutorState): ...


class TestBlock:
    def test_name(self):
        a = CustomBlock()
        assert a.name == "CustomBlock"

    def test_parameters(self):
        a = CustomBlock(a=1, b=2, c="bar")
        assert a.parameters == {"a": 1, "b": 2, "c": "bar"}

        a = CustomBlock()
        assert a.parameters == {}

    def test_extend_body(self):
        a = CustomBlock()
        b = CustomBlock()
        a.extend(b)
        assert a.body == [b]

    def test_context_body(self):
        a = CustomBlock()
        b = CustomBlock()
        with a:
            with b:
                ...
        assert a.body == [b]

    def test_find(self):
        a = CustomBlock()
        b = CustomBlock()
        c = CustomBlock()
        a.extend(b)
        b.extend(c)

        assert a.find(CustomBlock) == [b]
        assert a.find(CustomBlock, recursive=True) == [b, c]
        assert a.find(TaskBlock) == []
        assert b.find(CustomBlock) == [c]


class TestTaskBlock:
    @task
    def no_args_callable(): ...

    def test_name(self):
        block = TaskBlock(self.no_args_callable)
        assert block.name == "no_args_callable"

    def test_repr(self):
        block = TaskBlock(self.no_args_callable)
        assert (
            str(block) == "TaskBlock(task=Task(name=no_args_callable), parameters={})"
        )

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

    def test_execute(self):
        @task
        def addition(x, y):
            return x + y

        block = TaskBlock(addition, x=Reference("x"), y=Reference("y"))
        state = ExecutorState()
        result = WorkflowResult("test")
        with state.set_active_result(result):
            state.set_state("x", 1)
            state.set_state("y", 5)
            block.execute(state)
            assert state.get_state(block) == 6
        assert result.tasks[0].output == 6
