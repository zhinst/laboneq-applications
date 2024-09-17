from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq_applications.workflow.blocks.block import Block
from laboneq_applications.workflow.blocks.task_block import TaskBlock

if TYPE_CHECKING:
    from laboneq_applications.workflow.executor import ExecutorState


class CustomBlock(Block):
    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs)

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
