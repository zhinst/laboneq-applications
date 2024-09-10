import re

import pytest

from laboneq_applications.workflow.blocks.block import Block
from laboneq_applications.workflow.exceptions import WorkflowError
from laboneq_applications.workflow.executor import ExecutorState, _ExecutorInterrupt
from laboneq_applications.workflow.reference import Reference


class NoOpBlock(Block):
    def execute(self, executor: ExecutorState) -> None: ...


class TestExecutorState:
    def test_variables(self):
        obj = ExecutorState()
        assert obj.block_variables == {}

        obj.set_variable("a", 2)
        assert obj.block_variables == {"a": 2}

    def test_set_get_variable(self):
        obj = ExecutorState()
        obj.set_variable("a", 1)
        obj.set_variable("a", 2)
        assert obj.get_variable("a") == 2

        with pytest.raises(KeyError):
            obj.get_variable("b")

    def test_recorder_callbacks(self):
        class Recorder:
            def __init__(self) -> None:
                self.results = []

            def on_task_end(self, task):
                self.results.append(task)

        obj = ExecutorState()
        recorder1 = Recorder()
        recorder2 = Recorder()
        obj.add_recorder(recorder1)
        obj.add_recorder(recorder2)
        obj.recorder.on_task_end(1)

        assert recorder1.results == [1]
        assert recorder2.results == [1]

    def test_resolve_inputs(self):
        obj = ExecutorState()
        block = NoOpBlock(x=1, y=None)
        assert obj.resolve_inputs(block) == {"x": 1, "y": None}

    def test_resolve_reference_inputs(self):
        # Test straight reference
        obj = ExecutorState()
        block = NoOpBlock(x=1, y=Reference("z"))
        obj.set_variable("z", 5)
        assert obj.resolve_inputs(block) == {"x": 1, "y": 5}

        # Test unwrap reference
        obj = ExecutorState()
        z = Reference("z")
        block = NoOpBlock(x=1, y=z[0])
        obj.set_variable("z", [5])
        assert obj.resolve_inputs(block) == {"x": 1, "y": 5}

        # Test unresolved reference
        obj = ExecutorState()
        block = NoOpBlock(x=1, y=Reference("z"))
        with pytest.raises(WorkflowError, match="Result for 'z' is not resolved."):
            obj.resolve_inputs(block)

    def test_resolve_reference_default(self):
        obj = ExecutorState()
        block = NoOpBlock(y=Reference(None, default=1))
        assert obj.resolve_inputs(block) == {"y": 1}

    def test_interrupt(self):
        obj = ExecutorState()
        assert obj.has_active_context is False
        # Context not active (Internal error)
        with pytest.raises(
            WorkflowError,
            match=re.escape(
                "interrupt() cannot be called outside of active executor context."
            ),
        ):
            obj.interrupt()

        assert obj.has_active_context is False
        # Test interrupt signal caught within the context
        with obj:
            assert obj.has_active_context
            with pytest.raises(_ExecutorInterrupt):
                obj.interrupt()
        assert obj.has_active_context is False
