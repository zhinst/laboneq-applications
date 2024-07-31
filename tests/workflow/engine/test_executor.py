import pytest

from laboneq_applications.workflow.engine.block import Block
from laboneq_applications.workflow.engine.executor import ExecutorState
from laboneq_applications.workflow.engine.reference import Reference
from laboneq_applications.workflow.exceptions import WorkflowError


class NoOpBlock(Block):
    def execute(self, executor: ExecutorState) -> None: ...


class TestExecutorState:
    def test_states(self):
        obj = ExecutorState()
        assert obj.states == {}

        obj.set_state("a", 2)
        assert obj.states == {"a": 2}

    def test_set_get_state(self):
        obj = ExecutorState()
        obj.set_state("a", 1)
        obj.set_state("a", 2)
        assert obj.get_state("a") == 2

        with pytest.raises(KeyError):
            obj.get_state("b")

    def test_result_handler(self):
        obj = ExecutorState()
        assert obj.result_handler is None
        obj.set_result_callback(1)
        assert obj.result_handler == 1

    def test_resolve_inputs(self):
        obj = ExecutorState()
        block = NoOpBlock(x=1, y=None)
        assert obj.resolve_inputs(block) == {"x": 1, "y": None}

    def test_resolve_reference_inputs(self):
        # Test straight reference
        obj = ExecutorState()
        block = NoOpBlock(x=1, y=Reference("z"))
        obj.set_state("z", 5)
        assert obj.resolve_inputs(block) == {"x": 1, "y": 5}

        # Test unwrap reference
        obj = ExecutorState()
        z = Reference("z")
        block = NoOpBlock(x=1, y=z[0])
        obj.set_state("z", [5])
        assert obj.resolve_inputs(block) == {"x": 1, "y": 5}

        # Test unresolved reference
        obj = ExecutorState()
        block = NoOpBlock(x=1, y=Reference("z"))
        with pytest.raises(WorkflowError, match="Result for 'z' is not resolved."):
            obj.resolve_inputs(block)
