import pytest

from laboneq_applications.workflow.blocks.block import Block
from laboneq_applications.workflow.exceptions import WorkflowError
from laboneq_applications.workflow.executor import ExecutorState
from laboneq_applications.workflow.reference import Reference


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

    def test_resolve_reference_default(self):
        obj = ExecutorState()
        block = NoOpBlock(y=Reference(None, default=1))
        assert obj.resolve_inputs(block) == {"y": 1}
