from __future__ import annotations

from laboneq_applications.workflow import WorkflowOptions, WorkflowResult
from laboneq_applications.workflow.blocks.workflow_block import WorkflowBlock
from laboneq_applications.workflow.executor import ExecutionStatus, ExecutorState
from laboneq_applications.workflow.reference import Reference, get_default, notset


class Opts(WorkflowOptions): ...


class TestWorkflowBlock:
    def test_name(self):
        block = WorkflowBlock(name="test")
        assert block.name == "test"

    def test_parameters(self):
        block = WorkflowBlock(name="test")
        assert block.parameters == {}

        params = {"foo": 1, "bar": 5}
        block = WorkflowBlock(name="test", parameters=params)
        assert params == {"foo": 1, "bar": 5}
        assert get_default(block.parameters["foo"]) == 1
        assert get_default(block.parameters["bar"]) == 5

        block = WorkflowBlock(
            name="test", parameters={"foo": Reference(ref=None, default=3)}
        )
        assert get_default(block.parameters["foo"]) == 3

    def test_options(self):
        block = WorkflowBlock(name="test")
        assert block.options_type == WorkflowOptions

    def test_from_callable_defaults(self):
        def work(x, y=5): ...

        block = WorkflowBlock.from_callable("test", work)
        assert block.options_type == WorkflowOptions
        assert get_default(block.parameters["x"]) == notset
        assert get_default(block.parameters["y"]) == 5

    def test_from_callable_default_options(self):
        def work_opts(options: Opts | None = None): ...

        block = WorkflowBlock.from_callable("test", work_opts)
        assert block.options_type == Opts
        assert get_default(block.parameters["options"]) is None

    def test_from_callable_overwrite_default(self):
        def work_opts(x: int = 5): ...

        block = WorkflowBlock.from_callable("test", work_opts, x=7)
        assert get_default(block.parameters["x"]) == 7

    def test_execution_status(self):
        task_calls = 0

        def t():
            nonlocal task_calls
            task_calls += 1

        def work():
            t()

        block = WorkflowBlock.from_callable("test", work)
        executor = ExecutorState()
        result = WorkflowResult("test")

        with executor.set_active_workflow_settings(result):
            block.execute(executor)
        assert executor.get_block_status(block) == ExecutionStatus.FINISHED

        with executor.set_active_workflow_settings(result):
            block.execute(executor)
        assert executor.get_block_status(block) == ExecutionStatus.FINISHED
        assert task_calls == 1

    def test_result_input(self):
        def work(x, y: int = 5): ...

        block = WorkflowBlock.from_callable("test", work)
        executor = ExecutorState()
        block.set_params(executor, x=1)
        block.execute(executor)
        result = executor.get_state(block)
        assert result.input == {"x": 1, "y": 5, "options": WorkflowOptions()}

    def test_result_input_reference(self):
        def work(x, y: int = 5): ...

        block = WorkflowBlock.from_callable("test", work, y=Reference("param"))
        executor = ExecutorState()
        block.set_params(executor, x=1)
        executor.set_state("param", 8)
        block.execute(executor)
        result = executor.get_state(block)
        assert result.input == {"x": 1, "y": 8, "options": WorkflowOptions()}
