"""Block for workflows."""

from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Callable

from laboneq_applications.core import utc_now
from laboneq_applications.workflow import variable_tracker
from laboneq_applications.workflow.blocks.block import Block
from laboneq_applications.workflow.blocks.task_block import TaskBlock
from laboneq_applications.workflow.executor import ExecutionStatus, ExecutorState
from laboneq_applications.workflow.options import WorkflowOptions
from laboneq_applications.workflow.options_parser import get_and_validate_param_type
from laboneq_applications.workflow.reference import Reference, notset
from laboneq_applications.workflow.result import WorkflowResult

if TYPE_CHECKING:
    from laboneq_applications.workflow.options_base import BaseOptions


class WorkflowBlock(Block):
    """Workflow block."""

    def __init__(
        self,
        name: str,
        options_type: type[WorkflowOptions] | None = WorkflowOptions,
        parameters: dict | None = None,
    ) -> None:
        self._name = name
        self._options_type = options_type or WorkflowOptions
        params = {}
        for param, default in (parameters or {}).items():
            if isinstance(default, Reference):
                params[param] = default
            else:
                params[param] = Reference((self, param), default=default)
        super().__init__(parameters=params)
        self._ref = Reference(self)

    @property
    def name(self) -> str:
        """Name of the block."""
        return self._name

    @property
    def options_type(self) -> type[WorkflowOptions]:
        """Type of workflow options."""
        return self._options_type

    def create_options(self) -> WorkflowOptions:
        """Create options for the block.

        The method goes over the sub-blocks and finds which blocks has
            options available.

        Same task option instance is shared across the same named task executions
        per workflow, therefore the same task calls within a single workflow block
        cannot have multiple option definitions.

        Returns:
            Workflow options where `tasks` is populated with the default options
                of sub-blocks.
        """

        def get_options(block: Block, opts: dict) -> BaseOptions | None:
            if isinstance(block, WorkflowBlock):
                return block.create_options()
            if isinstance(block, TaskBlock) and block.options_type is not None:
                return block.options_type()
            for x in block.body:
                maybe_opts = get_options(x, opts)
                if maybe_opts:
                    opts[x.name] = maybe_opts
            return None

        tasks = {}
        for x in self.body:
            maybe_opts = get_options(x, tasks)
            if maybe_opts:
                tasks[x.name] = maybe_opts
        return self.options_type(task_options=tasks)

    @property
    def ref(self) -> Reference:
        """Reference to the object."""
        return self._ref

    def set_params(self, executor: ExecutorState, **kwargs: object) -> None:
        """Set the initial parameters of the block.

        Arguments:
            executor: Active executor.
            **kwargs: Input parameters of the block.
        """
        inputs = kwargs
        input_opts = kwargs.get("options")  # Options from input arguments
        # Options might be in a dict
        if isinstance(input_opts, dict):
            input_opts = self.options_type.from_dict(input_opts)
        elif input_opts is None:
            # Options from parent options
            input_opts = executor.get_options(self.name)
        if input_opts is None:
            # TODO: Replace with create_options() when new options are in
            input_opts = self.options_type()  # Default options
        inputs["options"] = input_opts
        for k, v in inputs.items():
            executor.set_variable((self, k), v)

    def execute(self, executor: ExecutorState) -> None:
        """Execute the block."""
        # TODO: Separate executor results and WorkflowResult
        if executor.get_block_status(self) == ExecutionStatus.NOT_STARTED:
            executor.set_block_status(self, ExecutionStatus.IN_PROGRESS)
            inputs = executor.resolve_inputs(self)
            self.set_params(executor, **inputs)
            input_opts = executor.get_variable((self, "options"))
            # NOTE: Correct input options are resolved only after 'set_param()'
            # Therefore they need to be overwritten for result
            inputs["options"] = input_opts
            result = WorkflowResult(self.name, input=inputs)
            result._start_time = utc_now()
            executor.recorder.on_start(result)
            executor.set_variable(self, result)
        elif executor.get_block_status(self) == ExecutionStatus.IN_PROGRESS:
            result = executor.get_variable(self)
            input_opts = executor.get_variable((self, "options"))
        else:
            # Block is finished
            return
        executor.add_workflow_result(result)
        try:
            with executor:
                with executor.set_active_workflow_settings(result, input_opts):
                    for block in self._body:
                        if executor.get_block_status(block) == ExecutionStatus.FINISHED:
                            continue
                        block.execute(executor)
                    executor.set_block_status(self, ExecutionStatus.FINISHED)
                    result._end_time = utc_now()
                    executor.recorder.on_end(result)
            if executor.settings.run_until == self.name and executor.has_active_context:
                executor.interrupt()
        except Exception as error:
            executor.set_block_status(self, ExecutionStatus.FINISHED)
            result._end_time = utc_now()
            executor.recorder.on_error(result, error)
            executor.recorder.on_end(result)
            raise

    @classmethod
    def from_callable(
        cls, name: str, func: Callable, **kwargs: object
    ) -> WorkflowBlock:
        """Create the block from a callable.

        By default the signature of the function is used to define
        the default parameters of the block.

        Arguments:
            name: Name of the block.
            func: A function defining the workflow
            **kwargs: Default parameter values that overwrite function
                signature
        """
        params = {}
        for k, v in signature(func).parameters.items():
            if k in kwargs:
                value = kwargs[k]
            elif v.default == v.empty:
                value = notset
            else:
                value = v.default
            params[k] = value

        opt_type_hint = None
        if "options" in params:
            opt_type_hint = get_and_validate_param_type(
                func, "options", WorkflowOptions
            )
        obj = cls(name, opt_type_hint, params)
        with variable_tracker.WorkflowFunctionVariableTrackerContext.scoped(
            variable_tracker.WorkflowFunctionVariableTracker()
        ):
            with obj:
                func(**obj.parameters)
        return obj
