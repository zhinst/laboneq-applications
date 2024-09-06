"""A workflow graph."""

from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Callable, cast

from laboneq_applications.core import now
from laboneq_applications.workflow.engine.block import Block, TaskBlock
from laboneq_applications.workflow.executor import ExecutionStatus
from laboneq_applications.workflow.options import (
    WorkflowOptions,
)
from laboneq_applications.workflow.options_parser import (
    get_and_validate_param_type,
)
from laboneq_applications.workflow.reference import Reference, notset
from laboneq_applications.workflow.result import WorkflowResult

if TYPE_CHECKING:
    from laboneq_applications.workflow.executor import ExecutorState
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
        super().__init__(**params)
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

    def set_params(self, executor: ExecutorState, **kwargs) -> None:
        """Set the initial parameters of the block."""
        inputs = kwargs
        input_opts = kwargs.get("options")  # Options from input arguments
        if input_opts is None:
            # Options from parent options
            input_opts = executor.get_options(self.name)
        if input_opts is None:
            # TODO: Replace with create_options() when new options are in
            input_opts = self.options_type()  # Default options
        inputs["options"] = input_opts
        for k, v in inputs.items():
            executor.set_state((self, k), v)

    def execute(self, executor: ExecutorState) -> None:
        """Execute the block."""
        # TODO: Separate executor results and WorkflowResult
        if executor.get_block_status(self) == ExecutionStatus.NOT_STARTED:
            executor.set_block_status(self, ExecutionStatus.IN_PROGRESS)
            inputs = executor.resolve_inputs(self)
            self.set_params(executor, **inputs)
            input_opts = executor.get_state((self, "options"))
            # NOTE: Correct input options are resolved only after 'set_param()'
            # Therefore they need to be overwritten for result
            inputs["options"] = input_opts
            result = WorkflowResult(self.name, input=inputs)
            result._start_time = now()
            executor.recorder.on_start(result)
            executor.set_state(self, result)
        elif executor.get_block_status(self) == ExecutionStatus.IN_PROGRESS:
            result = executor.get_state(self)
            input_opts = executor.get_state((self, "options"))
        else:
            # Block is finished
            return
        executor.add_workflow_result(result)
        try:
            with executor.set_active_workflow_settings(result, input_opts):
                with executor:
                    for block in self._body:
                        if executor.get_block_status(block) == ExecutionStatus.FINISHED:
                            continue
                        block.execute(executor)
                    executor.set_block_status(self, ExecutionStatus.FINISHED)
                    result._end_time = now()
                    executor.recorder.on_end(result)
        except Exception as error:
            executor.set_block_status(self, ExecutionStatus.FINISHED)
            result._end_time = now()
            if not getattr(
                error,
                "_is_recorded",
                False,
            ):  # TODO: better mechanism
                executor.recorder.on_error(result, error)
            executor.recorder.on_end(result)
            error._is_recorded = True
            raise

    @classmethod
    def from_callable(cls, name: str, func: Callable, **kwargs) -> WorkflowBlock:
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
        with obj:
            func(**obj.parameters)
        return obj


class WorkflowGraph:
    """Workflow graph.

    A graph contains blocks which defines the flow of the workflow.

    Arguments:
        root: Root block of the Workflow.
        name: The name of the workflow.
    """

    def __init__(self, root: WorkflowBlock) -> None:
        self._root = root

    @property
    def name(self) -> str:
        """Name of the graph."""
        return self._root.name

    def create_options(self) -> WorkflowOptions:
        """Create options for the graph."""
        return self._root.create_options()

    @property
    def options_type(self) -> type[WorkflowOptions]:
        """Type of graph options."""
        return self._root.options_type

    @property
    def tasks(self) -> list[TaskBlock]:
        """A flat list of individual tasks within the graph."""
        return cast(list[TaskBlock], self._root.find(by=TaskBlock, recursive=True))

    @classmethod
    def from_callable(cls, func: Callable, name: str | None = None) -> WorkflowGraph:
        """Create the graph from a callable."""
        return cls(WorkflowBlock.from_callable(name or func.__name__, func))

    def validate_input(self, **kwargs: object) -> None:
        """Validate input parameters of the graph.

        Raises:
            TypeError: `options`-parameter is of wrong type.
        """
        if "options" in kwargs:
            opt_param = kwargs["options"]
            if opt_param is not None and not isinstance(
                opt_param,
                (self._root.options_type, dict),
            ):
                msg = (
                    "Workflow input options must be of "
                    f"type '{self._root.options_type.__name__}', 'dict' or 'None'"
                )
                raise TypeError(msg)

    def execute(self, executor: ExecutorState, **kwargs: object) -> None:
        """Execute the graph.

        Arguments:
            executor: Block executor.
            **kwargs: Input parameters of the workflow.
        """
        self._root.set_params(executor, **kwargs)
        self._root.execute(executor)
