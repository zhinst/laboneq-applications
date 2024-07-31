"""A workflow graph."""

from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Any, Callable

from laboneq_applications.workflow import exceptions
from laboneq_applications.workflow._context import TaskExecutorContext
from laboneq_applications.workflow.engine.block import (
    Block,
    WorkflowBlockBuilder,
)
from laboneq_applications.workflow.engine.options import WorkflowOptions
from laboneq_applications.workflow.engine.reference import Reference
from laboneq_applications.workflow.options import get_and_validate_param_type

if TYPE_CHECKING:
    from laboneq_applications.workflow.engine.executor import ExecutorState


class WorkflowBlock(Block):
    """Workflow block."""

    def __init__(self, **parameters: object) -> None:
        super().__init__(**parameters)
        self._options: type[WorkflowOptions] = WorkflowOptions
        self._output: Reference | object = None

    @property
    def options(self) -> type[WorkflowOptions]:
        """Type of block options."""
        return self._options

    def execute(self, executor: ExecutorState) -> Any:  # noqa: ANN401
        """Execute the block."""
        input_opts = executor.resolve_inputs(self).get("options")
        if input_opts is None:
            input_opts = self._options()
        executor.options = input_opts
        for block in self._body:
            block.execute(executor)
        if isinstance(self._output, Reference):
            return executor.get_state(self._output.ref)
        return self._output

    @classmethod
    def from_callable(cls, func: Callable) -> WorkflowBlock:
        """Create the block from a callable."""
        params = {}
        arg_opt = None
        for arg in signature(func).parameters:
            if arg == "options":
                arg_opt = get_and_validate_param_type(func, "options", WorkflowOptions)
            # TODO: Improve reference system to unique reference,
            #       Otherwise blocks nesting workflows
            params[arg] = Reference(arg)
        obj = cls(**params)
        with obj:
            obj._output = func(**obj.parameters)
        if arg_opt:
            obj._options = arg_opt
        return obj


class WorkflowGraph:
    """Workflow graph.

    A graph contains blocks which defines the flow of the workflow.

    Arguments:
        root: Root block of the Workflow.
    """

    def __init__(self, root: WorkflowBlock) -> None:
        if isinstance(
            TaskExecutorContext.get_active(),
            WorkflowBlockBuilder,
        ):
            msg = "Nesting Workflows is not allowed."
            raise exceptions.WorkflowError(msg)
        self._root = root

    @classmethod
    def from_callable(cls, func: Callable) -> WorkflowGraph:
        """Create the graph from a callable."""
        return cls(WorkflowBlock.from_callable(func))

    def validate_input(self, **kwargs: object) -> None:
        """Validate input parameters of the graph.

        Raises:
            TypeError: `options`-parameter is of wrong type.
        """
        if "options" in kwargs:
            opt_param = kwargs["options"]
            if opt_param is not None and not isinstance(opt_param, self._root.options):
                msg = (
                    "Workflow input options must be of "
                    f"type '{self._root.options}' or 'None'"
                )
                raise TypeError(msg)

    def execute(self, executor: ExecutorState, **kwargs: object) -> Any:  # noqa: ANN401
        """Execute the graph.

        Arguments:
            executor: Block executor.
            **kwargs: Input parameters of the workflow.
        """
        for k, v in kwargs.items():
            executor.set_state(k, v)
        return self._root.execute(executor)
