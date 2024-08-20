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
from laboneq_applications.workflow.engine.reference import Reference
from laboneq_applications.workflow.options import (
    WorkflowOptions,
    get_and_validate_param_type,
)

if TYPE_CHECKING:
    from laboneq_applications.workflow.engine.executor import ExecutorState


class WorkflowBlock(Block):
    """Workflow block."""

    def __init__(self, **parameters: object) -> None:
        super().__init__(**parameters)
        self._options: type[WorkflowOptions] = WorkflowOptions

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
        with executor:
            for block in self._body:
                block.execute(executor)

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
            func(**obj.parameters)
        if arg_opt:
            obj._options = arg_opt
        return obj


class WorkflowGraph:
    """Workflow graph.

    A graph contains blocks which defines the flow of the workflow.

    Arguments:
        root: Root block of the Workflow.
        name: The name of the workflow.
    """

    def __init__(self, root: WorkflowBlock, name: str) -> None:
        if isinstance(
            TaskExecutorContext.get_active(),
            WorkflowBlockBuilder,
        ):
            msg = "Nesting Workflows is not allowed."
            raise exceptions.WorkflowError(msg)
        self._root = root
        self.name = name

    @classmethod
    def from_callable(cls, func: Callable) -> WorkflowGraph:
        """Create the graph from a callable."""
        return cls(WorkflowBlock.from_callable(func), func.__name__)

    def validate_input(self, **kwargs: object) -> None:
        """Validate input parameters of the graph.

        Raises:
            TypeError: `options`-parameter is of wrong type.
        """
        if "options" in kwargs:
            opt_param = kwargs["options"]
            if opt_param is not None and not isinstance(
                opt_param,
                (self._root.options, dict),
            ):
                msg = (
                    "Workflow input options must be of "
                    f"type '{self._root.options.__name__}', 'dict' or 'None'"
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
