"""A block of work in a workflow."""

from __future__ import annotations

import abc
from collections import defaultdict
from typing import Any

from laboneq_applications.workflow._context import LocalContext
from laboneq_applications.workflow.engine.promise import PromiseResultNotResolvedError
from laboneq_applications.workflow.engine.resolver import ArgumentResolver
from laboneq_applications.workflow.exceptions import WorkflowError


class BlockResult:
    """A class representing block result.

    A collection of results recorded within the block.
    """

    def __init__(self) -> None:
        self._log: dict[str, list] = defaultdict(list)

    @property
    def log(self) -> dict[str, list]:
        """Log of the block."""
        return dict(self._log)

    def merge(self, other: BlockResult) -> None:
        """Merge block results."""
        for key, value in other.log.items():
            self._log[key].extend(value)

    def add_result(self, key: str, result: Any) -> None:  # noqa: ANN401
        """Add result to the log."""
        self._log[key].append(result)


class Block(abc.ABC):
    """A base class for workflow blocks.

    A block can be an individual task or a collection of other blocks.

    Classes inheriting from `Block` must define the following methods:

        - `execute()`: A method that executes the block and it's children defined
            in `Block.body`.

    Arguments:
        *args: Arguments of the block.
        **kwargs: Keyword arguments of the block.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self._body: list[Block] = []
        self._resolver = ArgumentResolver(*self.args, **self.kwargs)

    @property
    def name(self) -> str:
        """Name of the block."""
        return self.__class__.__name__

    @property
    def body(self) -> list[Block]:
        """Body of the block.

        A list of other blocks that are defined within this block.
        """
        return self._body

    def __enter__(self):
        LocalContext.enter()

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        # TODO: Graph to flow all the way to the top Block (e.g Workflow).
        #       This would be mainly for inspection
        inner = LocalContext.exit().instances
        self._body.extend(inner)
        if LocalContext.is_active():
            LocalContext.active_context().register(self)

    def _run_block(self, block: Block) -> BlockResult:
        """Run a block belonging to this block.

        Argument:
            block: Child block of this block.
        """
        try:
            return block.execute()
        except PromiseResultNotResolvedError as error:
            raise WorkflowError(error) from error

    @abc.abstractmethod
    def execute(self) -> BlockResult:
        """Execute the block.

        Classes implementing this method should run any child block
        via `_run_block()` method.

        Returns:
            Block result.
        """
