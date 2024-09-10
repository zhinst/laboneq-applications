"""Workflow block base class."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from laboneq_applications.workflow._context import LocalContext

if TYPE_CHECKING:
    from collections.abc import Iterable

    from laboneq_applications.workflow.executor import ExecutorState


class Block(abc.ABC):
    """A base class for workflow blocks.

    A block can be an individual task or a collection of other blocks.

    Classes inheriting from `Block` must define the following methods:

        - `execute()`: A method that executes the block and it's children defined
            in `Block.body`.

    Arguments:
        **parameters: Input parameters of the block.
    """

    def __init__(self, **parameters: object) -> None:
        self._parameters = parameters
        self._body: list[Block] = []

    @property
    def parameters(self) -> dict:
        """Input parameters of the block."""
        return self._parameters

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

    def extend(self, blocks: Block | Iterable[Block]) -> None:
        """Extend the body of the block."""
        if isinstance(blocks, Block):
            self._body.append(blocks)
        else:
            self._body.extend(blocks)

    def find(
        self,
        by: type[Block],
        *,
        recursive: bool = False,
    ) -> list[Block]:
        """Search blocks within this block.

        Arguments:
            by: Block type to be searched.
            recursive: Searches recursively and returns a flat list of all
                the results.

        Returns:
            List of blocks that matches the search criteria.
            Empty list if no matches are found.
        """
        if not recursive:
            return [t for t in self.body if isinstance(t, by)]
        objs = []
        for x in self.body:
            if isinstance(x, by):
                objs.append(x)
            objs.extend(x.find(by=by, recursive=True))
        return objs

    def __enter__(self):
        BlockBuilderContext.enter(WorkflowBlockBuilder())

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        register = BlockBuilderContext.exit()
        self.extend(register.blocks)
        active_ctx = BlockBuilderContext.get_active()
        if active_ctx:
            active_ctx.register(self)

    @abc.abstractmethod
    def execute(self, executor: ExecutorState) -> None:
        """Execute the block."""


class WorkflowBlockBuilder:
    """Workflow block builder."""

    def __init__(self):
        self._blocks: list[Block] = []

    @property
    def blocks(self) -> list[Block]:
        """Workflow blocks."""
        return self._blocks

    def register(self, block: Block) -> None:
        """Register a block."""
        self._blocks.append(block)


class BlockBuilderContext(LocalContext[WorkflowBlockBuilder]):
    """Workflow block builder context."""

    _scope = "block_builder"
