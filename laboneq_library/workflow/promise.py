"""Module for promises.

Promises are used at Workflow definition page to delay the execution of the operations.
"""
from __future__ import annotations

import operator
from typing import Any

# TODO: Make sure promise attributes do not clash with underlying objects.


class Promise:
    """Promise.

    The promise records operations done on itself, which can then
    be later resolved.

    Attributes:
        head: Head promise of the given promise.
        ops: Operations that was done on the Promise
            operations can be, for example, using Python magic methods.
    """
    def __init__(
        self,
        head: Promise | None = None,
        ops: list[tuple[str, Any]] | None = None,
    ):
        self.head = self if head is None else head
        self.ops = ops if ops is not None else []

    def set_result(self, result: Any) -> None:  # noqa: ANN401
        """Set the promise result.

        Arguments:
            result: Result of the promise.

        Raises:
            ValueError: If result if tried to set to a child promise.
        """
        if self.head is not self:
            msg = "Cannot resolve child promises."
            raise ValueError(msg)
        self._result = result

    def result(self) -> Any:  # noqa: ANN401
        """Result of the promise.

        Raises:
            RuntimeError: Promise is not resolved.
        """
        try:
            res = self.head._result
        except AttributeError:
            msg = "Promise not resolved."
            raise RuntimeError(msg) from None
        for op, *args in self.ops:
            res = getattr(operator, op)(res, *args)
        return res

    def _create_child(
        self,
        head: Promise | None,
        ops: list[tuple[str, Any]] | None,
    ) -> Promise:
        """Creates a child of the promise.

        A child is created each time an operation is performed on the promise
        and each child keeps a reference to its' head promise.

        Once head promises result is set, each child promise result is also resolved.
        """
        return type(self)(head, ops)

    def __getitem__(self, item):  # noqa: ANN001
        return self._create_child(self.head, [*self.ops, ("getitem", item)])


class ReferencePromise(Promise):
    """A reference Promise.

    Reference promise keeps a reference to an object it is attached to.
    This way the object producing the result is not lost.

    Attributes:
        ref: Reference object of the promise.
        head: Head promise of the given promise.
        ops: Operations that was done on the Promise
            operations can be, for example, using Python magic methods.
    """
    def __init__(
        self,
        ref: object,
        head: Promise | None = None,
        ops: list[tuple[str, Any]] | None= None,
    ):
        super().__init__(head, ops)
        self._ref = ref

    @property
    def ref(self) -> object:
        """Promise result producing object reference."""
        return self._ref

    def _create_child(
        self,
        head: Promise,
        ops: list[tuple[str, Any]] | None = None,
    ) -> ReferencePromise:
        return type(self)(self._ref, head, ops)
