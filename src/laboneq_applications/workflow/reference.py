"""Module for workflow reference."""

from __future__ import annotations

import operator
from typing import Any, Callable

notset = object()


def get_default(reference: Reference) -> object:
    """Get reference default value."""
    return reference._default


class Reference:
    """A reference class.

    A `Reference` is a placeholder for input and result objects used while a workflow
    graph is being constructed.

    If tasks are the nodes in the workflow graph, then references define the edges.

    For example, in a workflow builder like:

    ```python
    @workflow
    def add_and_multiply(a, b):
        c = add_task(a, 2)
        d = multiply_task(c, 3)
        return d
    ```

    the values `a`, `b`, `c` and `d` are all references. By tracking where references
    are used, the graph builder can determine how inputs and outputs are passed between
    tasks in the graph.

    References also track some Python operations performed on them, so one can also use
    references like `a["value"]`. This creates a reference for "the item named 'value'
    from reference 'a'".

    When a workflow graph is run, the references are used to determine the input values
    to the next task from the outputs returned by the earlier tasks.

    The following operations are supported:

        * __getitem__()
        * __getattr__()
        * __eq__()

    Attributes:
        ref: Reference object.
        default: Default value of the reference.
    """

    # TODO: Which operators should be supported?
    #       To have a sensible error message on unsupported operation, on top of
    #       typical Python error message, the operators must either way be implemented.
    #       And therefore not supporting them might not be needed since they are
    #       implemented either way.
    def __init__(self, ref: object, default: object = notset):
        self._ref = ref
        self._default = default
        # Head of the reference
        self._head: Reference = self
        # Operations that was done on the reference
        self._ops: list[tuple[Callable, Any]] = []

    @property
    def ref(self) -> object:
        """The object reference points to."""
        return self._ref

    def unwrap(self, value: Any) -> Any:  # noqa: ANN401
        """Unwrap the reference and any operations done on it."""
        res = value
        for op, *args in self._ops:
            res = op(res, *args)
        return res

    def _create_child_reference(
        self,
        head: Reference,
        ops: list[tuple[Callable, Any]],
    ) -> Reference:
        obj = Reference(self._ref)
        obj._head = head
        obj._ops = ops
        return obj

    def __getitem__(self, item):  # noqa: ANN001
        return self._create_child_reference(
            self._head,
            [*self._ops, (operator.getitem, item)],
        )

    def __eq__(self, other: object):
        return self._create_child_reference(
            self._head,
            [*self._ops, (operator.eq, other)],
        )

    def __getattr__(self, other: object):
        return self._create_child_reference(
            self._head,
            [*self._ops, (getattr, other)],
        )
