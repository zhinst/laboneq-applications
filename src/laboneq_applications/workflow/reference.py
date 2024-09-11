"""Module for workflow reference."""

from __future__ import annotations

import operator
from typing import Any, Callable

notset = object()


def get_default(reference: Reference) -> object:
    """Get reference default value."""
    return reference._default


def unwrap(reference: Reference, value: object) -> object:
    """Unwrap the reference and any operations done on it."""
    res = value
    for op, *args in reference._ops:
        res = op(res, *args)
    return res


def get_ref(reference: Reference) -> object:
    """Return an object the reference points to."""
    return reference._ref


class Reference:
    """A reference class.

    A `Reference` is a placeholder for objects used while a workflow
    graph is being constructed.

    If tasks are the nodes in the workflow graph, then references define the edges.

    For example, in a workflow definition like:

    ```python
    @workflow
    def add_and_multiply(a, b):
        c = add_task(a, 2)
        d = multiply_task(c, 3)
    ```

    the values `a`, `b`, `c` and `d` are all references. By tracking where references
    are used, the graph builder can determine how inputs and outputs are passed between
    tasks in the graph.

    References also track some Python operations performed on them, so one can also use
    references like `a["value"]`. This creates a reference for "the item named 'value'
    from reference 'a'".

    When a workflow graph is run, the references are used to determine the input values
    to the next task from the outputs returned by the earlier tasks.

    The following operations are supported

    * __getitem__()
    * __getattr__()
    * __eq__()

    Notes on specific Python operations
    ---

    *Equality comparison*

    For equality comparison, especially with booleans, use `==` instead of `is`.
        Equality with `is` will always return `False`.

    Arguments:
        ref: An object this reference points to.
        default: Default value of `ref`.
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

    def _create_child_reference(
        self,
        head: Reference,
        ops: list[tuple[Callable, Any]],
    ) -> Reference:
        obj = Reference(self._ref)
        obj._head = head
        obj._ops = ops
        return obj

    def __getitem__(self, item: object):
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

    def __iter__(self):
        raise NotImplementedError("Iterating a workflow Reference is not supported.")

    def __repr__(self):
        return f"Reference(ref={self._ref}, default={self._default})"
