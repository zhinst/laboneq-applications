""" Core classes for defining sets of quantum operations on qubits. """

from __future__ import annotations

from collections import defaultdict
from typing import Type, Callable
import inspect
import textwrap
import threading

from laboneq.dsl.experiment import builtins
from laboneq.simple import Experiment, Qubit, Section, pulse_library

# TODO: Understand how to support for self.add_cal_states_sections(qubit, add_to=self.acquire_loop)

# TODO: Finish basic gate implementations. Define gate signatures. Check somehow?

# TODO: Add optional operation section parameters:
#   - alignment
#   - on_system_grix
#   - play_after
#   - ... any others

# TODO: add support for broadcasting operations over lists of qubits
#   - broadcast over qubits
#   - check qubits are unique
#   - document that operations accept multiple qubits


def quantum_operation(f: Callable) -> Callable:
    """Decorator that marks an method as a quantum operation.

    Methods marked as quantum operations are moved into the `BASE_OPS` dictionary
    of the `QuantumOperations` class they are defined in at the end of class
    creation.

    Quantum operations do not take an initial `self` parameter.

    Arguments:
       f:
           The method to decorate. `f` should take as positional arguments the
           qubits to operate on. It may take additional arguments that
           specify other parameters of the operation.

    Returns:
        The decorated method.
    """
    f._quantum_op = True
    return f


# Thread-local list of actice quantum operation contexts:

_QUANTUM_OPERATION = threading.local()
_QUANTUM_OPERATION.contexts = []


def current_quantum_operations() -> QuantumOperations:
    """Return the current quantum operations or raise an error if there are None.

    Returns:
        The current quantum operations.

    Raises:
        RuntimeError:
            If there is no quantum operations context set.
    """
    if not _QUANTUM_OPERATION.contexts:
        raise RuntimeError("No quantum operations context is currently set.")
    return _QUANTUM_OPERATION.contexts[-1]


class _CurrentQuantumOperation:
    """Proxy for accessing the current quantum operations.

    Attribute and item lookups on this proxy object return the
    same attribute or item from the current quantum operations,
    or raise an error if there are no current quantum operations.

    This class is only accessed via the ``qop`` variable which
    holds a singleton instance of the class.
    """

    def __getattr__(self, name):
        qop = current_quantum_operations()
        return getattr(qop, name)

    def __getitem__(self, name):
        qop = current_quantum_operations()
        return qop[name]


qop = _CurrentQuantumOperation()


class _DSLBuiltinOperations:
    """A convenience class for accessing explicitly supported LabOne Q DSL operations.

    The DSL operations listed here are typically functions from
    `laboneq.dsl.experiment.builtins`.

    This class is only accessed via the ``dsl`` variable which holds a
    singleton instance of the class.
    """

    acquire_loop_rt = staticmethod(builtins.acquire_loop_rt)
    delay = staticmethod(builtins.delay)
    measure = staticmethod(builtins.measure)
    play = staticmethod(builtins.play)
    reserve = staticmethod(builtins.reserve)
    section = staticmethod(builtins.section)
    sweep = staticmethod(builtins.sweep)

    pulse_library = pulse_library


dsl = _DSLBuiltinOperations()


class QuantumOperations:
    """Quantum operations for a given qubit type.

    Attributes:
        QUBIT_TYPE:
            (class attribute) The class of qubits supported by this set of
            operations.
        BASE_OPS:
            (class attribute) A dictionary of names and functions that define
            the base operations provided.
    """

    QUBIT_TYPE: Type[Qubit] = None
    BASE_OPS: dict[str, Callable] = None

    def __init__(self):
        if self.QUBIT_TYPE is None:
            raise ValueError(
                "Sub-classes of QuantumOperations must set the qubit type."
            )

        self._ops = {}
        self._uid_counts = defaultdict(int)

        for name, f in self.BASE_OPS.items():
            self.register(f, name=name)

    def __init_subclass__(cls, **kw):
        """Move any quantum operations into BASE_OPS."""
        if cls.BASE_OPS is None:
            cls.BASE_OPS = {}

        quantum_ops = {
            k: v for k, v in cls.__dict__.items() if getattr(v, "_quantum_op", False)
        }
        for k in quantum_ops:
            delattr(cls, k)
        cls.BASE_OPS = {**cls.BASE_OPS, **quantum_ops}

        super().__init_subclass__(**kw)

    def __enter__(self):
        """Enter a context with these quantum operations."""
        _QUANTUM_OPERATION.contexts.append(self)
        return self

    def __exit__(self, *exc):
        """Exit a context with these quantum operations."""
        quantum_ops = _QUANTUM_OPERATION.contexts.pop()
        assert (
            quantum_ops is self
        ), "Unexpected quantum operations when exiting operations context"

    def __getattr__(self, name):
        """Retrieve an operation."""
        op = self._ops.get(name, None)
        if op is None:
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute {name!r}"
            )
        return op

    def __getitem__(self, name):
        """Retrieve an operation."""
        return self._ops[name]

    def __contains__(self, name):
        """Return true if the set of operations contains the given name."""
        return name in self._ops

    def __dir__(self):
        """Return the attributes these quantum operations."""
        return sorted(super().__dir__() + list(self._ops.keys()))

    def _uid_generator(self, prefix: str) -> str:
        """Generate a unique id.

        Generates a unique identifier (in the context of this quantum
        operations object) by appending a count to a given prefix.

        Arguments:
            prefix:
                The prefix for the unique identifier.

        Returns:
            A unique identifier.
        """
        count = self._uid_counts[prefix]
        self._uid_counts[prefix] += 1
        return f"{prefix}_{count}"

    def register(self, f: Callable, name: str = None):
        """Registers a quantum operation.

        The given operation is wrapped in a `Operation` instance
        and added to this set of operations.

        Arguments:
            f:
                The function to register as a quantum operation.
                `f` should take as positional arguments the
                qubits to operate on. It may take additional
                arguments that specify other parameters of the
                operation.
            name:
                The name of the operation. Defaults to `f.__name__`.
        """
        name = name if name is not None else f.__name__
        self._ops[name] = Operation(f, name, self)

    def build(self, exp_func: Callable, *args, uid: str = None, **kw) -> Experiment:
        """Build an experiment with these QuantumOperations.

        Arguments:
            exp_func:
                The function that builds the experiment.
                The `*args` and `**kw` are passed directly to
                this function.
            uid:
                A name for this experiment. Defaults to
                `exp_func.__name__` plus a unique number if
                no name is given.

        Returns:
            A LabOne Q experiment.
        """
        if uid is None:
            uid = self._uid_generator(exp_func.__name__)

        with self:
            with builtins.experiment(uid=uid) as exp:
                exp_func(*args, **kw)

        return exp


class Operation:
    """An operation on one or more qubits.

    Arguments:
        op:
            The callable that implements the operation.
            `op` should take as positional arguments the
            qubits to operate on. It may take additional
            arguments that specify other parameters of the
            operation.
        op_name:
            The name of the operation (usually the same as that
            of the implementing function).
        quantum_ops:
            The quantum operations object the operation is for.
        kw:
            Dictionary of pre-filled keyword arguments for the operation.
    """

    def __init__(
        self, op: Callable, op_name: str, quantum_ops: QuantumOperations, kw=None
    ):
        self._op = op
        self._op_name = op_name
        self._quantum_ops = quantum_ops
        self._partial_kw = kw if kw is not None else {}

    def __call__(self, *args, **kw) -> Section:
        """Build a section using the operation.

        The operation is called in the context of a pre-built
        section instance.

        The UID of the section is generated with the name of the operation
        as a prefix and a unique count as a suffix.

        Arguments:
            *args:
                Positional arguments to the operation.
            **kw:
                Keyword parameters for the operation.

        Returns:
            A LabOne Q section built by the operation.
        """
        uid = self._quantum_ops._uid_generator(self._op_name)
        with dsl.section(uid=uid) as op_section:
            self._op(*args, **self._partial_kw, **kw)
        return op_section

    def partial(self, **kw) -> Operation:
        """Return a copy of the operation with some or all keyword parameters already set.

        Arguments:
            **kw:
                The keyword parameters to supply.

        Returns:
            A operation with some or all keyword parameters set.
        """
        partial_kw = {**self._partial_kw, **kw}
        return Operation(self._op, self._op_name, self._quantum_ops, partial_kw)

    @property
    def op(self) -> Callable:
        """Return the implementation of the operation.

        Returns:
            The function implementing the operation.
        """
        return self._op

    @property
    def src(self) -> str:
        """Return the source code of the underlying operation.

        Returns:
            The source code of the underlying operation.
        """
        src = inspect.getsource(self._op)
        return textwrap.dedent(src)
