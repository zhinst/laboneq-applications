"""Core classes for defining sets of quantum operations on qubits."""

from __future__ import annotations

import contextlib
import inspect
import textwrap
from typing import TYPE_CHECKING, Callable, ClassVar

from laboneq.dsl.experiment import builtins
from laboneq.simple import (
    ExecutionType,
    QuantumElement,
    Section,
    SectionAlignment,
    pulse_library,
)

from .build_experiment import _qubits_from_args

if TYPE_CHECKING:
    from laboneq.dsl.experiment.pulse import Pulse


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


class _PulseCache:
    """A cache for pulses to ensure that each unique pulse is only created once."""

    GLOBAL_CACHE: ClassVar[dict[tuple, Pulse]] = {}

    def __init__(self, cache: dict | None = None):
        if cache is None:
            cache = {}
        self.cache = cache

    @classmethod
    def experiment_or_global_cache(cls) -> _PulseCache:
        """Return an pulse cache.

        If there is an active experiment context, return its cache. Otherwise
        return the global pulse cache.
        """
        context = builtins.current_experiment_context()
        if context is None:
            return cls(cls.GLOBAL_CACHE)
        if not hasattr(context, "_pulse_cache"):
            context._pulse_cache = cls()
        return context._pulse_cache

    @classmethod
    def reset_global_cache(cls) -> None:
        cls.GLOBAL_CACHE.clear()

    def get(self, name: str, function: str, parameters: dict) -> Pulse | None:
        """Return the cache pulse or `None`."""
        key = (function, name, tuple(sorted(parameters.items())))
        return self.cache.get(key, None)

    def store(self, pulse: Pulse, name: str, function: str, parameters: dict) -> None:
        """Store the given pulse in the cache."""
        key = (function, name, tuple(sorted(parameters.items())))
        self.cache[key] = pulse


def create_pulse(
    parameters: dict,
    overrides: dict | None = None,
    name: str | None = None,
) -> Pulse:
    """Create a pulse from the given parameters and parameter overrides.

    The parameters are dictionary that contains:

      - a key `"function"` that specifies which function from the LabOne Q
        `pulse_library` to use to construct the pulse.
      - any other parameters required by the given pulse function.

    Arguments:
        parameters:
            A dictionary of pulse parameters. If `None`, then the overrides
            must completely define the pulse.
        overrides:
            A dictionary of overrides for the pulse parameters.
            If the overrides changes the pulse function, then the
            overrides completely replace the existing pulse parameters.
            Otherwise they extend or override them.
            The dictionary of overrides may contain sweep parameters.
        name:
            The name of the pulse. This is used as a prefix to generate the
            pulse `uid`.

    Returns:
        pulse:
            The pulse described by the parameters.
    """
    if overrides is None:
        overrides = {}
    if "function" in overrides and overrides["function"] != parameters["function"]:
        parameters = overrides.copy()
    else:
        parameters = {**parameters, **overrides}

    function = parameters.pop("function")

    pulse_function = getattr(dsl.pulse_library, function, None)
    if pulse_function is None:
        raise ValueError(f"Unsupported pulse function {function!r}.")

    if name is None:
        name = "unnamed"

    pulse_cache = _PulseCache.experiment_or_global_cache()
    pulse = pulse_cache.get(name, function, parameters)
    if pulse is None:
        pulse = pulse_function(uid=dsl.uid(name), **parameters)
        pulse_cache.store(pulse, name, function, parameters)

    return pulse


class _DSLBuiltinOperations:
    """A convenience class for accessing explicitly supported LabOne Q DSL operations.

    The DSL operations listed here are typically functions from
    `laboneq.dsl.experiment.builtins`.

    This class is only accessed via the ``dsl`` variable which holds a
    singleton instance of the class.
    """

    acquire_loop_rt = staticmethod(builtins.acquire_loop_rt)
    add = staticmethod(builtins.add)
    delay = staticmethod(builtins.delay)
    experiment = staticmethod(builtins.experiment)
    measure = staticmethod(builtins.measure)
    play = staticmethod(builtins.play)
    reserve = staticmethod(builtins.reserve)
    section = staticmethod(builtins.section)
    sweep = staticmethod(builtins.sweep)
    uid = staticmethod(builtins.uid)

    pulse_library = pulse_library


dsl = _DSLBuiltinOperations()


UNSET = object()  # default sentinel


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

    QUBIT_TYPE: type[QuantumElement] = None
    BASE_OPS: dict[str, Callable] = None

    def __init__(self):
        if self.QUBIT_TYPE is None:
            raise ValueError(
                "Sub-classes of QuantumOperations must set the qubit type.",
            )

        self._ops = {}

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

    def __getattr__(self, name: str):
        """Retrieve an operation."""
        op = self._ops.get(name, None)
        if op is None:
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute {name!r}",
            )
        return op

    def __getitem__(self, name: str):
        """Retrieve an operation."""
        return self._ops[name]

    def __setitem__(self, name: str, f: Operation | Callable):
        """Replace or register an operation."""
        if isinstance(f, Operation):
            self._ops[name] = f
        else:
            self.register(f, name=name)

    def __contains__(self, name: str):
        """Return true if the set of operations contains the given name."""
        return name in self._ops

    def __dir__(self):
        """Return the attributes these quantum operations."""
        return sorted(super().__dir__() + list(self._ops.keys()))

    def register(self, f: Callable, name: str | None = None) -> None:
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
        self,
        op: Callable,
        op_name: str,
        quantum_ops: QuantumOperations,
        kw: dict | None = None,
    ):
        self._op = op
        self._op_name = op_name
        self._quantum_ops = quantum_ops
        self._partial_kw = kw if kw is not None else {}
        self._section_kw = {}
        self._omit_section = False

    def __call__(self, *args, **kw) -> Section | None:
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
            If `.section(omit=True)` is used, the operation
            will return `None`.
        """
        qubits = _qubits_from_args(args)
        qubits_with_incorrect_type = [
            q.uid for q in qubits if not isinstance(q, self._quantum_ops.QUBIT_TYPE)
        ]

        if qubits_with_incorrect_type:
            raise TypeError(
                f"Quantum operation {self._op_name!r} was passed the following"
                f" qubits that are not of type {self._quantum_ops.QUBIT_TYPE.__name__}:"
                f" {', '.join(qubits_with_incorrect_type)}",
            )

        section_name = "_".join([self._op_name] + [q.uid for q in qubits])

        if not self._omit_section:
            maybe_section = dsl.section(
                name=section_name,
                **self._section_kw,
            )
        else:
            maybe_section = contextlib.nullcontext()

        with maybe_section as op_section:
            if not self._omit_section:
                self._reserve_signals(qubits)
            self._op(self._quantum_ops, *args, **self._partial_kw, **kw)

        return op_section

    def _reserve_signals(self, qubits: list[QuantumElement]) -> None:
        """Reserve all the signals of a list of qubits."""
        for q in qubits:
            for signal in q.signals.values():
                dsl.reserve(signal)

    def partial(self, **kw) -> Operation:
        """Return a copy of the operation with the specified parameters set.

        Only keyword arguments may be specified.

        Arguments:
            **kw (dict):
                The keyword parameters to supply.

        Returns:
            An operation with some or all keyword parameters set.
        """
        partial_kw = {**self._partial_kw, **kw}
        return Operation(self._op, self._op_name, self._quantum_ops, partial_kw)

    def section(  # noqa: PLR0913
        self,
        *,
        omit: bool = UNSET,
        alignment: SectionAlignment = UNSET,
        execution_type: ExecutionType | None = UNSET,
        length: float | None = UNSET,
        play_after: str | Section | list[str | Section] | None = UNSET,
        on_system_grid: bool = UNSET,
    ) -> Operation:
        """Return a copy of the operation with the given section parameters set.

        Arguments:
            omit:
                Omit creating a section and add the contents of the operation directly
                to an existing section. This is intended to reduce the number of
                sections created when one operation consists entirely of calling another
                operation. In other cases it should be used with care since omitting a
                section may affect the generated signals.

                Calling an operation with omit set will return `None`.
            alignment:
                Specifies the time alignment of operations and sections within
                this section.
            execution_type:
                Whether the section is near-time or real-time. By default the
                execution type is automatically determined by the compiler.
            length:
                Minimum length of the section in seconds.
            play_after:
                A list of sections that must complete before this section
                may be played.
            on_system_grid:
                If True, the section boundaries are always rounded to the system grid,
                even if the contained signals would allow for tighter alignment.

        Any section arguments not specified are left as they were in the existing
        operation.

        Returns:
            An operation with the given section keyword arguments set.
        """
        op = Operation(self._op, self._op_name, self._quantum_ops, self._partial_kw)
        op._section_kw = self._section_kw.copy()
        op._omit_section = self._omit_section

        if omit is not UNSET:
            op._omit_section = omit
        if alignment is not UNSET:
            op._section_kw["alignment"] = alignment
        if execution_type is not UNSET:
            op._section_kw["execution_type"] = execution_type
        if length is not UNSET:
            op._section_kw["length"] = length
        if play_after is not UNSET:
            op._section_kw["play_after"] = play_after
        if on_system_grid is not UNSET:
            op._section_kw["on_system_grid"] = on_system_grid
        return op

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
