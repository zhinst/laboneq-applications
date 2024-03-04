"""Build DSL experiments that use quantum operations on qubits."""

from __future__ import annotations

import functools
from typing import Any, Callable

from laboneq.core.exceptions import LabOneQException
from laboneq.dsl.experiment import builtins
from laboneq.dsl.quantum import QuantumElement
from laboneq.simple import (
    Calibration,
    Experiment,
    ExperimentSignal,
)


class ExperimentBuilder:
    """A builder for functions that create DSL experiments for qubits.

    The builder takes care of creating the DSL `Experiment` object
    including:

    - giving the experiment a name
    - adding the qubit signals as experiment signals
    - adding the qubit calibration to the experiment calibration

    Arguments:
        exp_func:
            The function that builds the experiment.

            When calling `__call__` the `*args` and `**kw`
            are passed directly to `exp_func`.
        name:
            A name for this experiment. Defaults to
            `exp_func.__name__` if no name is given.
    """

    def __init__(self, exp_func: Callable, name: str | None = None):
        if name is None:
            name = exp_func.__name__
        self.exp_func = exp_func
        self.name = name

    def __call__(self, *args, **kw):
        """Build the experiment.

        Arguments:
            *args:
                Positional arguments to pass to `exp_func`.
            **kw:
                Keyword arguments to pass to `exp_func`.

        Returns:
            A LabOne Q experiment.
        """
        qubits = _qubits_from_args(args)
        signals = _exp_signals_from_qubits(qubits)
        calibration = _calibration_from_qubits(qubits)

        with builtins.experiment(uid=self.name, signals=signals) as exp:
            self.exp_func(*args, **kw)

        exp.set_calibration(Calibration(calibration))

        return exp


def qubit_experiment(
    exp_func: Callable | None = None,
    name: str | None = None,
) -> Callable:
    """Decorator for functions that build experiments for qubits.

    Arguments:
        exp_func:
            The function that builds the experiment.
            The `*args` and `**kw` are passed directly to
            this function.
        name:
            A name for this experiment. Defaults to
            `exp_func.__name__` if no name is given.

    Returns:
        If `exp_func` is given, returns `exp_func` wrapped in an `ExperimentBuilder`.
        Otherwise returns a partial evaluation of `qubit_experiment` with the other
        parameters already set.

    Examples:
        @qubit_experiment
        def rx_exp(qops, q, angle):
            qops.rx(q, angle)

        @qubit_experiment(name="rx_exp")
        def my_exp(qops, q, angle):
            qops.rx(q, angle)
    """
    if exp_func is None:
        return functools.partial(qubit_experiment, name=name)

    builder = ExperimentBuilder(exp_func, name=name)

    @functools.wraps(exp_func)
    def build_qubit_experiment(*args, **kw) -> Experiment:
        return builder(*args, **kw)

    return build_qubit_experiment


def build(exp_func: Callable, *args, name: str | None = None, **kw) -> Experiment:
    """Build an experiment that accepts qubits as arguments.

    Arguments:
        exp_func:
            The function that builds the experiment.
            The `*args` and `**kw` are passed directly to
            this function.
        name:
            A name for this experiment. Defaults to
            `exp_func.__name__` if no name is given.
        *args:
            Positional arguments to pass to `exp_func`.
        **kw:
            Keyword arguments to pass to `exp_func`.

    Returns:
        A LabOne Q experiment.
    """
    builder = ExperimentBuilder(exp_func, name=name)
    return builder(*args, **kw)


def _qubits_from_args(args: tuple[Any]) -> list[QuantumElement]:
    """Return a list of qubits found in positional arguments."""
    qubits = []
    for arg in args:
        if isinstance(arg, QuantumElement):
            qubits.append(arg)
        elif isinstance(arg, (tuple, list)) and all(
            isinstance(x, QuantumElement) for x in arg
        ):
            qubits.extend(arg)
    return qubits


def _exp_signals_from_qubits(qubits: list[QuantumElement]) -> list[ExperimentSignal]:
    """Return a list of experiment signals from a list of qubits."""
    signals = []
    for qubit in qubits:
        for exp_signal in qubit.experiment_signals():
            if exp_signal in signals:
                msg = f"Signal with id {exp_signal.uid} already assigned."
                raise LabOneQException(msg)
            signals.append(exp_signal)
    return signals


def _calibration_from_qubits(
    qubits: list[QuantumElement],
) -> dict[str,]:
    """Return the calibration objects from a list of qubits."""
    calibration = {}
    for qubit in qubits:
        calibration.update(qubit.calibration())
    return calibration
