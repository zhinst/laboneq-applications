"""Utilities for the LabOne Q Applications library."""

from __future__ import annotations

from collections.abc import Sequence

from laboneq.dsl.quantum.quantum_element import QuantumElement
from numpy import ndarray

from laboneq_applications.workflow import task


def _is_sequence_of_numbers_or_nparray(obj: object) -> bool:
    return (
        isinstance(obj, ndarray)
        or bool(obj)
        and isinstance(obj, Sequence)
        and all(isinstance(item, (float, int)) for item in obj)
    )


def _is_sequence_of_quantum_elements(obj: object) -> bool:
    return isinstance(obj, QuantumElement) or (
        isinstance(obj, Sequence)
        and all(isinstance(item, QuantumElement) for item in obj)
    )


@task
def validate_and_convert_qubits_sweeps(
    qubits: QuantumElement | Sequence[QuantumElement],
    sweep_points: Sequence[float]
    | Sequence[Sequence[float] | ndarray]
    | ndarray
    | None = None,
) -> (
    tuple[Sequence[QuantumElement], Sequence[Sequence[float] | ndarray]]
    | Sequence[QuantumElement]
):
    """Validate the qubits and sweep points.

    Check for the following conditions:
        - qubits must be a QuantumElement or a sequence of QuantumElement.
        - If a single qubit is passed, the sweep points must be a list of numbers.
        - Length of qubits and sweep points must be the same.
        - All elements of sweep points must be lists of numbers or arrays.
    If single qubit is passed, convert it to a list.
    If the conditions are met, return the qubits and sweep points.

    Args:
        qubits: Either a single `QuantumElement` or a list of `QuantumElement`.
        sweep_points: The sweep points for each qubit. If `qubits` is a
            single `QuantumElement`, `sweep_points` must be a list of numbers
            or an array. Otherwise it must be a list of lists of numbers or arrays.

    Returns:
        A tuple containing the validated qubits and sweep points.

    Raises:
        ValueError if the conditions are not met.

    """
    if not _is_sequence_of_quantum_elements(qubits):
        raise ValueError(
            "Qubits must be a QuantumElement or a sequence of QuantumElements.",
        )
    if sweep_points is None:
        if isinstance(qubits, QuantumElement):
            qubits = [qubits]
        return qubits

    if not isinstance(qubits, Sequence):
        if not _is_sequence_of_numbers_or_nparray(sweep_points):
            raise ValueError(
                "If a single qubit is passed, the sweep points must be a list"
                " of numbers.",
            )
        qubits = [qubits]
        sweep_points = [sweep_points]
    else:
        if len(qubits) != len(sweep_points):
            raise ValueError("Length of qubits and sweep points must be the same.")
        if not all(_is_sequence_of_numbers_or_nparray(amps) for amps in sweep_points):
            raise ValueError(
                "All elements of sweep points must be lists of numbers.",
            )
    return qubits, sweep_points
