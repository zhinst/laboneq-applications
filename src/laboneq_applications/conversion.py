# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Convert qubit and sweep point arguments into the forms experiments work with."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, overload

import numpy as np
from laboneq.dsl.quantum.quantum_element import QuantumElement

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


def as_quantum_element_list(quantum_elements: QuantumElements) -> list[QuantumElement]:
    """Return the quantum elements as a list, wrapping a single element in one."""
    if isinstance(quantum_elements, QuantumElement):
        return [quantum_elements]
    if not isinstance(quantum_elements, Sequence) or not all(
        isinstance(q, QuantumElement) for q in quantum_elements
    ):
        raise TypeError(
            "Qubits must be a QuantumElement or a sequence of QuantumElement."
        )
    return list(quantum_elements)


def as_single_quantum_element(quantum_elements: QuantumElements) -> QuantumElement:
    """Return the only quantum element, given bare or in a list of one."""
    quantum_elements = as_quantum_element_list(quantum_elements)
    if len(quantum_elements) != 1:
        raise TypeError("Only a single quantum element is supported.")
    return quantum_elements[0]


def as_array(sweep_points: QubitSweepPoints) -> NDArray:
    """Return the sweep points as a 1D or 2D array of numbers."""
    try:
        sweep_points = np.asarray(sweep_points)
    except ValueError as err:
        raise ValueError(
            "The sweep points of all qubits must have the same length."
        ) from err
    if sweep_points.ndim not in (1, 2) or sweep_points.dtype.kind not in "iuf":
        raise ValueError("The sweep points must be a 1D or 2D array of numbers.")
    return sweep_points


def _sweep_rows(sweep_points: QubitSweepPoints) -> list[NDArray]:
    """Return the sweep points as 1D rows, which may differ in length."""
    if not isinstance(sweep_points, Sequence) or all(
        np.isscalar(p) for p in sweep_points
    ):
        sweep = as_array(sweep_points)
        return [sweep] if sweep.ndim == 1 else list(sweep)
    rows = [as_array(row) for row in sweep_points]
    if any(row.ndim != 1 for row in rows):
        raise ValueError("The sweep points must be a 1D or 2D array of numbers.")
    return rows


@overload
def sweeps_per_quantum_element(
    quantum_elements: QuantumElements, sweep_points: QubitSweepPoints, /
) -> Iterator[tuple[QuantumElement, NDArray]]: ...


@overload
def sweeps_per_quantum_element(
    quantum_elements: QuantumElements,
    sweep_points: QubitSweepPoints,
    other_sweep_points: QubitSweepPoints,
    /,
) -> Iterator[tuple[QuantumElement, NDArray, NDArray]]: ...


def sweeps_per_quantum_element(
    quantum_elements: QuantumElements, *sweep_points: QubitSweepPoints
) -> Iterator[tuple[QuantumElement | NDArray, ...]]:
    """Yield each qubit and its sweep points.

    Each sweep needs one row per qubit, and the rows may differ in length. A 1D
    sweep counts as one row, so it works only with a single qubit.
    """
    quantum_elements_list = as_quantum_element_list(quantum_elements)
    sweeps = []
    for points in sweep_points:
        sweep = _sweep_rows(points)
        if len(quantum_elements_list) != len(sweep):
            raise ValueError(
                f"Each sweep needs one row per qubit, but the number of rows "
                f"({len(sweep)}) does not match the number of qubits "
                f"({len(quantum_elements_list)})."
            )
        sweeps.append(sweep)
    for i, q in enumerate(quantum_elements_list):
        yield q, *(sweep[i] for sweep in sweeps)
