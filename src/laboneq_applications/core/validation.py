# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Utilities for the LabOne Q Applications library."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np
from laboneq.dsl.quantum.qpu import QPU
from laboneq.dsl.quantum.quantum_element import QuantumElement
from laboneq.workflow import task
from laboneq.workflow.tasks.run_experiment import RunExperimentResults
from numpy import ndarray
from numpy.typing import ArrayLike, NDArray
from typing_extensions import deprecated

from laboneq_applications.conversion import as_array

if TYPE_CHECKING:
    from laboneq.dsl.quantum.qpu_topology import TopologyEdge


@deprecated("'_is_sequence_of_numbers_or_nparray' is deprecated and will be removed.")
def _is_sequence_of_numbers_or_nparray(obj: object) -> bool:
    return isinstance(obj, ndarray) or (
        bool(obj)
        and isinstance(obj, Sequence)
        and all(isinstance(item, (float, int)) for item in obj)
    )


@deprecated("'_is_sequence_of_quantum_elements' is deprecated and will be removed.")
def _is_sequence_of_quantum_elements(obj: object) -> bool:
    return isinstance(obj, QuantumElement) or (
        isinstance(obj, Sequence)
        and all(isinstance(item, QuantumElement) for item in obj)
    )


@deprecated(
    "`validate_and_convert_sweeps_to_arrays` is deprecated. "
    "Use `laboneq_applications.conversion.as_array` instead.",
    category=FutureWarning,
)
def validate_and_convert_sweeps_to_arrays(sweep_points: ArrayLike) -> NDArray:
    """Return the sweep points as a 1D array, or a 2D array with one row per qubit.

    !!! version-changed "Deprecated in version 26.10.0"
        Use `laboneq_applications.conversion.as_array` instead.
    """
    return as_array(sweep_points)


@task(save=False)
@deprecated(
    "`validate_length_qubits_sweeps()` is deprecated. Use "
    "`laboneq_applications.conversion.sweeps_per_quantum_element()` instead.",
    category=FutureWarning,
)
def validate_length_qubits_sweeps(
    qubits: QuantumElement | Sequence[QuantumElement],
    sweep_points: Sequence[float]
    | Sequence[Sequence[float] | ndarray]
    | ndarray
    | None = None,
) -> (
    tuple[Sequence[QuantumElement], Sequence[Sequence[float] | ndarray] | ndarray]
    | Sequence[QuantumElement]
):
    """Validate the length of the qubits and sweep points.

    !!! version-changed "Deprecated in version 26.10.0"
        Use `laboneq_applications.conversion.sweeps_per_quantum_element` instead.

    Checks the following conditions:

    - `qubits` must be a `QuantumElement` or a sequence of `QuantumElement`.
    - If a single qubit is passed, the sweep points must be a list or array of
      numbers.
    - The qubits and sweep points must have the same length.
    - All elements of the sweep points must be lists of numbers or arrays.

    If the conditions are met, returns the qubits and sweep points.

    Args:
        qubits: Either a single `QuantumElement` or a list of `QuantumElement`.
        sweep_points: The sweep points for each qubit. If `qubits` is a
            single `QuantumElement`, `sweep_points` must be a list of numbers
            or an array. Otherwise it must be a list of lists of numbers or arrays.
            The sweep_points can also be None, in which case the function returns
            the validated qubits

    Returns:
        A tuple containing the validated qubits and sweep points or only the validated
            qubits if sweep_points is None.

    Raises:
        ValueError: If the conditions are not met.

    """
    if not _is_sequence_of_quantum_elements(qubits):
        raise ValueError(
            "Qubits must be a QuantumElement or a sequence of QuantumElements.",
        )

    if sweep_points is None:
        return qubits

    if not isinstance(qubits, Sequence):
        if (
            not _is_sequence_of_numbers_or_nparray(sweep_points)
            or np.ndim(sweep_points) != 1
        ):
            raise ValueError(
                "If a single qubit is passed, the sweep points must be an array or a "
                "list of numbers.",
            )
    else:
        if len(qubits) != len(sweep_points):
            raise ValueError("Length of qubits and sweep points must be the same.")
        if not all(_is_sequence_of_numbers_or_nparray(swpts) for swpts in sweep_points):
            raise ValueError(
                "All elements of sweep points must be arrays or lists of numbers.",
            )
    return qubits, sweep_points


@task(save=False)
@deprecated(
    "`convert_qubits_sweeps_to_lists()` is deprecated. Use "
    "`laboneq_applications.conversion.sweeps_per_quantum_element()` instead.",
    category=FutureWarning,
)
def convert_qubits_sweeps_to_lists(
    qubits: QuantumElement | Sequence[QuantumElement],
    sweep_points: Sequence[float]
    | Sequence[Sequence[float] | ndarray]
    | ndarray
    | None = None,
) -> (
    tuple[Sequence[QuantumElement], Sequence[Sequence[float] | ndarray] | ndarray]
    | Sequence[QuantumElement]
):
    """Convert the qubits and sweep points to lists.

    !!! version-changed "Deprecated in version 26.10.0"
        Use `laboneq_applications.conversion.sweeps_per_quantum_element` instead.

    Checks the following conditions:

    - `qubits` must be a `QuantumElement` or a sequence of `QuantumElement`.
    - All elements of the sweep points must be lists of numbers or arrays.

    If a single qubit is passed, converts it and the sweep points (if not `None`)
    to a list. If the conditions are met, returns the qubits and sweep points.

    Args:
        qubits: Either a single `QuantumElement` or a list of `QuantumElement`.
        sweep_points: The sweep points for each qubit. If `qubits` is a
            single `QuantumElement`, `sweep_points` must be a list of numbers
            or an array. Otherwise it must be a list of lists of numbers or arrays.
            The sweep_points can also be None, in which case the function returns
            the validated and converted qubits

    Returns:
        A tuple containing the validated qubits and sweep points or only the validated
            and converted qubits if sweep_points is None.

    Raises:
        ValueError: If the conditions are not met.
    """
    # Ensure qubits and sweep points are lists, and if not, convert them to lists
    if not _is_sequence_of_quantum_elements(qubits):
        raise ValueError(
            "Qubits must be a QuantumElement or a sequence of QuantumElements.",
        )
    if sweep_points is None:
        if isinstance(qubits, QuantumElement):
            qubits = [qubits]
        return qubits

    if not isinstance(qubits, Sequence):
        qubits = [qubits]
        sweep_points = [sweep_points]
    elif not all(_is_sequence_of_numbers_or_nparray(swpts) for swpts in sweep_points):
        raise ValueError(
            "All elements of sweep points must be arrays or lists of numbers.",
        )
    return qubits, sweep_points


@task(save=False)
@deprecated(
    "`validate_and_convert_qubits_sweeps()` is deprecated. Use "
    "`laboneq_applications.conversion.sweeps_per_quantum_element()` instead.",
    category=FutureWarning,
)
def validate_and_convert_qubits_sweeps(
    qubits: QuantumElement | Sequence[QuantumElement],
    sweep_points: Sequence[float]
    | Sequence[Sequence[float] | ndarray]
    | ndarray
    | None = None,
) -> (
    tuple[Sequence[QuantumElement], Sequence[Sequence[float] | ndarray] | ndarray]
    | Sequence[QuantumElement]
):
    """Validate the qubits and sweep points, and return them as lists.

    !!! version-changed "Deprecated in version 26.10.0"
        Use `laboneq_applications.conversion.sweeps_per_quantum_element` instead.

    A single qubit needs a 1D sweep, and several qubits need one row of sweep points
    per qubit.

    Returns:
        A tuple containing the validated qubits and sweep points or only the validated
            and converted qubits if sweep_points is None.

    """
    if not _is_sequence_of_quantum_elements(qubits):
        raise ValueError(
            "Qubits must be a QuantumElement or a sequence of QuantumElements.",
        )
    if sweep_points is None:
        return [qubits] if isinstance(qubits, QuantumElement) else qubits

    sweep_points = as_array(sweep_points)
    if isinstance(qubits, QuantumElement):
        if sweep_points.ndim != 1:
            raise ValueError(
                "If a single qubit is passed, the sweep points must be an array or a "
                "list of numbers.",
            )
        return [qubits], [sweep_points]
    if len(qubits) != len(sweep_points):
        raise ValueError("Length of qubits and sweep points must be the same.")
    if sweep_points.ndim != 2:  # noqa: PLR2004
        raise ValueError(
            "All elements of sweep points must be arrays or lists of numbers.",
        )
    return qubits, sweep_points


@deprecated(
    "`validate_and_convert_single_qubit_sweeps()` is deprecated. Use "
    "`laboneq_applications.conversion.sweeps_per_quantum_element()` instead.",
    category=FutureWarning,
)
def validate_and_convert_single_qubit_sweeps(
    qubit: QuantumElement,
    sweep_points: Sequence[float] | Sequence[Sequence[float]] | ndarray | None = None,
) -> tuple[QuantumElement, Sequence[float] | ndarray] | QuantumElement:
    """Validates that single qubit is passed and converts a 1D sweep to array.

    !!! version-changed "Deprecated in version 26.10.0"
        Use `laboneq_applications.conversion.sweeps_per_quantum_element` instead.

    Args:
        qubit: A single `QuantumElement`.
        sweep_points: The sweep points for a qubit.

    Returns:
        A tuple containing the qubit and sweep points as array, or
            a single `qubit` if `sweep_points` is not passed.
    """
    if not isinstance(qubit, QuantumElement):
        raise TypeError("Only a single qubit is supported.")

    if sweep_points is not None:
        sweep_points = as_array(sweep_points)
        if sweep_points.ndim != 1:
            raise ValueError(
                "If a single qubit is passed, the sweep points must be an array or a "
                "list of numbers.",
            )
        return qubit, sweep_points

    return qubit


def validate_result(result: RunExperimentResults) -> None:
    """Checks that result is an instance of RunExperimentResults.

    Args:
        result: the acquired results

    Raises:
        TypeError: If result is not an instance of RunExperimentResults.
    """
    if not (
        isinstance(result, RunExperimentResults)
        or (
            isinstance(result, Sequence)
            and all(isinstance(item, RunExperimentResults) for item in result)
        )
    ):
        raise TypeError(
            "The result must be either an instance of RunExperimentResults "
            "or a sequence of RunExperimentResults."
        )


@task(save=False)
def validate_flux_driven_edges(edges: list[TopologyEdge]) -> None:
    """Checks that the quantum element of every edge has a flux line.

    The experiments drive the flux line of the element on the edge, so any element
    that has one works, not only a `TunableCoupler`.

    Args:
        edges: The edges to check.

    Raises:
        ValueError: If an edge has no quantum element, or its quantum element has no
            flux signal.
    """
    for edge in edges:
        key = f"({edge.tag}, {edge.source_node.uid}, {edge.target_node.uid})"
        element = edge.quantum_element
        if element is None:
            raise ValueError(f"The edge {key} has no quantum element.")
        if "flux" not in element.signals:
            raise ValueError(
                f"The quantum element {element.uid} on the edge {key} has no flux "
                f"signal and its flux line cannot be driven."
            )


@task(save=False)
def validate_parallel_two_qubit_experiment(
    qpu: QPU,
    qubit_pair_uids: list[list[str]],
) -> list[QuantumElement]:
    """Checks that an experiment can be run in parallel on the qubit pairs.

    Args:
        qpu: The QPU where the topology is stored.
        qubit_pair_uids: The pairs of UIDs for the qubits that will be used as source
            and target node from the edge.

    Returns:
        The flat list of qubit objects corresponding to `qubit_pair_uids`.

    Raises:
        ValueError: If a qubit is passed more than once in the full list.
            (One cannot act on them in parallel.)
    """
    uids = list(chain.from_iterable(qubit_pair_uids))
    if len(uids) != len(set(uids)):
        raise ValueError(
            "Quantum elements appear more than once in the edges. "
            "Calibration cannot be parallelized."
        )
    return qpu[uids]
