# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Utilities for the LabOne Q Applications library."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import chain

import numpy as np
from laboneq.dsl.quantum.qpu import QPU
from laboneq.dsl.quantum.qpu_topology import TopologyEdge
from laboneq.dsl.quantum.quantum_element import QuantumElement
from laboneq.workflow import task
from laboneq.workflow.tasks.run_experiment import RunExperimentResults
from numpy import ndarray


def _is_sequence_of_numbers_or_nparray(obj: object) -> bool:
    return isinstance(obj, ndarray) or (
        bool(obj)
        and isinstance(obj, Sequence)
        and all(isinstance(item, (float, int)) for item in obj)
    )


def _is_sequence_of_numbers_or_sequences_or_nparrays(obj: object) -> bool:
    return isinstance(obj, ndarray) or (
        bool(obj)
        and isinstance(obj, Sequence)
        and all(isinstance(item, (float, int)) for item in obj)
    )


def _is_sequence_of_quantum_elements(obj: object) -> bool:
    return isinstance(obj, QuantumElement) or (
        isinstance(obj, Sequence)
        and all(isinstance(item, QuantumElement) for item in obj)
    )


def validate_and_convert_sweeps_to_arrays(
    sweep_points: Sequence[float] | Sequence[Sequence[float] | ndarray] | ndarray,
) -> Sequence[Sequence[float] | ndarray] | ndarray:
    """Convert sweep_points into numpy arrays.

    Check for the following conditions:
        - sweep_points must be a list or an array
        - All elements of sweep points must be lists of numbers or arrays.

    Args:
        sweep_points: The sweep points to be converted into numpy arrays.

    Returns:
        The sweep points as a numpy array if sweep_points was a list of numbers, or
        as a list of numpy arrays, if sweep_points was a list of lists of numbers.

    Raises:
        ValueError: If the conditions are not met.
    """
    if not isinstance(sweep_points, (Sequence, ndarray)):
        raise TypeError("The sweep points must be an array or a list.")

    if not all(isinstance(swpts, (Sequence, ndarray)) for swpts in sweep_points):
        # the elements of sweep_points are not iterables
        if not _is_sequence_of_numbers_or_nparray(sweep_points):
            raise ValueError(
                "The sweep points must be an array or a list of numbers.",
            )
        sweep_points = np.array(sweep_points)
    else:
        if not all(_is_sequence_of_numbers_or_nparray(swpts) for swpts in sweep_points):
            raise ValueError(
                "All elements of sweep points must be arrays or lists of numbers.",
            )
        # we do not do list comprehension here in order to keep the original type of
        # sweep_points, which might be an array
        for i in range(len(sweep_points)):
            sweep_points[i] = np.array(sweep_points[i])

    return sweep_points


@task(save=False)
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

    Check for the following conditions:
        - qubits must be a QuantumElement or a sequence of QuantumElement.
        - If a single qubit is passed, the sweep points must be a list or array of
            numbers.
        - Length of qubits and sweep points must be the same.
        - All elements of sweep points must be lists of numbers or arrays.
    If the conditions are met, return the qubits and sweep points.

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
        if not _is_sequence_of_numbers_or_nparray(sweep_points):
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

    Check for the following conditions:
        - qubits must be a QuantumElement or a sequence of QuantumElement.
        - All elements of sweep points must be lists of numbers or arrays.
    If single qubit is passed, convert it and the sweep points to a list (if sweep
        points is not None).
    If the conditions are met, return the qubits and sweep points.

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
    """Validate and convert the qubits and sweep points.

    Validates the length of the qubits and sweep points by calling
    validate_length_qubits_sweeps. See docstring there for details.

    Then converts the qubits and sweep points to lists. by calling
    convert_qubits_sweeps_to_lists. See docstring there for details.

    Returns:
        A tuple containing the validated qubits and sweep points or only the validated
        and converted qubits if sweep_points is None.

    """
    # Make sure the sweep points are arrays or lists of arrays
    if sweep_points is not None:
        sweep_points = validate_and_convert_sweeps_to_arrays(sweep_points)

    # Make sure the length of qubits and sweep points is the same
    qubits_sweep_points = validate_length_qubits_sweeps(qubits, sweep_points)
    qubits, sweep_points = (
        (qubits, sweep_points)
        if isinstance(qubits_sweep_points, Sequence) == 1
        else (qubits, None)
    )

    # Ensure qubits and sweep points are lists, and if not, convert them to lists
    return convert_qubits_sweeps_to_lists(qubits, sweep_points)


def validate_and_convert_single_qubit_sweeps(
    qubits: QuantumElement | Sequence[QuantumElement],
    sweep_points: Sequence[float]
    | Sequence[Sequence[float] | ndarray]
    | ndarray
    | None = None,
) -> tuple[QuantumElement, Sequence[float] | ndarray] | QuantumElement:
    """Converts and validates qubits, sweep points and the experiment result.

    Check for the following conditions:
        - type of qubits must be `QuantumElement` or a subclass of it.
        - sweep_points is a list or array with all its elements lists of numbers or
            arrays (see validate_and_convert_sweeps_to_arrays)
        - the length of qubits and sweep_points is the same
            (see validate_length_qubits_sweeps)
    If sweep_points was a list of numbers, sweep_points is converted into a numpy array.
    If sweep_points was a list of lists of numbers, every elements in sweep_points is
    converted into a numpy array (see validate_and_convert_sweeps_to_arrays).

    Args:
        qubits: Either a single `QuantumElement` or a list of `QuantumElement`.
        sweep_points: The sweep points for each qubit.

    Returns:
        A tuple containing the validated qubits and sweep points or only the validated
        and converted qubits if sweep_points is None.
    """
    if not isinstance(qubits, QuantumElement):
        raise TypeError("Only a single qubit is supported.")

    if sweep_points is not None:
        sweep_points = validate_and_convert_sweeps_to_arrays(sweep_points)
        qubits, sweep_points = validate_length_qubits_sweeps(qubits, sweep_points)
        return qubits, sweep_points

    return qubits


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
def validate_and_extract_edges_from_qubit_pairs(
    qpu: QPU,
    tag: str,
    qubit_pair_uids: list[list[str]],
    *,
    element_class: type[QuantumElement] | None = None,
) -> list[TopologyEdge]:
    """Extracts all edges of type tag in the QPU with the qubit pair passed.

    Args:
        qpu: The QPU where the topology is stored.
        qubit_pair_uids: The pairs of UIDs for the qubits that will be used as source
            and target node from the edge, respectively.
        tag: The edge tag to be searched.
        element_class: The quantum element class (optional). By default, the quantum
            element class is None.

    Returns:
        The list of extracted edges.

    Raises:
        TypeError: If the elements in the extracted edges are not of the type specified
            in `element_class`.
    """
    edges = [qpu.topology[tag, q0, q1] for q0, q1 in qubit_pair_uids]

    for e in edges:
        if not isinstance(e.quantum_element, element_class):
            raise TypeError(
                f"Quantum element on edge "
                f"({e.tag}, {e.source_node.uid}, {e.target_node.uid}) has "
                f"invalid type: {type(e.quantum_element)}. "
                f"Expected type: {element_class}."
            )

    return edges


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
    all_qubit_uids = list(chain.from_iterable(qubit_pair_uids))
    all_qubits = qpu[all_qubit_uids]

    # check that both lists are unique to avoid clashing
    all_qubit_are_unique = len(all_qubit_uids) == len(set(all_qubit_uids))
    if not all_qubit_are_unique:
        raise ValueError(
            "Quantum elements appear more than once in the edges. "
            "Calibration cannot be parallelized."
        )

    return all_qubits
