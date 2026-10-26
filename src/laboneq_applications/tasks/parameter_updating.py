# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the task for updating setup parameters."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import attrs
from laboneq import workflow
from laboneq.dsl.quantum import QPU, QuantumParameters

if TYPE_CHECKING:
    import uncertainties as unc

    from laboneq_applications.typing import QuantumElements


@workflow.task(save=False, hidden=True)
def _handle_deprecated_qubit_argument(
    qubits: QuantumElements | list[str] | str | None,
    qubit: QuantumElements | list[str] | str | None,
) -> QuantumElements | list[str] | str:
    """Resolve the deprecated singular `qubit` argument of a workflow.

    Entry task for workflows whose `qubit` argument was generalized to
    `qubits`: it accepts exactly one of the two arguments and returns the
    effective quantum elements. If the deprecated `qubit` is passed, a
    `FutureWarning` is emitted and its value is used as `qubits`.

    Args:
        qubits: The value of the new `qubits` argument, or `None`.
        qubit: The value of the deprecated `qubit` argument, or `None`.

    Returns:
        The effective `qubits` value.

    Raises:
        ValueError: If both or neither of the arguments are provided.
    """
    if qubit is not None:
        if qubits is not None:
            raise ValueError(
                "Only one of `qubits` and `qubit` may be passed. We recommend "
                "passing `qubits`; the `qubit` argument is deprecated."
            )
        warnings.warn(
            "The `qubit` argument is deprecated in v26.10.0 and will no "
            "longer be supported in v27.1.0. Please use `qubits` instead.",
            FutureWarning,
            stacklevel=2,
        )
        return qubit
    if qubits is None:
        raise ValueError("The `qubits` argument is required.")
    return qubits


def _valid_temporary_parameters(
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None,
) -> bool:
    """Returns True if temporary parameters are of the correct type, False otherwise."""
    if temporary_parameters is None:
        return True
    if not isinstance(temporary_parameters, dict):
        return False
    for key, value in temporary_parameters.items():
        if not isinstance(key, str | tuple):
            return False
        if isinstance(key, tuple):
            edge_key_tuple_length = 3
            if len(key) != edge_key_tuple_length:
                return False
            if not all(isinstance(item, str) for item in key):
                return False
        if not isinstance(value, dict | QuantumParameters):
            return False
    return True


@workflow.task
def update_qpu(
    qpu: QPU,
    parameters: dict[
        str | tuple[str, str, str],
        dict[str, dict[str, int | float | unc.core.Variable | None]],
    ],
    eval_flags: dict[str, dict[str, bool]] | None = None,
) -> None:
    """Updates the parameters of the quantum objects in the qpu.

    A quantum object is any object that has associated quantum parameters. This
    includes quantum elements and topology edges.

    Args:
        qpu: The qpu containing the quantum objects to be updated.
        parameters: Quantum object parameters and the new values to be updated.
            This dictionary has the following form:
            ```python
            {key: {param_name: param_value}}
            ```
        eval_flags: The dictionary of evaluation flags (optional). The keys are the
            qubit UIDs and the values are dictionaries of booleans. The `success` flag
            tells us whether the experiment is successful, and the `update` flag tells
            us whether the qubit parameter values have additionally been significantly
            updated. If both the `success` and `update` flags are `True`, then the
            qubit in the QPU will be updated.

    !!! note
        The key for a quantum element is the quantum element UID. The key for a
        topology edge is the tuple `(tag, source node UID, target node UID)`, as
        returned by the `qpu.topology.edge_keys()` method.
    """
    parameters_numeric = {}
    for key, params_dict in parameters.items():
        if len(params_dict) == 0:
            workflow.log(
                logging.WARNING,
                f"{key} could not be updated because its "
                f"parameters could not be extracted.",
            )
        if eval_flags and not eval_flags[key]["update"]:
            continue

        params_dict_numeric = {
            k: v.nominal_value if hasattr(v, "nominal_value") else v
            for k, v in params_dict.items()
        }
        parameters_numeric[key] = params_dict_numeric

    qpu.update(parameters_numeric)


@workflow.task
def update_qpu_edges(
    qpu: QPU,
    parameters: dict[
        str,
        dict[str, dict[str, int | float | unc.core.Variable | None]],
    ],
) -> None:
    """Updates the parameters of topology edges in the qpu.

    Behaves like the `update_qpu` task, but takes the edge keys as strings
    of the form ``"tag:source_node_uid:target_node_uid"`` instead of the
    tuples accepted by `qpu.update`. This allows analysis tasks to return the edge
    parameters in a form that is serializable by the logbook stores (which
    only support dictionaries with string keys).

    Args:
        qpu: The qpu containing the topology edges to be updated.
        parameters: Edge parameters and the new values to be updated.
            This dictionary has the following form:
            ```python
            {"tag:source_node_uid:target_node_uid": {param_name: param_value}}
            ```
    """
    parameters_tuple_keys = {}
    for key, params_dict in parameters.items():
        tag, source_uid, target_uid = key.split(":")
        parameters_tuple_keys[(tag, source_uid, target_uid)] = params_dict
    update_qpu(qpu, parameters_tuple_keys)


@workflow.task
def temporary_qpu(
    qpu: QPU,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
) -> QPU:
    """Modify the QPU temporarily with the given parameters.

    Args:
        qpu: The QPU to be temporarily modified.
        temporary_parameters: The parameters to be temporarily modified.
            If None, the QPU is returned as is.
            The dictionary has the following form:
            ```python
            {key: {"param": param_value}}
            ```
            or
            ```python
            {key: QuantumParameters}
            ```
            where `key` may be either a quantum element UID string or edge key tuple of
            the form `(tag, source node UID, target node UID)`.

    !!! note
        The quantum element attached to a topology edge cannot be temporarily modified.

    Returns:
        QPU: The QPU with the temporary parameters applied to each quantum element or
            edge.

    Raises:
        TypeError: If the temporary parameters have invalid type.
    """
    if not _valid_temporary_parameters(temporary_parameters):
        raise TypeError(
            f"The temporary parameters have invalid type: {type(temporary_parameters)}."
            f" Expected type:"
            f" dict[str | tuple[str, str, str], dict | QuantumParameters] | None."
        )

    if temporary_parameters:
        new_quantum_elements = []
        for q in qpu.quantum_elements:
            if q.uid in temporary_parameters:
                temp_param = temporary_parameters[q.uid]
                if isinstance(temp_param, QuantumParameters):
                    temp_param = attrs.asdict(temp_param)
                new_q = q.replace(**temp_param)
                new_quantum_elements.append(new_q)
            else:
                new_quantum_elements.append(q.copy())
        new_topology_edges = []
        for e in qpu.topology.edges():
            edge_key = (e.tag, e.source_node.uid, e.target_node.uid)
            if edge_key in temporary_parameters:
                temp_param = temporary_parameters[edge_key]
                if isinstance(temp_param, QuantumParameters):
                    temp_param = attrs.asdict(temp_param)
                new_topology_edges.append(
                    (edge_key, e.parameters.replace(**temp_param), e.quantum_element),
                )
            else:
                new_topology_edges.append((edge_key, e.parameters, e.quantum_element))
    else:
        new_quantum_elements = [q.copy() for q in qpu.quantum_elements]
        new_topology_edges = [
            (
                (e.tag, e.source_node.uid, e.target_node.uid),
                e.parameters,
                e.quantum_element,
            )
            for e in qpu.topology.edges()
        ]
    new_quantum_operations = qpu.quantum_operations.copy()

    new_qpu = QPU(
        quantum_elements=new_quantum_elements, quantum_operations=new_quantum_operations
    )
    for edge_key, parameters, quantum_element in new_topology_edges:
        new_parameters = parameters.copy() if parameters else None
        new_quantum_element = quantum_element.copy() if quantum_element else None
        new_qpu.topology.add_edge(
            *edge_key,
            parameters=new_parameters,
            quantum_element=new_quantum_element,
        )
    return new_qpu


@workflow.task
def temporary_quantum_elements_from_qpu(
    qpu: QPU,
    quantum_elements: list[str] | str | None = None,
) -> QuantumElements:
    """Return temporarily-modified quantum elements from the QPU.

    !!! version-removed "Removed in version 26.7.0"
        The `qubits` argument of type `QuantumElements` has been removed.
        Please pass `qubits` of type `list[str] | str | None` instead, i.e., the quantum
        element UIDs instead of the quantum element instances.

    Args:
        qpu: The temporarily-modified QPU.
        quantum_elements: The quantum elements to return, passed by UID.

    Returns:
        The temporarily-modified quantum elements.

    Raises:
        TypeError: If the quantum elements have invalid type.
    """
    if quantum_elements is None:
        return qpu.quantum_elements
    if isinstance(quantum_elements, str):
        return qpu[quantum_elements]
    if isinstance(quantum_elements, list) and all(
        isinstance(q, str) for q in quantum_elements
    ):
        return [qpu[q] for q in quantum_elements]
    raise TypeError(
        f"The quantum elements have invalid type: {quantum_elements}. "
        f"Expected type: list[str] | str | None."
    )
