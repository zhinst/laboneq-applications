# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the task for updating setup parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
from laboneq.dsl.quantum import QPU, QuantumElement, QuantumParameters
from laboneq.workflow import (
    comment,
    task,
)
from typing_extensions import deprecated

if TYPE_CHECKING:
    import uncertainties as unc

    from laboneq_applications.typing import QuantumElements


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


@task
def update_qubits(
    qpu: QPU,
    qubit_parameters: dict[
        str,
        dict[str, dict[str, int | float | unc.core.Variable | None]],
    ],
) -> None:
    """Updates the parameters of the qubits in the qpu.

    Args:
        qpu: the qpu containing the qubits to be updated.
        qubit_parameters: qubit parameters and the new values to be updated.
            This  dictionary has the following form:
            ```python
            {
                q.uid: {
                    qb_param_name: qb_param_value
                    }
            }
            ```
    """
    qubit_parameters_numeric = {}
    for qid, params_dict in qubit_parameters.items():
        if len(params_dict) == 0:
            comment(
                f"{qid} could not be updated because its "
                f"parameters could not be extracted."
            )
        params_dict_numeric = {
            k: v.nominal_value if hasattr(v, "nominal_value") else v
            for k, v in params_dict.items()
        }
        qubit_parameters_numeric[qid] = params_dict_numeric
    qpu.update_quantum_elements(qubit_parameters_numeric)


@task
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
            {
                key: {
                    "param": param_value
                }
            }
            ```
            or
            ```python
            {
                key: QuantumParameters
            }
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
                new_quantum_elements.append(q)
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
        new_qpu.topology.add_edge(
            *edge_key,
            parameters=parameters,
            quantum_element=quantum_element,
        )
    return new_qpu


@task
def temporary_quantum_elements_from_qpu(
    qpu: QPU,
    quantum_elements: QuantumElements | list[str] | str | None = None,
) -> QuantumElements:
    """Return temporarily-modified quantum elements from the QPU.

    Args:
        qpu: The temporarily-modified QPU.
        quantum_elements: The quantum elements to return. Either the QuantumElement
            objects or the UIDs may be provided.

    Returns:
        The temporarily-modified quantum elements.

    Raises:
        TypeError: If the quantum elements have invalid type.
    """
    if quantum_elements is None:
        return qpu.quantum_elements

    if isinstance(quantum_elements, QuantumElement):
        return qpu[quantum_elements.uid]

    if isinstance(quantum_elements, str):
        return qpu[quantum_elements]

    if isinstance(quantum_elements, list):
        quantum_elements_uids = []
        for q in quantum_elements:
            if isinstance(q, QuantumElement):
                quantum_elements_uids.append(q.uid)
            elif isinstance(q, str):
                quantum_elements_uids.append(q)
            else:
                raise TypeError(
                    f"The quantum elements list items have invalid type: {type(q)}. "
                    f"Expected type: QuantumElement | str."
                )

        return [qpu[q] for q in quantum_elements_uids]

    raise TypeError(
        f"The quantum elements have invalid type: {type(quantum_elements)}. "
        f"Expected type: QuantumElements | list[str] | str | None."
    )


@task
@deprecated(
    "The temporary_modify method is deprecated. Use `temporary_qpu` instead. "
    "Instead of passing temporary qubits to an experiment, we now pass a temporary "
    "QPU.",
    category=FutureWarning,
)
def temporary_modify(
    qubits: QuantumElements,
    temporary_parameters: dict[str, dict | QuantumParameters] | None = None,
) -> QuantumElements:
    """Modify the quantum elements temporarily with the given parameters.

    Args:
        qubits: the quantum elements to be temporarily modified.
        temporary_parameters: the parameters to be temporarily modified.
            If None, the quantum elements are returned as is.
            The dictionary has the following form:
            ```python
            {
                quantum_element_uid: {
                    "quantum_element_uid": param_value
                }
            }
            ```
            or
            ```python
            {
                "quantum_element_uid": QuantumParameters
            }
            ```

    Returns:
        The list of quantum elements with the temporary parameters applied, including
        the original quantum elements that were not modified.
        If a single quantum element is passed, returns the modified quantum element.

    Raises:
        TypeError: If the temporary parameters have invalid type.

    !!! version-changed "Deprecated in version 2.54.0."
            The method `temporary_modify` was deprecated and replaced with the method
            `temporary_qpu`. Instead of passing temporary qubits to an experiment, we
            now pass a temporary QPU.
    """
    if not temporary_parameters:
        return qubits

    if not _valid_temporary_parameters(temporary_parameters):
        raise TypeError(
            f"The temporary parameters have invalid type: {type(temporary_parameters)}."
            f" Expected type: dict[str, dict | QuantumParameters] | None."
        )

    _single_qubit = False
    if not isinstance(qubits, list):
        qubits = [qubits]
        _single_qubit = True

    new_qubits = []
    for q in qubits:
        if q.uid in temporary_parameters:
            temp_param = temporary_parameters[q.uid]
            if isinstance(temp_param, QuantumParameters):
                temp_param = attrs.asdict(temp_param)
            new_q = q.replace(**temp_param)
            new_qubits.append(new_q)
        else:
            new_qubits.append(q)
    if _single_qubit:
        return new_qubits[0]
    return new_qubits
