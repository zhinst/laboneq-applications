# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the task for updating setup parameters."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from laboneq.dsl.quantum import TransmonParameters
from laboneq.workflow import (
    comment,
    task,
)

if TYPE_CHECKING:
    import uncertainties as unc
    from laboneq.dsl.quantum.qpu import QPU

    from laboneq_applications.typing import Qubits


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
    qpu.update_qubits(qubit_parameters_numeric)


@task
def temporary_modify(
    qubits: Qubits,
    temporary_parameters: dict[str, dict | TransmonParameters] | None = None,
) -> Qubits:
    """Modify the qubits temporarily with the given parameters.

    Args:
        qubits: the qubits to be temporarily modified.
        temporary_parameters: the parameters to be temporarily modified.
            If None, the qubits are returned as is.
            The dictionary has the following form:
            ```python
            {
                qubit_uid: {
                    "qubit_uid": param_value
                }
            }
            ```
            or
            ```python
            {
                "qubit_uid": TransmonParameters
            }
            ```

    Returns:
        The list of qubits with the temporary parameters applied, including
        the original qubits that were not modified.
        If a single qubit is passed, returns the modified qubit.
    """
    if not temporary_parameters:
        return qubits
    _single_qubit = False
    if not isinstance(qubits, list):
        qubits = [qubits]
        _single_qubit = True

    new_qubits = []
    for q in qubits:
        if q.uid in temporary_parameters:
            temp_param = temporary_parameters[q.uid]
            if isinstance(temp_param, TransmonParameters):
                temp_param = asdict(temp_param)
            new_q = q.replace(temp_param)
            new_qubits.append(new_q)
        else:
            new_qubits.append(q)
    if _single_qubit:
        return new_qubits[0]
    return new_qubits
