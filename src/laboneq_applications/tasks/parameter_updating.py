"""This module defines the task for updating setup parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq_applications.workflow import (
    comment,
    task,
)

if TYPE_CHECKING:
    import uncertainties as unc

    from laboneq_applications.qpu_types import QPU


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
        qpu: the qpu containing the qubits to be updated
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
