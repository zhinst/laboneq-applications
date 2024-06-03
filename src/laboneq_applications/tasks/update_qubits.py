"""This module contains the task to update the qubits in the workflow."""

from collections.abc import Sequence

from laboneq.dsl.result.results import Results

from laboneq_applications.qpu_types.tunable_transmon import TunableTransmonQubit
from laboneq_applications.workflow.task import task


@task
def update_qubits(
    qubit_parameter_pairs: Sequence[tuple[TunableTransmonQubit, dict]],
) -> Results:
    """Update qubits with the given parameters.

    Args:
        qubit_parameter_pairs: A sequence of tuples, each consists of a qubit and
            a parameter dictionary to update.

    Raises:
        ValueError: If any of the parameters are invalid.

    Example:
        ```python
        from laboneq_applications.qpu_types.tunable_transmon import TunableTransmonQubit
        from laboneq_applications.tasks.update_qubits import update_qubits
        from laboneq_applications.workflow.engine import Workflow
        q0 = TunableTransmonQubit()
        q1 = TunableTransmonQubit()
        update_qubits(
            [
                (q0, {"drive_parameters_ge": {"amplitude": 1.0}}),
                (q1, {"drive_parameters_ef": {"amplitude": 1.0}}),
            ],
        )
        ```
    """
    invalid_params = []
    for qubit, parameter_updates in qubit_parameter_pairs:
        invalid_param = qubit._get_invalid_params_to_update(parameter_updates)
        if invalid_param:
            invalid_params.append((qubit.uid, invalid_param))
    if invalid_params:
        raise ValueError(
            f"Qubits were not updated due to the following parameters "
            f"being invalid: {invalid_params}",
        )

    for qubit, parameter_updates in qubit_parameter_pairs:
        qubit.update(parameter_updates)
