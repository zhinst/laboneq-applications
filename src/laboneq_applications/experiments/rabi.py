"""Tasks for generating experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from laboneq.simple import Experiment, SweepParameter

from laboneq_applications.core.build_experiment import qubit_experiment
from laboneq_applications.core.quantum_operations import QuantumOperations, dsl
from laboneq_applications.workflow.task import task

if TYPE_CHECKING:
    from laboneq.dsl.quantum.quantum_element import QuantumElement


def _is_list_of_numbers(obj: object) -> bool:
    return (
        bool(obj)
        and isinstance(obj, list)
        and all(isinstance(item, (float, int)) for item in obj)
    )


@task
@qubit_experiment
def amplitude_rabi(
    qop: QuantumOperations,
    qubits: QuantumElement | Sequence[QuantumElement],
    qubit_amplitudes: Sequence[float] | Sequence[Sequence[float]],
    count: int = 10,
    transition: str = "ge",
) -> Experiment:
    """Amplitude Rabi experiment.

    Arguments:
        qop:
            Quantum operations to use.
        qubits:
            Either a single qubit or a list of qubits to perform the experiment on.
        qubit_amplitudes:
            If a single qubit is passed to `qubits` then a list of amplitudes to sweep.
            Otherwise a list of lists of amplitudes to sweep for each qubit.
        count:
            Number of real-time experiment data acquisitions (measurement shots).
        transition:
            Transition to perform the experiment on. May be any transition supported
            by the quantum operations. Default: `"ge"` (i.e. ground to first
            excited state).

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Raises:
        ValueError:
            If the qubits and qubit_amplitudes are not of the same length.

        ValueError:
            If qubit_amplitudes is not a list of numbers when a single qubit is passed.

        ValueError:
            If qubit_amplitudes is not a list of lists of numbers.

    Example:
    ```python
    amplitude_rabi(
        qop = TunableTransmonOperations(),
        qubits = [TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
        amplitudes = [[0.1, 0.5, 1], [0.1, 0.5, 1]],
        count = 10,
        transition = 'ge'
    )
    ```
    """
    # TODO: Check that qubits are of the same type = QuantumElement.
    # The implementation can be used for other experiment tasks.
    if not isinstance(qubits, Sequence):
        if not _is_list_of_numbers(qubit_amplitudes):
            raise ValueError(
                "If a single qubit is passed, the qubit_amplitudes must be a list"
                "of numbers.",
            )
        qubits = [qubits]
        qubit_amplitudes = [qubit_amplitudes]
    else:
        if len(qubits) != len(qubit_amplitudes):
            raise ValueError("Length of qubits and qubit_amplitudes must be the same.")
        if not all(_is_list_of_numbers(amps) for amps in qubit_amplitudes):
            raise ValueError(
                "All elements of qubit_amplitudes must be lists of numbers.",
            )
    with dsl.acquire_loop_rt(count=count):
        for q, q_amplitudes in zip(qubits, qubit_amplitudes):
            with dsl.sweep(
                name=f"amps_{q.uid}",
                parameter=SweepParameter(f"amplitude_{q.uid}", q_amplitudes),
            ) as amplitude:
                qop.prep(q, transition[0])
                qop.x180(q, amplitude=amplitude, transition=transition)
                qop.measure(q, f"result_{q.uid}")

            with dsl.section(
                name=f"cal_{q.uid}",
            ):
                for state in transition:
                    qop.prep(q, state)
                    qop.measure(q, f"cal_state_{state}_{q.uid}")
