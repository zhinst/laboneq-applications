"""Tasks for generating experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from laboneq.simple import Experiment, SweepParameter

from laboneq_applications.core.build_experiment import qubit_experiment
from laboneq_applications.core.quantum_operations import QuantumOperations, dsl
from laboneq_applications.tasks import compile_experiment, run_experiment
from laboneq_applications.workflow import task, workflow

if TYPE_CHECKING:
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.dsl.simple import Session

    from laboneq_applications.workflow import Workflow, WorkflowResult


def _is_list_of_numbers(obj: object) -> bool:
    return (
        bool(obj)
        and isinstance(obj, list)
        and all(isinstance(item, (float, int)) for item in obj)
    )


@workflow
def amplitude_rabi_workflow(  # noqa: PLR0913
    session: Session,
    qop: QuantumOperations,
    qubits: QuantumElement | Sequence[QuantumElement],
    amplitudes: Sequence[float] | Sequence[Sequence[float]],
    count: int = 10,
    transition: str = "ge",
) -> tuple[Workflow, WorkflowResult]:
    """Amplitude Rabi workflow builder.

    The amplitude Rabi workflow consists of the following steps:

    - [amplitude_rabi]()
    - [compile_experiment]()
    - [run_experiment]()

    Arguments:
        session:
            The connected session to use for running the experiment.
        qop:
            The quantum operations to use when building the experiment.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        amplitudes:
            The amplitudes to sweep over for each qubit. If `qubits` is a
            single qubit, `amplitudes` must be a list of numbers. Otherwise
            it must be a list of lists of numbers.
        count:
            Number of experiment shots.
        transition:
            Transition to perform the experiment on. May be any transition supported
            by the quantum operations. Default: `"ge"` (i.e. ground to first
            excited state).

    Returns:
        workflow:
            The workflow created.
        result:
            The result of the workflow.

    Example:
        The workflow builder may be called directly to create and run the workflow:

        ```python
        wf, result = amplitude_rabi_workflow(
            session=session,
            qop=qop,
            qubits=q0,
            amplitudes=[0.1, 0.2, 0.3],
            count=10,
            transition="ge",
        )
        ```

        Or its `.create` method may be used to build a workflow instance and run it
        separately:

        ```python
        wf = amplitude_rabi_workflow.create()
        result = wf.run(
            session=session,
            qop=qop,
            qubits=q0,
            amplitudes=[0.1, 0.2, 0.3],
            count=10,
            transition="ge",
        )
        ```
    """
    exp = amplitude_rabi(
        qop,
        qubits,
        amplitudes=amplitudes,
        count=count,
        transition=transition,
    )
    compiled_exp = compile_experiment(session, exp)
    _result = run_experiment(session, compiled_exp)


@task
@qubit_experiment
def amplitude_rabi(
    qop: QuantumOperations,
    qubits: QuantumElement | Sequence[QuantumElement],
    amplitudes: Sequence[float] | Sequence[Sequence[float]],
    count: int = 10,
    transition: str = "ge",
) -> Experiment:
    """Amplitude Rabi experiment.

    Arguments:
        qop:
            The quantum operations to use when building the experiment.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        amplitudes:
            The amplitudes to sweep over for each qubit. If `qubits` is a
            single qubit, `amplitudes` must be a list of numbers. Otherwise
            it must be a list of lists of numbers.
        count:
            Number of experiment shots.
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
            qop=TunableTransmonOperations(),
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            amplitudes=[[0.1, 0.5, 1], [0.1, 0.5, 1]],
            count=10,
            transition="ge",
        )
        ```
    """
    # TODO: Check that qubits are of the same type = QuantumElement.
    # The implementation can be used for other experiment tasks.
    if not isinstance(qubits, Sequence):
        if not _is_list_of_numbers(amplitudes):
            raise ValueError(
                "If a single qubit is passed, the qubit_amplitudes must be a list"
                "of numbers.",
            )
        qubits = [qubits]
        amplitudes = [amplitudes]
    else:
        if len(qubits) != len(amplitudes):
            raise ValueError("Length of qubits and qubit_amplitudes must be the same.")
        if not all(_is_list_of_numbers(amps) for amps in amplitudes):
            raise ValueError(
                "All elements of qubit_amplitudes must be lists of numbers.",
            )
    with dsl.acquire_loop_rt(count=count):
        for q, q_amplitudes in zip(qubits, amplitudes):
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
