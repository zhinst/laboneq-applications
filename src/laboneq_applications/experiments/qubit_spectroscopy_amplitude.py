"""This module defines the qubit spectroscopy experiment.

In this experiment, we sweep the frequency and the amplitude
of a qubit drive pulse to characterize the qubit transition frequency and its
amplitude.

The qubit spectroscopy experiment has the following pulse sequence:

    qb --- [ prep transition ] --- [ x180_transition (swept frequency)] --- [ measure ]

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.simple import Experiment, SweepParameter

from laboneq_applications import dsl, workflow
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.experiments.options import (
    QubitSpectroscopyExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks import compile_experiment, run_experiment

if TYPE_CHECKING:
    from laboneq.dsl.session import Session

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import Qubits, QubitSweepPoints


@workflow.workflow(name="qubit_spectroscopy_amplitude")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    frequencies: QubitSweepPoints,
    amplitudes: QubitSweepPoints,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Qubit Spectroscopy Workflow.

    The workflow consists of the following steps:

    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        frequencies:
            The qubit frequencies to sweep over for the qubit drive pulse. If `qubits`
            is a single qubit, `frequencies` must be a list of numbers or an array.
            Otherwise, it must be a list of lists of numbers or arrays.
        amplitudes:
            The amplitudes to sweep over for each qubit drive pulse.  `amplitudes` must
            be a list of numbers or an array. Otherwise it must be a list of lists of
             numbers or arrays.
        options:
            The options for building the workflow as an in stance of
            [QubitSpectroscopyWorkflowOptions]. See the docstring of
            [QubitSpectroscopyWorkflowOptions] for more details.

    Returns:
        result:
            The result of the workflow.

    Example:
        ```python
        options = experiment_workflow.options()
        options.create_experiment.count(10)
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=temp_qubits,
            frequencies = [
                np.linspace(5.8e9, 6.2e9, 101),
                np.linspace(0.8e9, 1.2e9, 101)
            ],
            amplitudes=[[0.1, 0.5, 1], [0.1, 0.5, 1]],
            options=options,
        ).run()
        ```
    """
    exp = create_experiment(
        qpu,
        qubits,
        frequencies=frequencies,
        amplitudes=amplitudes,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: Qubits,
    frequencies: QubitSweepPoints,
    amplitudes: QubitSweepPoints,
    options: QubitSpectroscopyExperimentOptions | None = None,
) -> Experiment:
    """Creates a Qubit Spectroscopy Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        frequencies:
            The qubit frequencies to sweep over for the qubit drive pulse. If `qubits`
            is a single qubit, `frequencies` must be a list of numbers or an array.
            Otherwise, it must be a list of lists of numbers or arrays.
        amplitudes:
            The amplitudes to sweep over for each qubit. It must be a list of numbers
            or arrays or a list of lists of numbers or arrays.
        options:
            The options for building the experiment.
            See [QubitSpectroscopyExperimentOptions] and [BaseExperimentOptions] for
            accepted options.
            Overwrites the options from [BaseExperimentOptions].

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Raises:
        ValueError:
            If the qubits, amplitudes, and frequencies are not of the same length.

        ValueError:
            If amplitudes and frequencies are not a list of numbers when a single
            qubit is passed.

        ValueError:
            If frequencies is not a list of lists of numbers.
            If amplitudes is not None or a list of lists of numbers.

    Example:
        ```python
        options = QubitSpectroscopyExperimentOptions()
        options.count = 10
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        create_experiment(
            qpu=qpu,
            qubits=temp_qubits,
            frequencies = [
                np.linspace(5.8e9, 6.2e9, 101),
                np.linspace(0.8e9, 1.2e9, 101)
            ],
            amplitudes=[[0.1, 0.5, 1], [0.1, 0.5, 1]],
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = QubitSpectroscopyExperimentOptions() if options is None else options

    _, frequencies = validate_and_convert_qubits_sweeps(qubits, frequencies)
    qubits, amplitudes = validate_and_convert_qubits_sweeps(qubits, amplitudes)

    qop = qpu.quantum_operations
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for q, q_frequencies, q_amplitudes in zip(qubits, frequencies, amplitudes):
            with dsl.sweep(
                name=f"amps_{q.uid}",
                parameter=SweepParameter(f"amplitude_{q.uid}", q_amplitudes),
            ) as amplitude:
                with dsl.sweep(
                    name=f"freqs_{q.uid}",
                    parameter=SweepParameter(f"frequency_{q.uid}", q_frequencies),
                ) as frequency:
                    qop.set_frequency(q, frequency)
                    qop.spectroscopy_drive(q, amplitude=amplitude)
                    qop.measure(q, dsl.handles.result_handle(q.uid))
                    qop.passive_reset(q)
