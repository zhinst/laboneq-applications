# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

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

from laboneq import workflow
from laboneq.simple import Experiment, SweepParameter, dsl
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.analysis.spectroscopy_two_dimensional_plotting import (
    analysis_workflow,
)
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.experiments.options import (
    QubitSpectroscopyExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.workflow(name="qubit_spectroscopy_amplitude")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    frequencies: QubitSweepPoints,
    amplitudes: QubitSweepPoints,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Qubit Spectroscopy Workflow.

    The workflow consists of the following steps:

    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()
    - [analysis_workflow]()

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
        temporary_parameters:
            The temporary parameters with which to update the quantum elements and
            topology edges. For quantum elements, the dictionary key is the quantum
            element UID. For topology edges, the dictionary key is the edge tuple
            `(tag, source node UID, target node UID)`.
        options:
            The options for building the workflow as an instance of
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
            quantum_elements=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_quantum_elements()
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
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    exp = create_experiment(
        temp_qpu,
        qubits,
        frequencies=frequencies,
        amplitudes=amplitudes,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_workflow(
            result=result,
            qubits=qubits,
            sweep_points_1d=frequencies,
            sweep_points_2d=amplitudes,
            label_sweep_points_1d="Qubit Frequency,\n$f_{\\mathrm{QB}}$ (GHz)",
            label_sweep_points_2d="Spectroscopy-Drive Amp.,\n$A$ (a.u.)",
        )
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
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
            quantum_elements=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_quantum_elements()
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

    max_measure_section_length = qpu.measure_section_length(qubits)
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
                    qop.qubit_spectroscopy_drive(q, amplitude=amplitude)
                    sec = qop.measure(q, dsl.handles.result_handle(q.uid))
                    # we fix the length of the measure section to the longest section
                    # among the qubits to allow the qubits to have different readout
                    # and/or integration lengths.
                    sec.length = max_measure_section_length
                    qop.passive_reset(q, delay=opts.spectroscopy_reset_delay)
