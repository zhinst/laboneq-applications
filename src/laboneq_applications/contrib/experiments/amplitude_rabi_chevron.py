# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the amplitude rabi chevron experiment.

In this experiment, we sweep the frequency and the amplitude
of the drive pulse.

The amplitude_rabi_chevron experiment has the following pulse sequence:

    qb --- [ prep transition ] --- [ qop (swept frequency, swept amplitude)]
       --- [ measure ]

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.simple import Experiment, SweepParameter, dsl
from laboneq.workflow import if_, task, workflow
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.contrib.analysis.amplitude_rabi_chevron import (
    analysis_workflow,
)
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import Qubits, QubitSweepPoints


@workflow(name="amplitude_rabi_chevron")
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
    - [analysis_workflow]() (optional)

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
            The options for building the workflow.
            In addition to options from [WorkflowOptions], the following
            custom options are supported:
                - create_experiment: The options for creating the experiment.

    Returns:
        result:
            The result of the workflow.

    Example:
        ```python
        options = experiment_workflow.options()
        options.count(10)
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
    with if_(options.do_analysis):
        analysis_workflow(result, qubits, frequencies, amplitudes)


@task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: Qubits,
    frequencies: QubitSweepPoints,
    amplitudes: QubitSweepPoints,
    options: TuneupExperimentOptions | None = None,
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
            See [TuneupExperimentOptions] and [BaseExperimentOptions] for
            accepted options.
            Overwrites the options from [TuneupExperimentOptions] and
            [BaseExperimentOptions].

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
        options = TuneupExperimentOptions()
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
    opts = TuneupExperimentOptions() if options is None else options

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
                    qop.set_frequency(q, frequency, transition=opts.transition)
                    qop.x180(q, amplitude=amplitude)
                    qop.measure(q, dsl.handles.result_handle(q.uid))
                    qop.passive_reset(q)

                    if opts.use_cal_traces:
                        with dsl.section(
                            name=f"cal_{q.uid}",
                        ):
                            for state in opts.cal_states:
                                qop.prepare_state(q, state)
                                qop.measure(
                                    q,
                                    dsl.handles.calibration_trace_handle(q.uid, state),
                                )
                                qop.passive_reset(q)
