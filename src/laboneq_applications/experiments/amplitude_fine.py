# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the amplitude_fine experiment.

In this experiment, we apply the same quantum operation a variable number of times. If
each quantum operation has a small rotation error theta, then the sequence of multiple
quantum operations will accumulate the error reps*theta, where reps is the number of
time the quantum operation is repeated. From the experiment result we can obtain the
correction value for the amplitude of imperfect drive pulses.

The amplitude_fine experiment has the following pulse sequence

    qb --- [ prep transition ] --- [ quantum_operation ]**reps --- [ measure ]

where reps is varied.

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from laboneq import workflow
from laboneq.simple import (
    AveragingMode,
    Experiment,
    SectionAlignment,
    SweepParameter,
    dsl,
)
from laboneq.workflow.tasks import compile_experiment, run_experiment

from laboneq_applications.analysis.amplitude_fine import analysis_workflow
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks import (
    temporary_modify,
    update_qubits,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.workflow(name="amplitude_fine")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    amplification_qop: str,
    target_angle: float,
    phase_offset: float,
    repetitions: QubitSweepPoints[int],
    parameter_to_update: str | None = None,
    temporary_parameters: dict[str, dict | QuantumParameters] | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The amplitude fine experiment workflow.

    The workflow consists of the following steps:

    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()
    - [analysis_workflow]()
    - [update_qubits]()

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        amplification_qop:
            str to identify the quantum operation to repeat to produce error
            amplification. The quantum operation must exist in
            qop.keys().
        target_angle:
            target angle the specified quantum operation shuould rotate.
            The target_angle is used as initial guess for fitting.
        phase_offset:
            initial guess for phase_offset of fit.
        repetitions:
            The sweep values corresponding to the number of times to repeat the
            amplification_qop for each qubit. If `qubits` is a single qubit,
            `repetitions` must be a list of integers. Otherwise, it must be a list of
            lists of integers.
        parameter_to_update:
            str that defines the qubit parameter to be updated.
        temporary_parameters:
            The temporary parameters to update the qubits with.
        options:
            The options for building the workflow.
            In addition to options from [WorkflowOptions], the following
            custom options are supported:
                - create_experiment: The options for creating the experiment.

    Returns:
        WorkflowBuilder:
            The builder of the experiment workflow.

    Example:
        ```python
        options = experiment_workflow.options()
        options.count(10)
        options.transition("ge")
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=temp_qubits,
            amplification_qop='x180',
            repetitions=[
                [1,2,3,4],
                [1,2,3,4],
            ],
            options=options,
        ).run()
        ```
    """
    qubits = temporary_modify(qubits, temporary_parameters)
    exp = create_experiment(
        qpu,
        qubits,
        amplification_qop,
        repetitions=repetitions,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_workflow(
            result,
            qubits,
            amplification_qop,
            repetitions,
            target_angle,
            phase_offset,
            parameter_to_update,
        )
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    amplification_qop: str,
    repetitions: QubitSweepPoints,
    options: TuneupExperimentOptions | None = None,
) -> Experiment:
    """Creates an Amplitude Rabi experiment Workflow.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        amplification_qop:
            String to define the quantum operation that should be applied multiple
            times to produce error amplification. The quantum operation must exist in
            qop.keys().
        repetitions:
            Number of time to repeat the quantum operation used to amplify the rotation
            error. If `qubits` is a single qubit, `repetitions` must be a list of
            integers. Otherwise, it must be a list of lists of integers.
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
            If the qubits and qubit_amplitudes are not of the same length.

        ValueError:
            If qubit_amplitudes is not a list of numbers when a single qubit is passed.

        ValueError:
            If qubit_amplitudes is not a list of lists of numbers.

        ValueError:
            If the experiment uses calibration traces and the averaging mode is
            sequential.

    Example:
        ```python
        options = TuneupExperimentOptions()
        options.count(10)
        options.cal_traces(True)
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        create_experiment(
            qpu=qpu,
            qubits=temp_qubits,
            amplification_qop="x180",
            repetitions=[
                [1,2,3,4],
                [1,2,3,4],
            ],
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = TuneupExperimentOptions() if options is None else options
    qubits, repetitions = validate_and_convert_qubits_sweeps(qubits, repetitions)
    if (
        opts.use_cal_traces
        and AveragingMode(opts.averaging_mode) == AveragingMode.SEQUENTIAL
    ):
        raise ValueError(
            "'AveragingMode.SEQUENTIAL' (or {AveragingMode.SEQUENTIAL}) cannot be used "
            "with calibration traces because the calibration traces are added "
            "outside the sweep."
        )

    reps_sweep_pars = [
        SweepParameter(f"repetitions_{q.uid}", q_reps, axis_name=f"{q.uid}")
        for q, q_reps in zip(qubits, repetitions)
    ]
    # We will fix the length of the measure section to the longest section among
    # the qubits to allow the qubits to have different readout and/or
    # integration lengths.
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
        with dsl.sweep(
            name="amplitude_fine_sweep",
            parameter=reps_sweep_pars,
        ):
            if opts.active_reset:
                qop.active_reset(
                    qubits,
                    active_reset_states=opts.active_reset_states,
                    number_resets=opts.active_reset_repetitions,
                    measure_section_length=max_measure_section_length,
                )
            with dsl.section(name="main", alignment=SectionAlignment.RIGHT):
                with dsl.section(name="main_drive", alignment=SectionAlignment.RIGHT):
                    for qbidx, q in enumerate(qubits):
                        qop.prepare_state.omit_section(q, opts.transition[0])
                        sec = qop.x90(q, transition=opts.transition)
                        sec.alignment = SectionAlignment.RIGHT
                        with dsl.section(
                            name=f"match_{q.uid}", alignment=SectionAlignment.RIGHT
                        ):
                            with dsl.match(
                                sweep_parameter=reps_sweep_pars[qbidx],
                            ):
                                for _i, num in enumerate(repetitions[qbidx]):
                                    with dsl.case(num):
                                        for _j in range(num):
                                            qop[amplification_qop].omit_section(
                                                q, transition=opts.transition
                                            )
                with dsl.section(name="main_measure", alignment=SectionAlignment.LEFT):
                    for q in qubits:
                        sec = qop.measure(q, dsl.handles.result_handle(q.uid))
                        # Fix the length of the measure section
                        sec.length = max_measure_section_length
                        qop.passive_reset(q)
        if opts.use_cal_traces:
            qop.calibration_traces.omit_section(
                qubits=qubits,
                states=opts.cal_states,
                active_reset=opts.active_reset,
                active_reset_states=opts.active_reset_states,
                active_reset_repetitions=opts.active_reset_repetitions,
                measure_section_length=max_measure_section_length,
            )


@workflow.workflow
def experiment_workflow_x180(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    repetitions: QubitSweepPoints[int],
    temporary_parameters: dict[str, dict | QuantumParameters] | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The amplitude fine experiment workflow for a x180 gate.

    This workflow is the same as experiment_workflow above but with the following
    input parameters fixed:
        amplification_qop = "x180"
        target_angle = np.pi
        phase_offset = -np.pi / 2
        parameter_to_update = "drive_amplitude_pi"

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        repetitions:
            The sweep values corresponding to the number of times to repeat the
            amplification_qop for each qubit. If `qubits` is a single qubit,
            `repetitions` must be a list of integers. Otherwise it must be a list of
            lists of integers.
        temporary_parameters:
            The temporary parameters to update the qubits with.
        options:
            The options for building the workflow.
            In addition to options from [WorkflowOptions], the following
            custom options are supported:
                - create_experiment: The options for creating the experiment.

    Returns:
        WorkflowBuilder:
            The builder of the experiment workflow.

    Example:
        ```python
        options = TuneUpExperimentWorkflowOptions()
        options.create_experiment.count = 10
        options.create_experiment.transition = "ge"
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=temp_qubits,
            repetitions=[
                [1,2,3,4],
                [1,2,3,4],
            ],
            options=options,
        ).run()
        ```
    """
    amplification_qop = "x180"
    target_angle = np.pi
    phase_offset = -np.pi / 2
    parameter_to_update = "drive_amplitude_pi"

    qubits = temporary_modify(qubits, temporary_parameters)
    exp = create_experiment(
        qpu,
        qubits,
        amplification_qop,
        repetitions=repetitions,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(
            result,
            qubits,
            amplification_qop,
            repetitions,
            target_angle,
            phase_offset,
            parameter_to_update,
        )
        qubit_parameters = analysis_results.output
        with workflow.if_(options.update):
            update_qubits(qpu, qubit_parameters["new_parameter_values"])
    workflow.return_(result)


@workflow.workflow
def experiment_workflow_x90(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    repetitions: QubitSweepPoints[int],
    temporary_parameters: dict[str, dict | QuantumParameters] | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The amplitude fine experiment workflow for a x90 gate.

    This workflow is the same as experiment_workflow above but with the following
    input parameters fixed:
        amplification_qop = "x90"
        target_angle = np.pi / 2
        phase_offset = -np.pi / 2
        parameter_to_update = "drive_amplitude_pi2"

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        repetitions:
            The sweep values corresponding to the number of times to repeat the
            amplification_qop for each qubit. If `qubits` is a single qubit,
            `repetitions` must be a list of integers. Otherwise it must be a list of
            lists of integers.
        temporary_parameters:
            The temporary parameters to update the qubits with.
        options:
            The options for building the workflow.
            In addition to options from [WorkflowOptions], the following
            custom options are supported:
                - create_experiment: The options for creating the experiment.

    Returns:
        WorkflowBuilder:
            The builder of the experiment workflow.

    Example:
        ```python
        options = TuneUpExperimentWorkflowOptions()
        options.create_experiment.count = 10
        options.create_experiment.transition = "ge"
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=temp_qubits,
            repetitions=[
                [1,2,3,4],
                [1,2,3,4],
            ],
            options=options,
        ).run()
        ```
    """
    amplification_qop = "x90"
    target_angle = np.pi / 2
    phase_offset = -np.pi / 2
    parameter_to_update = "drive_amplitude_pi2"

    qubits = temporary_modify(qubits, temporary_parameters)
    exp = create_experiment(
        qpu,
        qubits,
        amplification_qop,
        repetitions=repetitions,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(
            result,
            qubits,
            amplification_qop,
            repetitions,
            target_angle,
            phase_offset,
            parameter_to_update,
        )
        qubit_parameters = analysis_results.output
        with workflow.if_(options.update):
            update_qubits(qpu, qubit_parameters["new_parameter_values"])
    workflow.return_(result)
