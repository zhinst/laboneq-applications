"""This module defines the amplitude_fine experiment.

In this experiment, we apply the same quantum operation a variable number of times. If
each qop has a small error theta, then the sequence of multiple qop's will accumulate
the error reps*theta, where reps is the number of repetitions. From the experiment
result we can obtain the correction value for the amplitude of imperfect pulses.

The amplitude_fine experiment has the following pulse sequence

    qb --- [ prep transition ] --- [ qop ]**reps --- [ measure ]

where reps is varied.

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from laboneq.simple import Experiment, SweepParameter

from laboneq_applications import dsl, workflow
from laboneq_applications.analysis.amplitude_fine import analysis_workflow
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks import compile_experiment, run_experiment, update_qubits

if TYPE_CHECKING:
    from laboneq.dsl.session import Session

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import Qubits, QubitSweepPoints


@workflow.workflow
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    amplification_qop: str,
    target_angle: float,
    phase_offset: float,
    repetitions: QubitSweepPoints[int],
    parameter_to_update: str | None = None,
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
            amplification. The quantum operation must exist in qpu.qop.keys().
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
            qop=TunableTransmonOperations(),
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
    qubits: Qubits,
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
            qpu.qop.keys().
        repetitions:
            Number of qop repetitions to sweep over. If `qubits` is a
            single qubit, `repetitions` must be a list of integers. Otherwise,
            it must be a list of lists of integers.
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

    Example:
        ```python
        options = TuneupExperimentOptions()
        options.count(10)
        options.cal_traces(True)
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            qop=TunableTransmonOperations(),
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
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for q, q_n in zip(qubits, repetitions):
            with dsl.sweep(
                name=f"loop_{q.uid}",
                parameter=SweepParameter(f"n_{q.uid}", q_n),
            ) as index:
                qpu.qop.prepare_state(q, opts.transition[0])
                qpu.qop.x90(q, transition=opts.transition)

                with dsl.match(
                    sweep_parameter=index,
                ):
                    for _i, num in enumerate(q_n):
                        with dsl.case(num):
                            for _j in range(num):
                                qpu.qop[amplification_qop](
                                    q, transition=opts.transition
                                )

                qpu.qop.measure(q, dsl.handles.result_handle(q.uid))
                qpu.qop.passive_reset(q)
            if opts.use_cal_traces:
                with dsl.section(
                    name=f"cal_{q.uid}",
                ):
                    for state in opts.cal_states:
                        qpu.qop.prepare_state(q, state)
                        qpu.qop.measure(
                            q,
                            dsl.handles.calibration_trace_handle(q.uid, state),
                        )
                        qpu.qop.passive_reset(q)


@workflow.workflow
def experiment_workflow_x180(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    repetitions: QubitSweepPoints[int],
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
            qop=TunableTransmonOperations(),
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

    exp = create_experiment(
        qpu,
        qubits,
        amplification_qop,
        repetitions=repetitions,
    )
    compiled_exp = compile_experiment(session, exp)
    _result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(
            _result,
            qubits,
            amplification_qop,
            repetitions,
            target_angle,
            phase_offset,
            parameter_to_update,
        )
        qubit_parameters = analysis_results.tasks["extract_qubit_parameters"].output
        with workflow.if_(options.update):
            update_qubits(qpu, qubit_parameters["new_parameter_values"])


@workflow.workflow
def experiment_workflow_x90(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    repetitions: QubitSweepPoints[int],
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
            qop=TunableTransmonOperations(),
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

    exp = create_experiment(
        qpu,
        qubits,
        amplification_qop,
        repetitions=repetitions,
    )
    compiled_exp = compile_experiment(session, exp)
    _result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(
            _result,
            qubits,
            amplification_qop,
            repetitions,
            target_angle,
            phase_offset,
            parameter_to_update,
        )
        qubit_parameters = analysis_results.tasks["extract_qubit_parameters"].output
        with workflow.if_(options.update):
            update_qubits(qpu, qubit_parameters["new_parameter_values"])
