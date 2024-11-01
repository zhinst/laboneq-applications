"""This module defines the lifetime_measurement experiment.

In this experiment, the qubit is first excited to either its first
or a higher excited state
and then allowed to relax back to the ground state over a variable delay period,
enabling us to measure the qubit's longitudinal
relaxation time, lifetime_measurement, for the respective state.

The lifetime_measurement experiment has the following pulse sequence:

    qb --- [ prep transition ] --- [ x180_transition ] --- [delay] --- [ measure ]

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.simple import AveragingMode, Experiment, SectionAlignment, SweepParameter
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications import dsl
from laboneq_applications.analysis.lifetime_measurement import analysis_workflow
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks.parameter_updating import (
    temporary_modify,
    update_qubits,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import TransmonParameters
    from laboneq.dsl.session import Session

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import Qubits, QubitSweepPoints


@workflow.workflow(name="lifetime_measurement")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    delays: QubitSweepPoints,
    temporary_parameters: dict[str, dict | TransmonParameters] | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The lifetime_measurement experiment Workflow.

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
        delays:
            The delays to sweep over for each qubit. If `qubits` is a
            single qubit, `delays` must be a list of numbers or an array. Otherwise
            it must be a list of lists of numbers or arrays.
        temporary_parameters:
            The temporary parameters to update the qubits with.
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
        options = TuneUpWorkflowOptions()
        options.create_experiment.count = 10
        options.create_experiment.transition = "ge"
        qpu = QPU(
            setup=DeviceSetup("my_device"),
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        result = run(
            session=session,
            qpu=qpu,
            qubits=temp_qubits,
            delays=[[10e-9, 50e-9, 1], [10e-9, 50e-9, 1]],
            options=options,
        )
        ```
    """
    qubits = temporary_modify(qubits, temporary_parameters)
    exp = create_experiment(
        qpu,
        qubits,
        delays=delays,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(result, qubits, delays)
        qubit_parameters = analysis_results.output
        with workflow.if_(options.update):
            update_qubits(qpu, qubit_parameters["new_parameter_values"])
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: Qubits,
    delays: QubitSweepPoints,
    options: TuneupExperimentOptions | None = None,
) -> Experiment:
    """Creates a lifetime_measurement Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        delays:
            The delays to sweep over for each qubit. If `qubits` is a
            single qubit, `amplitudes` must be a list of numbers or an array. Otherwise
            it must be a list of lists of numbers or arrays.
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
            If the qubits and qubit_delays are not of the same length.

        ValueError:
            If qubit_delays is not a list of numbers when a single qubit is passed.

        ValueError:
            If qubit_delays is not a list of lists of numbers.

        ValueError:
            If the experiment uses calibration traces and the averaging mode is
            sequential.

    Example:
        ```python
        options = {
            "count": 10,
            "transition": "ge",
            "averaging_mode": "cyclic",
            "acquisition_type": "integration_trigger",
            "cal_traces": True,
        }
        options = TuneupExperimentOptions(**options)
        setup = DeviceSetup()
        qpu = QPU(
            setup=DeviceSetup("my_device"),
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        create_experiment(
            qpu=qpu,
            qubits=temp_qubits,
            delays=[[10e-9, 50e-9, 1], [10e-9, 50e-9, 1]],
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = TuneupExperimentOptions() if options is None else options
    qubits, delays = dsl.validation.validate_and_convert_qubits_sweeps(qubits, delays)
    if (
        opts.use_cal_traces
        and AveragingMode(opts.averaging_mode) == AveragingMode.SEQUENTIAL
    ):
        raise ValueError(
            "'AveragingMode.SEQUENTIAL' (or {AveragingMode.SEQUENTIAL}) cannot be used "
            "with calibration traces because the calibration traces are added "
            "outside the sweep."
        )

    max_measure_section_length = qpu.measure_section_length(qubits)
    qop = qpu.quantum_operations
    if opts.transition == "ef":
        on_system_grid = True
    elif opts.transition == "ge":
        on_system_grid = False
    else:
        raise ValueError(
            f"Support only ge or ef transitions, not {options.transition!r}"
        )
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for q, q_delays in zip(qubits, delays):
            with dsl.sweep(
                name=f"delays_{q.uid}",
                parameter=SweepParameter(f"delay_{q.uid}", q_delays),
            ) as delay:
                qop.prepare_state(q, opts.transition[0])
                with dsl.section(
                    name=f"t1_{q.uid}",
                    on_system_grid=on_system_grid,
                    alignment=SectionAlignment.RIGHT,
                ):
                    sec_180 = qop.x180(q, transition=opts.transition)
                    qop.delay(q, time=delay)
                    sec_measure = qop.measure(q, dsl.handles.result_handle(q.uid))
                    # we fix the length of the measure section to the longest section
                    # among the qubits to allow the qubits to have different readout
                    # and/or integration lengths.
                    sec_measure.length = max_measure_section_length
                # to remove the gaps between ef_drive and measure pulses
                # introduced by system grid alignment.
                qop.passive_reset(q)
                sec_180.on_system_grid = False
                sec_measure.on_system_grid = False
            if opts.use_cal_traces:
                qop.calibration_traces(q, states=opts.cal_states)
