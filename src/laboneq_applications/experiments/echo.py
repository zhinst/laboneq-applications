# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the Hahn echo experiment.

In the Hahn echo experiment, we perform a Ramsey experiment and place one extra
refocusing pulse, typically y180, between the two x90 pulses. Due to this additional
pulse, the quasi-static contributions to dephasing can be “refocused” and by that the
experiment is less sensitive to quasi-static noise.

The pulses are generally chosen to be resonant with the qubit transition for a
Hahn echo, since any frequency detuning would be nominally refocused anyway.

The Hahn echo experiment has the following pulse sequence:

    qb --- [ prep transition ] --- [ x90_transition ] --- [ delay/2 ] ---
    [ refocusing pulse ] --- [ delay/2 ] --- [ x90_transition ] --- [ measure ]

If multiple qubits are passed to the experiment workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from laboneq import workflow
from laboneq.simple import (
    AveragingMode,
    Experiment,
    SectionAlignment,
    SweepParameter,
    dsl,
)
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.analysis.echo import analysis_workflow
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks import evaluate_parameter_and_fit_r2_thresholds
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
    update_qpu,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session
    from laboneq.workflow import WorkflowResult

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.task_options(base_class=TuneupExperimentOptions)
class EchoExperimentOptions:
    """Options for the Hahn echo experiment.

    Additional attributes:
        refocus_pulse:
            String to define the quantum operation in-between the x90 pulses.
            Default: "y180".
    """

    refocus_qop: str = workflow.option_field(
        "y180",
        description="String to define the quantum operation in-between the x90 pulses",
    )


@workflow.workflow(name="echo")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: list[str] | str,
    *,
    delays: QubitSweepPoints,
    evaluation_parameters: dict[str, Any] | None = None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Hahn echo experiment workflow.

    The workflow consists of the following steps:

    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()
    - [analysis_workflow]()
    - [evaluate_experiment]()
    - [update_qpu]()

    !!! version-removed "Removed in version 26.7.0."
        The `qubits` argument of type `QuantumElements` has been removed.
        Please pass `qubits` of type `list[str] | str` instead, i.e., the quantum
        element UIDs instead of the quantum element instances.

    !!! version-changed "Changed in version 26.4.0."
        The `evaluation_parameters` argument has been added, which is the dictionary of
        parameters used for the newly added evaluation task. All arguments apart from
        `session`, `qpu`, and `qubits` are now keyword arguments.

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits on which to run the experiments, passed by UID. May be either a
            single qubit or a list of qubits.
        delays:
            The delays to sweep over for each qubit. The delays between the two x90
            pulses and the refocusing pulse are `delays / 2`; see the schematic of
            the pulse sequence at the top of the file. Note that `delays` must be
            identical for qubits that use the same measure port.
        evaluation_parameters:
            The dictionary of parameters used for the evaluation task. The default
            evaluation parameters are defined in the `evaluate_experiment` task.
        temporary_parameters:
            The temporary parameters with which to update the quantum elements and
            topology edges. For quantum elements, the dictionary key is the quantum
            element UID. For topology edges, the dictionary key is the edge tuple
            `(tag, source node UID, target node UID)`.
        options:
            The options for building the workflow as an instance of
            [TuneUpWorkflowOptions]. See the docstrings of this class for more details.

    Returns:
        WorkflowBuilder:
            The builder for the experiment workflow.

    Example:
        ```python
        options = EchoWorkflowOptions()
        options.count(10)
        options.transition("ge")
        qpu = QPU(
            quantum_elements=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        result = run(
            session=session,
            qpu=qpu,
            qubits=["q0", "q1"],
            delays=[np.linspace(0, 30e-6, 51), np.linspace(0, 30e-6, 51)],
            options=options,
        )
        ```
    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    exp = create_experiment(
        temp_qpu,
        qubits,
        delays=delays,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(result, qubits, delays)
        qubit_parameters = analysis_results.output
        with workflow.if_(options.evaluate):
            eval_flags = evaluate_experiment(
                analysis_results, qubits, evaluation_parameters
            )
            with workflow.if_(options.update):
                update_qpu(
                    qpu,
                    qubit_parameters["new_parameter_values"],
                    eval_flags=eval_flags,
                )
        with workflow.else_():
            with workflow.if_(options.update):
                update_qpu(
                    qpu,
                    qubit_parameters["new_parameter_values"],
                )
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    delays: QubitSweepPoints,
    options: EchoExperimentOptions | None = None,
) -> Experiment:
    """Creates a Hahn echo Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits on which to run the experiments. May be either a single
            qubit or a list of qubits.
        delays:
            The delays to sweep over for each qubit. The delays between the two x90
            pulses and the refocusing pulse are `delays / 2`; see the schematic of
            the pulse sequence at the top of the file. Note that `delays` must be
            identical for qubits that use the same measure port.
        options:
            The options for building the workflow as an instance of
            [EchoExperimentOptions], inheriting from [TuneupExperimentOptions].
            See the docstrings of these classes for more details.

    Returns:
        Experiment:
            The generated LabOne Q Experiment instance to be compiled and executed.

    Raises:
        ValueError:
            If the conditions in validation.validate_and_convert_qubits_sweeps are not
            fulfilled.

        ValueError:
            If the experiment uses calibration traces and the averaging mode is
            sequential.

    Example:
        ```python
        options = TuneupExperimentOptions()
        options.count = 10
        options.cal_traces = True
        setup = DeviceSetup("my_device")
        qpu = QPU(
            quantum_elements=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        create_experiment(
            qpu=qpu,
            qubits=["q0", "q1"],
            delays=[np.linspace(0, 30e-6, 51), np.linspace(0, 30e-6, 51)],
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = EchoExperimentOptions() if options is None else options

    qubits, delays = validation.validate_and_convert_qubits_sweeps(qubits, delays)
    if (
        opts.use_cal_traces
        and AveragingMode(opts.averaging_mode) == AveragingMode.SEQUENTIAL
    ):
        raise ValueError(
            "'AveragingMode.SEQUENTIAL' (or {AveragingMode.SEQUENTIAL}) cannot be used "
            "with calibration traces because the calibration traces are added "
            "outside the sweep."
        )

    delays_sweep_pars = [
        SweepParameter(f"delays_{q.uid}", q_delays, axis_name=f"{q.uid}")
        for q, q_delays in zip(qubits, delays, strict=False)
    ]
    qop = qpu.quantum_operations
    # We will fix the length of the measure section to the longest section among
    # the qubits to allow the qubits to have different readout and/or
    # integration lengths.
    max_measure_section_length = qop.measure_section_length(qubits)
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        with dsl.sweep(
            name="echo_sweep",
            parameter=delays_sweep_pars,
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
                    for q, delay in zip(qubits, delays_sweep_pars, strict=False):
                        qop.prepare_state.omit_section(q, opts.transition[0])
                        qop.ramsey(
                            q,
                            delay,
                            0,
                            echo_pulse=opts.refocus_qop,
                            transition=opts.transition,
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


@workflow.task(save=False)
def evaluate_experiment(
    analysis_results: WorkflowResult,
    qubits: QuantumElements,
    evaluation_parameters: dict[str, Any] | None = None,
) -> dict[str, dict[str, bool]]:
    """Evaluates the Hahn echo analysis workflow result.

    Arguments:
        analysis_results:
            The analysis workflow results.
        qubits:
            The qubits to run the experiments on.
        evaluation_parameters:
            The evaluation parameters.

    Returns:
        The evaluation flags.
    """
    return evaluate_parameter_and_fit_r2_thresholds(
        analysis_results.output["old_parameter_values"],
        analysis_results.output["new_parameter_values"],
        analysis_results.tasks["fit_data"].output,
        qubits,
        parameter="ge_T2",
        default_parameter_threshold=0.00001,
        default_fit_r2_threshold=0.94,
        evaluation_parameters=evaluation_parameters,
    )
