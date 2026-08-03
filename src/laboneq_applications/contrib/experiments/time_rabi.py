# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the time-rabi experiment.

In this experiment, we sweep the length of a drive pulse on a given qubit transition
in order to determine the pulse length that induces a rotation of pi.

The time-rabi experiment has the following pulse sequence:

    qb --- [ prep transition ] --- [ x180_transition ] --- [ measure ]

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from laboneq.dsl.quantum import QuantumParameters
from laboneq.simple import Experiment, SweepParameter, dsl
from laboneq.workflow import else_, if_, task, workflow
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.contrib.analysis.time_rabi import analysis_workflow
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
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session
    from laboneq.workflow import WorkflowResult

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow(name="time_rabi")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: list[str] | str,
    *,
    lengths: QubitSweepPoints,
    evaluation_parameters: dict[str, Any] | None = None,
    temporary_parameters: dict[str, dict | QuantumParameters] | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Time Rabi Workflow.

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

    !!! version-changed "Changed in version 26.1.0."
        The `temporary_parameters` positional argument was added in the
        penultimate position. Note that this is a breaking change if
        calling the experiment workflow with the `options` positional argument.

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on, passed by UID. May be either a single
            qubit or a list of qubits.
        lengths:
            The drive-pulse lengths to sweep over for each qubit. If `qubits` is a
            single qubit, `lengths` must be a list of numbers or an array. Otherwise
            it must be a list of lists of numbers or arrays.
        evaluation_parameters:
            The dictionary of parameters used for the evaluation task. The default
            evaluation parameters are defined in the `evaluate_experiment` task.
        temporary_parameters:
            The temporary parameters to update the qubits with.
        options:
            The options for building the workflow.
            In addition to options from
            [WorkflowOptions][laboneq.workflow.WorkflowOptions], the following
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
        # QPU from a two-qubit device setup
        qpu = QPU(
            quantum_elements=TunableTransmonQubit.from_device_setup(setup),
            quantum_operations=TunableTransmonOperations(),
        )
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=["q0", "q1"],
            lengths=[np.linspace(100e-9, 500e-9, 11), np.linspace(100e-9, 500e-9, 11)],
            options=options,
        ).run()
        ```
    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    exp = create_experiment(
        qpu,
        qubits,
        lengths=lengths,
    )
    compiled_exp = compile_experiment(session, exp)
    _result = run_experiment(session, compiled_exp)
    with if_(options.do_analysis):
        analysis_results = analysis_workflow(_result, qubits, lengths)
        qubit_parameters = analysis_results.tasks["extract_qubit_parameters"].output
        with if_(options.evaluate):
            eval_flags = evaluate_experiment(
                analysis_results, qubits, evaluation_parameters
            )
            with if_(options.update):
                update_qpu(
                    qpu,
                    qubit_parameters["new_parameter_values"],
                    eval_flags=eval_flags,
                )
        with else_():
            with if_(options.update):
                update_qpu(
                    qpu,
                    qubit_parameters["new_parameter_values"],
                )


@task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    lengths: QubitSweepPoints,
    options: TuneupExperimentOptions | None = None,
) -> Experiment:
    """Creates a length-Rabi experiment Workflow.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        lengths:
            The drive-pulse lengths to sweep over for each qubit. If `qubits` is a
            single qubit, `lengths` must be a list of numbers or an array. Otherwise
            it must be a list of lists of numbers or arrays.
        options:
            The options for building the experiment.
            See [TuneupExperimentOptions] and
            [BaseExperimentOptions][laboneq_applications.experiments.options.BaseExperimentOptions]
            for accepted options.
            Overwrites the options from [TuneupExperimentOptions] and
            [BaseExperimentOptions][laboneq_applications.experiments.options.BaseExperimentOptions].

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Raises:
        ValueError:
            If the qubits and qubit_lengths are not of the same length.

        ValueError:
            If qubit_lengths is not a list of numbers when a single qubit is passed.

        ValueError:
            If qubit_lengths is not a list of lists of numbers.

    Example:
        ```python
        options = TuneupExperimentOptions()
        options.count = 10
        options.transition = "ge"
        options.use_cal_traces = True
        # QPU from a two-qubit device setup
        qpu = QPU(
            quantum_elements=TunableTransmonQubit.from_device_setup(setup),
            quantum_operations=TunableTransmonOperations(),
        )
        q0, q1 = qpu["q0"], qpu["q1"]
        create_experiment(
            qpu=qpu,
            qubits=[q0, q1],
            lengths=[np.linspace(100e-9, 500e-9, 11), np.linspace(100e-9, 500e-9, 11)],
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = TuneupExperimentOptions() if options is None else options
    qubits, lengths = validation.validate_and_convert_qubits_sweeps(qubits, lengths)
    qop = qpu.quantum_operations
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for q, q_lengths in zip(qubits, lengths, strict=False):
            with dsl.sweep(
                name=f"amps_{q.uid}",
                parameter=SweepParameter(f"length_{q.uid}", q_lengths),
            ) as length:
                qop.prepare_state(q, opts.transition[0])
                qop.x180(q, length=length, transition=opts.transition)
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


@task(save=False)
def evaluate_experiment(
    analysis_results: WorkflowResult,
    qubits: QuantumElements,
    evaluation_parameters: dict[str, Any] | None = None,
) -> dict[str, dict[str, bool]]:
    """Evaluates the time Rabi analysis workflow result.

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
        parameter="ge_drive_length",
        default_parameter_threshold=200e-9,
        default_fit_r2_threshold=0.999,
        evaluation_parameters=evaluation_parameters,
    )
