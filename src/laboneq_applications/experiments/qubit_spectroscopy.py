# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the qubit spectroscopy experiment.

In this experiment, we sweep the frequency of a qubit drive pulse to characterize
the qubit transition frequency.

The qubit spectroscopy experiment has the following pulse sequence:

    qb --- [ prep transition ] --- [ x180_transition (swept frequency)] --- [ measure ]

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from laboneq import workflow
from laboneq.simple import Experiment, SweepParameter, dsl
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.analysis.qubit_spectroscopy import analysis_workflow
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.experiments.options import (
    QubitSpectroscopyExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks import (
    evaluate_parameter_and_fit_r2_thresholds,
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


@workflow.workflow(name="qubit_spectroscopy")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: list[str] | str,
    *,
    frequencies: QubitSweepPoints,
    evaluation_parameters: dict[str, Any] | None = None,
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
    - [evaluate_experiment]()
    - [update_qpu]()

    !!! version-removed "Removed in version 26.7.0."
        The `qubits` argument of type `QuantumElements` has been removed.
        Please pass `qubits` of type `list[str] | str` instead, i.e., the quantum
        element UIDs instead of the quantum element instances.

    !!! version-changed "Changed in version 26.4.0."
        The `evaluation_parameters` argument has been added. This argument replaces the
        `evaluation_parameter`, `evaluation_parameter_thresholds`, and
        `evaluation_fit_r2_thresholds` arguments.

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on, passed by UID. May be either a single
            qubit or a list of qubits.
        frequencies:
            The qubit frequencies to sweep over for the qubit drive pulse. If `qubits`
            is a single qubit, `frequencies` must be a list of numbers or an array.
            Otherwise, it must be a list of lists of numbers or arrays.
        evaluation_parameters:
            The dictionary of parameters used for the evaluation task. The default
            evaluation parameters are defined in the `evaluate_experiment` task.
        temporary_parameters:
            The temporary parameters with which to update the quantum elements and
            topology edges. For quantum elements, the dictionary key is the quantum
            element UID. For topology edges, the dictionary key is the edge tuple
            `(tag, source node UID, target node UID)`.
        options:
            The options for building the workflow.
            In addition to options from
            [WorkflowOptions][laboneq.workflow.WorkflowOptions], the following
            custom options are supported:
                - create_experiment: The options for creating the experiment.

    Returns:
        WorkflowBuilder:
            The builder for the experiment workflow.

    Example:
        ```python
        options = experiment_workflow.options()
        options.count(10)
        # QPU from a two-qubit device setup
        qpu = QPU(
            quantum_elements=TunableTransmonQubit.from_device_setup(setup),
            quantum_operations=TunableTransmonOperations(),
        )
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=["q0", "q1"],
            frequencies=[
                np.linspace(6.0e9, 6.3e9, 101),
                np.linspace(5.8e9, 6.2e9, 101),
            ],
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
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(result, qubits, frequencies)
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
    frequencies: QubitSweepPoints,
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
        options:
            The options for building the experiment.
            See [QubitSpectroscopyExperimentOptions] and
            [BaseExperimentOptions][laboneq_applications.experiments.options.BaseExperimentOptions]
            for accepted options.
            Overwrites the options from [QubitSpectroscopyExperimentOptions] and
            [BaseExperimentOptions][laboneq_applications.experiments.options.BaseExperimentOptions].

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
        # QPU from a two-qubit device setup
        qpu = QPU(
            quantum_elements=TunableTransmonQubit.from_device_setup(setup),
            quantum_operations=TunableTransmonOperations(),
        )
        q0, q1 = qpu["q0"], qpu["q1"]
        create_experiment(
            qpu=qpu,
            qubits=[q0, q1],
            frequencies=[
                np.linspace(6.0e9, 6.3e9, 101),
                np.linspace(5.8e9, 6.2e9, 101),
            ],
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = QubitSpectroscopyExperimentOptions() if options is None else options

    qubits, frequencies = validate_and_convert_qubits_sweeps(qubits, frequencies)

    qop = qpu.quantum_operations
    max_measure_section_length = qop.measure_section_length(qubits)
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for q, q_frequencies in zip(qubits, frequencies, strict=False):
            with dsl.sweep(
                name=f"freqs_{q.uid}",
                parameter=SweepParameter(f"frequency_{q.uid}", q_frequencies),
            ) as frequency:
                qop.set_frequency(q, frequency)
                qop.qubit_spectroscopy_drive(q)
                sec = qop.measure(q, dsl.handles.result_handle(q.uid))
                # we fix the length of the measure section to the longest section among
                # the qubits to allow the qubits to have different readout and/or
                # integration lengths.
                sec.length = max_measure_section_length
                qop.passive_reset(q, delay=opts.spectroscopy_reset_delay)


@workflow.task(save=False)
def evaluate_experiment(
    analysis_results: WorkflowResult,
    qubits: QuantumElements,
    evaluation_parameters: dict[str, Any] | None = None,
) -> dict[str, dict[str, bool]]:
    """Evaluates the qubit spectroscopy analysis workflow result.

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
        parameter="resonance_frequency_ge",
        default_parameter_threshold=2e8,
        default_fit_r2_threshold=0.99,
        evaluation_parameters=evaluation_parameters,
    )
