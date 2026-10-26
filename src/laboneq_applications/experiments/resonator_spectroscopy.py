# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the resonator spectroscopy experiment.

In this experiment, we sweep the resonator frequency
of a measure pulse to characterize the resonators coupled to the qubits.

The resonator spectroscopy experiment has the following pulse sequence:

    qb --- [ measure ]

If multiple qubits are passed to the `experiment_workflow`, the above sequence
is applied in parallel on all the qubits. Since this experiment sweeps the
frequency of the hardware oscillator of the readout line, it can only be run in
parallel on qubits whose measure lines use distinct QA channels.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from laboneq import workflow
from laboneq.dsl.enums import AcquisitionType
from laboneq.simple import Experiment, SweepParameter, dsl
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.analysis.resonator_spectroscopy import analysis_workflow
from laboneq_applications.conversion import (
    as_quantum_element_list,
    sweeps_per_quantum_element,
)
from laboneq_applications.experiments.options import (
    ResonatorSpectroscopyExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks.parameter_updating import (
    _handle_deprecated_qubit_argument,
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
    update_qpu,
)
from laboneq_applications.tasks.verify_qa_channels import _verify_unique_qa_channels

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session
    from laboneq.workflow import WorkflowResult

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.workflow(name="resonator_spectroscopy")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: list[str] | str | None = None,
    *,
    frequencies: QubitSweepPoints,
    qubit: str | None = None,
    evaluation_parameters: dict[str, Any] | None = None,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Resonator Spectroscopy Workflow.

    The workflow consists of the following steps:

    - [create_experiment][]
    - [compile_experiment][]
    - [run_experiment][]
    - [analysis_workflow][]
    - [evaluate_experiment][]
    - [update_qpu][]

    !!! version-changed "Deprecated in version 26.10.0"
        The `qubit` argument is deprecated. Please use `qubits` instead.

    !!! version-changed "Changed in version 26.10.0"
        The `qubit` argument was renamed to `qubits` and now also accepts
        multiple qubits, passed as a list of UIDs. The frequencies of multiple
        qubits are swept in parallel, which is only possible if the measure
        lines of the qubits use distinct QA channels.

    !!! version-removed "Removed in version 26.7.0"
        The `qubit` argument of type `QuantumElement` has been removed.
        Please pass `qubit` of type `str` instead, i.e., the quantum
        element UID instead of the quantum element instance.

    !!! version-changed "Changed in version 26.4.0"
        The `evaluation_parameters` argument has been added, which is the dictionary of
        parameters used for the newly added evaluation task. All arguments apart from
        `session`, `qpu`, and `qubits` are now keyword arguments.

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on, passed by UID. May be either a
            single qubit or a list of qubits. The measure lines of the qubits
            must use distinct QA channels, because this experiment sweeps the
            frequency of the hardware oscillator of the readout line, which is
            shared by all qubits multiplexed on the same QA channel.
        frequencies:
            The resonator frequencies to sweep over for the readout pulse (or CW)
            sent to the resonator of each qubit. If `qubits` is a single qubit,
            `frequencies` must be a list of numbers or an array. Otherwise it
            must be a list of lists of numbers or arrays.
        evaluation_parameters:
            The dictionary of parameters used for the evaluation task. The default
            evaluation parameters are defined in the `evaluate_experiment` task.
        temporary_parameters:
            The temporary parameters with which to update the quantum elements and
            topology edges. For quantum elements, the dictionary key is the quantum
            element UID. For topology edges, the dictionary key is the edge tuple
            `(tag, source node UID, target node UID)`.
        qubit:
            Deprecated. The single qubit to run on, passed by UID for the
            experiment workflows (quantum element for the analysis). Use
            `qubits` instead.
        options:
            The options for building the workflow.
            In addition to options from
            [WorkflowOptions][laboneq.workflow.WorkflowOptions], the following
            custom options are supported:

            - create_experiment: The options for creating the experiment.

    Returns:
        result:
            The result of the workflow.

    Example:
        ```python
        options = experiment_workflow.options()
        options.count(10)
        # QPU from a single-qubit device setup
        qpu = QPU(
            quantum_elements=TunableTransmonQubit.from_device_setup(setup),
            quantum_operations=TunableTransmonOperations(),
        )
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits="q0",
            frequencies=np.linspace(7.1e9, 7.6e9, 501),
            options=options,
        ).run()
        ```
    """
    qubits = _handle_deprecated_qubit_argument(qubits, qubit)
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    temp_qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    _verify_unique_qa_channels(session, temp_qubits)
    exp = create_experiment(
        temp_qpu,
        temp_qubits,
        frequencies=frequencies,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(result, temp_qubits, frequencies)
        qubit_parameters = analysis_results.output
        with workflow.if_(options.evaluate):
            eval_flags = evaluate_experiment(
                analysis_results, temp_qubits, evaluation_parameters
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
    options: ResonatorSpectroscopyExperimentOptions | None = None,
) -> Experiment:
    """Creates a Resonator Spectroscopy Experiment.

    !!! version-changed "Changed in version 26.10.0"
        The `qubit` argument was renamed to `qubits` and now also accepts
        multiple qubits. The frequencies of multiple qubits are swept in
        parallel.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        frequencies:
            The resonator frequencies to sweep over for each qubit. If `qubits`
            is a single qubit, `frequencies` must be a list of numbers or an
            array. Otherwise it must be a list of lists of numbers or arrays.
            The frequencies of all the qubits must contain the same number of
            sweep points.
        options:
            The options for building the experiment.
            See [ResonatorSpectroscopyExperimentOptions][] and
            [BaseExperimentOptions][laboneq_applications.experiments.options.BaseExperimentOptions]
            for accepted options.
            Overwrites the options from
            [BaseExperimentOptions][laboneq_applications.experiments.options.BaseExperimentOptions].

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Raises:
        ValueError:
            If the acquisition_type is neither AcquisitionType.SPECTROSCOPY
            nor AcquisitionType.INTEGRATION.

        ValueError:
            If the frequencies of the qubits do not contain the same number of
            sweep points.

    Example:
        ```python
        options = {
            "count": 10,
            "spectroscopy_reset_delay": 3e-6,
        }
        options = ResonatorSpectroscopyExperimentOptions(**options)
        # QPU from a single-qubit device setup
        qpu = QPU(
            quantum_elements=TunableTransmonQubit.from_device_setup(setup),
            quantum_operations=TunableTransmonOperations(),
        )
        q0 = qpu["q0"]
        create_experiment(
            qpu=qpu,
            qubits=q0,
            frequencies=np.linspace(7.1e9, 7.6e9, 501),
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = ResonatorSpectroscopyExperimentOptions() if options is None else options
    qubits = as_quantum_element_list(qubits)

    # Guard against wrong options for the acquisition type
    if opts.acquisition_type not in (
        AcquisitionType.SPECTROSCOPY,
        AcquisitionType.INTEGRATION,
    ):
        raise ValueError(
            "The only allowed acquisition types for this experiment are "
            "'AcquisitionType.SPECTROSCOPY' (on QCCS) and 'AcquisitionType.INTEGRATION'"
            " (on ZQCS) because it sweeps the frequency of a hardware oscillator.",
        )

    freq_sweep_pars = [
        SweepParameter(f"frequencies_{q.uid}", q_frequencies)
        for q, q_frequencies in sweeps_per_quantum_element(qubits, frequencies)
    ]
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
            name="resonator_frequency_sweep",
            parameter=freq_sweep_pars,
        ):
            for q, q_frequency in zip(qubits, freq_sweep_pars, strict=False):
                qop.set_frequency(q, frequency=q_frequency, readout=True)
                if opts.use_cw:
                    qop.acquire(
                        q,
                        handle=dsl.handles.result_handle(q.uid),
                        kernel_pulses="continuous",
                    )
                else:
                    qop.measure(q, handle=dsl.handles.result_handle(q.uid))
                qop.delay(q, opts.spectroscopy_reset_delay)


@workflow.task(save=False)
def evaluate_experiment(
    analysis_results: WorkflowResult,
    qubits: QuantumElements,
    evaluation_parameters: dict[str, Any] | None = None,
) -> dict[str, dict[str, bool]]:
    """Evaluates the resonator spectroscopy analysis workflow result.

    !!! version-changed "Changed in version 26.10.0"
        The `qubit` argument was renamed to `qubits` and now also accepts
        multiple qubits.

    Arguments:
        analysis_results:
            The analysis workflow results.
        qubits:
            The qubits to run the experiments on. May be either a single qubit
            or a list of qubits.
        evaluation_parameters:
            The evaluation parameters.

    Returns:
        The evaluation flags.
    """
    qubits = as_quantum_element_list(qubits)
    workflow.log(
        logging.WARNING,
        "The `evaluate_experiment` task for "
        "`resonator_spectroscopy` has not been implemented by the user. Marking "
        "experiment as passed.",
    )
    return {q.uid: {"success": True, "update": False} for q in qubits}
