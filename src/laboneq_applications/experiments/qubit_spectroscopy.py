"""This module defines the qubit spectroscopy experiment.

In this experiment, we sweep the frequency of a qubit drive pulse to characterize
the qubit transition frequency.

The qubit spectroscopy experiment has the following pulse sequence:

    qb --- [ prep transition ] --- [ x180_transition (swept frequency)] --- [ measure ]

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.simple import Experiment, SweepParameter

from laboneq_applications import dsl, workflow
from laboneq_applications.analysis.qubit_spectroscopy import analysis_workflow
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.experiments.options import (
    QubitSpectroscopyExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks import compile_experiment, run_experiment, update_qubits
from laboneq_applications.tasks.parameter_updating import temporary_modify

if TYPE_CHECKING:
    from laboneq.dsl.quantum import TransmonParameters
    from laboneq.dsl.session import Session

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import Qubits, QubitSweepPoints


@workflow.workflow(name="qubit_spectroscopy")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    frequencies: QubitSweepPoints,
    temporary_parameters: dict[str, dict | TransmonParameters] | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Qubit Spectroscopy Workflow.

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
        frequencies:
            The qubit frequencies to sweep over for the qubit drive pulse. If `qubits`
            is a single qubit, `frequencies` must be a list of numbers or an array.
            Otherwise, it must be a list of lists of numbers or arrays.
        temporary_parameters:
            The temporary parameters to update the qubits with.
        options:
            The options for building the workflow.
            In addition to options from [WorkflowOptions], the following
            custom options are supported:
                - create_experiment: The options for creating the experiment.

    Returns:
        WorkflowBuilder:
            The builder for the experiment workflow.

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
                np.linspace(6.0e9, 6.3e9, 101),
                np.linspace(5.8e9, 6.2e9, 101)
            ]
            options=options,
        ).run()
        ```
    """
    qubits = temporary_modify(qubits, temporary_parameters)
    exp = create_experiment(
        qpu,
        qubits,
        frequencies=frequencies,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(result, qubits, frequencies)
        qubit_parameters = analysis_results.output
        with workflow.if_(options.update):
            update_qubits(qpu, qubit_parameters["new_parameter_values"])
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: Qubits,
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
            See [QubitSpectroscopyExperimentOptions] and [BaseExperimentOptions] for
            accepted options.
            Overwrites the options from [QubitSpectroscopyExperimentOptions] and
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
                np.linspace(6.0e9, 6.3e9, 101),
                np.linspace(5.8e9, 6.2e9, 101)
            ]
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = QubitSpectroscopyExperimentOptions() if options is None else options

    qubits, frequencies = validate_and_convert_qubits_sweeps(qubits, frequencies)

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
        for q, q_frequencies in zip(qubits, frequencies):
            with dsl.sweep(
                name=f"freqs_{q.uid}",
                parameter=SweepParameter(f"frequency_{q.uid}", q_frequencies),
            ) as frequency:
                qop.set_frequency(q, frequency)
                qop.spectroscopy_drive(q)
                sec = qop.measure(q, dsl.handles.result_handle(q.uid))
                # we fix the length of the measure section to the longest section among
                # the qubits to allow the qubits to have different readout and/or
                # integration lengths.
                sec.length = max_measure_section_length
                qop.passive_reset(q, delay=opts.spectroscopy_reset_delay)
