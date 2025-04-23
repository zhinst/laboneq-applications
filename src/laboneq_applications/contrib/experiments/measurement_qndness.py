# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the qubit measurement QND experiment.

In this experiment, we first prepare the qubit in a superposition state using an
x90 pulse, ensuring a random yet equal probability of measuring each qubit state.
We then perform an initial measurement pulse, followed by additional measurement
pulses. By comparing the results of these measurements, we can determine whether
the qubit state is preserved after the initial measurement.

The pulse sequence for the qubit measurement QND experiment is as follows:

    qb - [ x90 ] [ measure ] - [ delay (fixed) ] - [ measure ]

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from laboneq.dsl.enums import AcquisitionType, AveragingMode, SectionAlignment
from laboneq.simple import Experiment, dsl
from laboneq.workflow import if_, option_field, return_, task, task_options, workflow
from laboneq.workflow.tasks import compile_experiment, run_experiment

from laboneq_applications.contrib.analysis import measurement_qndness
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.experiments.options import (
    BaseExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks.parameter_updating import (
    temporary_modify,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import TransmonParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import Qubits


# create additional options for the QNDness experiment
@task_options(base_class=BaseExperimentOptions)
class QNDnessExperimentOptions:
    """Base options for the resonator spectroscopy experiment.

    Additional attributes:
        acquisition_type:
            Acquisition type to use for the experiment.
            Default: `AcquisitionType.DISCRIMINATION`.
        averaging_mode:
            Averaging mode to use for the experiment.
            Default: `AveragingMode.SINGLE_SHOT`.
    """
    acquisition_type: AcquisitionType = option_field(
        AcquisitionType.DISCRIMINATION,
        description="Acquisition type to use for the experiment.",
    )

    averaging_mode: AveragingMode = option_field(
        AveragingMode.SINGLE_SHOT,
        description="Averaging mode to use for the experiment.",
    )
    transition: Literal["ge", "ef"] = option_field(
        "ge",
        description="Transition to perform the experiment on. May be any"
        " transition supported by the quantum operations.",
    )
    use_cal_traces: bool = option_field(
        True, description="Whether to include calibration traces in the experiment."
    )
    delay_between_measurements: float = option_field(
        1e-6, description="Time delay between successive measurement operations."
    )


@workflow(name="measurement_qndness")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    temporary_parameters: dict[str, dict | TransmonParameters] | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Qubit Measurement QNDness Workflow.

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
        temporary_parameters:
            The temporary parameters to update the qubits with.
        options:
            The options for building the workflow.
            In addition to options from [WorkflowOptions], the following
            custom options are supported:
            - create_experiment: The options for creating the experiment.

    Returns:
        None

    Example:
        ```python
        options = experiment_workflow.options()
        options.count(2**9)
        options.delay_between_measurements(1e-6)
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        qubits = qpu.qubits
        temporary_parameters = {}
        for q in qubits_to_measure:
            temp_pars = deepcopy(q.parameters)
            temporary_parameters[q.uid] = temp_pars
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubits_to_measure,
            temporary_parameters=temporary_parameters,
            options=options,
        ).run()
        ```
    """
    qubits = temporary_modify(qubits, temporary_parameters)
    exp = create_experiment(qpu, qubits)
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with if_(options.do_analysis):
        analysis_results = measurement_qndness.analysis_workflow(result, qubits)
    return_(analysis_results)


@task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: Qubits,
    options: QNDnessExperimentOptions | None = None,
) -> Experiment:
    """Creates a Measurement QND Experiment.

    Arguments:
        qpu:
            The QPU consisting of the original QPUs and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        options:
            The options for building the experiment.
            See [QNDnessExperimentOptions] and [BaseExperimentOptions] for
            accepted options.

    Returns:
        Experiment: The generated LabOne Q experiment instance to be compiled
        and executed.

    Example:
        ```python
        options = QNDnessExperimentOptions()
        qpu = QPU(
            qubits=[q1, q2],
        )
        create_experiment(
            qpu=qpu,
            qubits=[q1, q2],
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    qubits = validate_and_convert_qubits_sweeps(qubits)
    opts = QNDnessExperimentOptions() if options is None else options
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
        with dsl.section(name="main", alignment=SectionAlignment.RIGHT):
            with dsl.section(name="main_drive", alignment=SectionAlignment.RIGHT):
                for q in qubits:
                    qop.prepare_state(q, opts.transition[0])
                    qop.x90(q, opts.transition)
            with dsl.section(name="measure_first", alignment=SectionAlignment.LEFT):
                for q in qubits:
                    sec = qop.measure(q, f"measure_1_{q.uid}")
                    # Fix the length of the measure section
                    sec.length = max_measure_section_length
            with dsl.section(name="delay", alignment=SectionAlignment.LEFT):
                for q in qubits:
                    qop.delay(q, opts.delay_between_measurements)
            with dsl.section(name="measure_second", alignment=SectionAlignment.LEFT):
                for q in qubits:
                    sec = qop.measure(q, f"measure_2_{q.uid}")
                    # Fix the length of the measure section
                    sec.length = max_measure_section_length
                    qop.passive_reset(q)
