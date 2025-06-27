# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the IQ_blob experiment.

In this experiment, we perform single-shot measurements with the qubits prepared
in the states g, e, and/or f.

The IQ blob experiment has the following pulse sequence:

    qb --- [ prepare transition ] --- [ measure ]

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.simple import AveragingMode, Experiment, dsl
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.analysis.iq_blobs import analysis_workflow
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import (
    BaseExperimentOptions,
)
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import QuantumElements


@workflow.task_options(base_class=BaseExperimentOptions)
class IQBlobExperimentOptions:
    """Options for the iq_blobs experiment.

    The purpose of this class is to change the default value of averaging_mode in
    BaseExperimentOptions.

    Attributes:
        averaging_mode:
            Averaging mode used for the experiment
            Default: `AveragingMode.SINGLE_SHOT`
    """

    averaging_mode: AveragingMode = workflow.option_field(
        AveragingMode.SINGLE_SHOT, description="Averaging mode used for the experiment"
    )


@workflow.workflow_options
class IQBlobExperimentWorkflowOptions:
    """Options for the iq_blobs experiment workflow.

    Attributes:
        do_analysis (bool):
            Whether to run the analysis workflow.
            Default: True
    """

    do_analysis: bool = workflow.option_field(
        True, description="Whether to run the analysis workflow."
    )


@workflow.workflow(name="iq_blobs")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    states: Sequence[str],
    temporary_parameters: dict[str, dict | QuantumParameters] | None = None,
    options: IQBlobExperimentWorkflowOptions | None = None,
) -> None:
    """The IQ-blob experiment Workflow.

    The workflow consists of the following steps:

    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()
    - [analysis_workflow]()

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].
        temporary_parameters:
            The temporary parameters to update the qubits with.
        options:
            The options for building the workflow as an instance of
            [TuneUpWorkflowOptions]. See the docstring of this class for more details.

    Returns:
        WorkflowBuilder:
            The builder for the experiment workflow.

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
            qubits=[temp_qubits[0],temp_qubits[1]],
            states="ge",
            options=options,
        )
        ```
    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(qpu, qubits)
    exp = create_experiment(
        temp_qpu,
        qubits,
        states,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_workflow(result, qubits, states)
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    states: Sequence[str],
    options: IQBlobExperimentOptions | None = None,
) -> Experiment:
    """Creates an IQ-blob Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubit to run the experiments on.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].
        options:
            The options for building the experiment as an in stance of
            [IQBlobExperimentOptions]. See the docstring of this class for more details.

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Example:
        ```python
        options = IQBlobExperimentOptions()
        options.count(10)
        setup = DeviceSetup()
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        create_experiment(
            qpu=qpu,
            qubits=[temp_qubits[0],temp_qubits[1]],
            states="ge",
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = IQBlobExperimentOptions() if options is None else options
    qubits = validation.validate_and_convert_qubits_sweeps(qubits)

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
        qop.calibration_traces.omit_section(
            qubits=qubits,
            states=states,
            active_reset=opts.active_reset,
            active_reset_states=opts.active_reset_states,
            active_reset_repetitions=opts.active_reset_repetitions,
            measure_section_length=max_measure_section_length,
        )
