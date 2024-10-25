"""This module defines the IQ_blob experiment.

In this experiment, we perform single-shot measurements with the qubits prepared
in the states g, e, and/or f.

The IQ blob experiment has the following pulse sequence:

    qb --- [ prepare transition ] --- [ measure ]

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.simple import AveragingMode, Experiment

from laboneq_applications import dsl, workflow
from laboneq_applications.analysis.iq_blobs import analysis_workflow
from laboneq_applications.experiments.options import (
    BaseExperimentOptions,
    WorkflowOptions,
)
from laboneq_applications.tasks import compile_experiment, run_experiment
from laboneq_applications.tasks.parameter_updating import temporary_modify
from laboneq_applications.workflow import option_field, options

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.dsl.quantum import (
        TransmonParameters,
    )
    from laboneq.dsl.session import Session

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import Qubits


@options
class IQBlobExperimentOptions(BaseExperimentOptions):
    """Options for the iq_blobs experiment.

    The purpose of this class is to change the default value of averaging_mode in
    BaseExperimentOptions.

    Attributes:
        averaging_mode:
            Averaging mode used for the experiment
            Default: `AveragingMode.SINGLE_SHOT`
    """

    averaging_mode: AveragingMode = option_field(
        AveragingMode.SINGLE_SHOT, description="Averaging mode used for the experiment"
    )


@options
class IQBlobExperimentWorkflowOptions(WorkflowOptions):
    """Options for the iq_blobs experiment workflow.

    Attributes:
        do_analysis (bool):
            Whether to run the analysis workflow.
            Default: True
    """

    do_analysis: bool = option_field(
        True, description="Whether to run the analysis workflow."
    )


@workflow.workflow(name="iq_blobs")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    states: Sequence[str],
    temporary_parameters: dict[str, dict | TransmonParameters] | None = None,
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
    qubits = temporary_modify(qubits, temporary_parameters)
    exp = create_experiment(
        qpu,
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
    qubits: Qubits,
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
    qubits = dsl.validation.validate_and_convert_qubits_sweeps(qubits)

    qop = qpu.quantum_operations
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for q in qubits:
            with dsl.section(
                name=f"iq_blobs_{q.uid}",
            ):
                for state in states:
                    qop.prepare_state(q, state)
                    qop.measure(
                        q,
                        dsl.handles.result_handle(q.uid, suffix=state),
                    )
                    qop.passive_reset(q)
