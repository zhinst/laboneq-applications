"""This module defines the IQ_blob experiment.

In this experiment, we perform single-shot measurements with the qubits prepared
in the states g, e, and/or f.

The IQ blob experiment has the following pulse sequence:

    qb --- [ prepare transition ] --- [ measure ]

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.simple import AveragingMode, Experiment

from laboneq_applications.core import handles
from laboneq_applications.core.build_experiment import qubit_experiment
from laboneq_applications.core.options import (
    BaseExperimentOptions,
)
from laboneq_applications.core.quantum_operations import dsl
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.tasks import compile_experiment, run_experiment
from laboneq_applications.workflow import WorkflowOptions, task, workflow

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.dsl.session import Session

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import Qubits


class IQBlobExperimentOptions(BaseExperimentOptions):
    """Base options for the iq_blobs experiment.

    Additional attributes:
        averaging_mode:
            Averaging mode used for the experiment
            Default: `AveragingMode.SINGLE_SHOT`
        use_cal_traces:
            Default: `False`.
    """

    averaging_mode: AveragingMode = AveragingMode.SINGLE_SHOT
    use_cal_traces: bool = False


class IQBlobsWorkflowOptions(WorkflowOptions):
    """Option for iq_blobs workflow.

    Attributes:
        create_experiment (IQBlobExperimentOptions):
            The options for creating the experiment.
    """

    create_experiment: IQBlobExperimentOptions = IQBlobExperimentOptions()


options = IQBlobsWorkflowOptions


@workflow
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    states: Sequence[str],
    options: IQBlobsWorkflowOptions | None = None,
) -> None:
    """The IQblob Workflow.

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
            The qubits to run the experiments on.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].
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
        options = SpectroscopyWorkflowOptions()
        options.create_experiment.count = 10
        qpu = QPU(
            setup=DeviceSetup("my_device"),
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            qop=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        result = run(
            session=session,
            qpu=qpu,
            qubits=[temp_qubits[0],temp_qubits[1]],
            states="ge",
            options=options,
        )
        ```
    """
    exp = create_experiment(
        qpu,
        qubits,
        states,
    )
    compiled_exp = compile_experiment(session, exp)
    _result = run_experiment(session, compiled_exp)


@task
@qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: Qubits,
    states: Sequence[str],
    options: IQBlobExperimentOptions | None = None,
) -> Experiment:
    """Creates an IQblob Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubit to run the experiments on.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].
        options:
            The options for building the experiment.
            See [IQBlobsExperimentOptions] and [BaseExperimentOptions] for
            accepted options.
            Overwrites the options from [TuneupExperimentOptions] and
            [BaseExperimentOptions].

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Raises:
        ValueError:
            The argument 'states' contains other letters than "g", "e" or "f".

    Example:
        ```python
        options = {
            "count": 10,
        }
        options = IQBlobExperimentOptions(**options)
        setup = DeviceSetup()
        qpu = QPU(
            setup=DeviceSetup("my_device"),
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            qop=TunableTransmonOperations(),
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
    qubits = validate_and_convert_qubits_sweeps(qubits)

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
                name=f"iq_{q.uid}",
            ):
                for state in states:
                    qpu.qop.prepare_state(q, state)
                    qpu.qop.measure(
                        q,
                        handles.calibration_trace_handle(q.uid, state),
                    )
                    qpu.qop.passive_reset(q)
