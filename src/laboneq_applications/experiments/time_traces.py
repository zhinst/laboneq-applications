"""This module defines the raw traces measurement.

In this measurement, raw traces are acquired for qubits in different states in order to
compute the optimal integration kernels for qubit readout.

The raw traces measurement has the following pulse sequence

    qb --- [prepare state] --- [measure]

The corresponding traces are acquired for each combination of the qubits and states
given by the user.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from laboneq.simple import AcquisitionType, Experiment

from laboneq_applications import dsl, workflow
from laboneq_applications.core.build_experiment import qubit_experiment
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.experiments.options import (
    BaseExperimentOptions,
)
from laboneq_applications.tasks import compile_experiment, run_experiment

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.dsl.session import Session

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import Qubits


class TimeTracesExperimentOptions(BaseExperimentOptions):
    """Options for the time traces experiment.

    Additional attributes:
        acquisition_type:
            Acquisition type to use for the experiment.
            Default: `AcquisitionType.RAW`.
    """

    acquisition_type: str | AcquisitionType = AcquisitionType.RAW


class TimeTracesWorkflowOptions(workflow.WorkflowOptions):
    """Option for time traces workflow.

    Attributes:
        create_experiment (TimeTracesExperimentOptions):
            The options for creating the experiment.
    """

    create_experiment: TimeTracesExperimentOptions = TimeTracesExperimentOptions()


options = TimeTracesWorkflowOptions


@workflow.workflow(name="time_traces")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    states: Sequence[Literal["g", "e", "f"]],
    options: TimeTracesWorkflowOptions | None = None,
) -> None:
    """The raw traces experiment workflow.

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
        states:
            The qubit states for which to acquire the raw traces. Must be a
            list of strings containing g, e or f.
        options:
            The options for building the workflow.
            In addition to options from [WorkflowOptions], the following
            custom options are supported:
                - create_experiment: The options for creating the experiment.

    Returns:
        result:
            A builder for the raw traces experiment.
    """
    qubits = validate_and_convert_qubits_sweeps(qubits)
    with workflow.for_(states) as state:
        with workflow.for_(qubits) as qubit:
            exp = create_experiment(qpu, qubit, state)
            compiled_exp = compile_experiment(session, exp)
            _ = run_experiment(session, compiled_exp)


@workflow.task
@qubit_experiment
def create_experiment(
    qpu: QPU,
    qubit: QuantumElement,
    state: Literal["g", "e", "f"],
    options: TimeTracesExperimentOptions | None = None,
) -> Experiment:
    """Creates a raw trace measurement.

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubit:
            The qubit to run the experiments on.
        state:
            The qubit state for which to acquire the raw trace. Must be 'g', 'e' or 'f'.
        options:
            The options for building the experiment.
            See [TimeTracesExperimentOptions] for
            accepted options.
            Overwrites the options from [BaseExperimentOptions].

    Returns:
        result:
            The result of the workflow.

    Raises:
        ValueError:
            If one the state is not 'g', 'e' or 'f'.
    """
    if state not in ("g", "e", "f"):
        raise ValueError(
            f"The given state '{state}' is not valid. Only g, e and f states are "
            "supported."
        )
    # Define the custom options for the experiment
    opts = TimeTracesExperimentOptions() if options is None else options
    if opts.acquisition_type not in [AcquisitionType.RAW, "raw"]:
        raise ValueError(
            "The only allowed acquisition_type for this experiment is "
            "AcquisitionType.RAW (or 'raw') because the "
            "experiment acquires raw traces."
        )

    qop = qpu.quantum_operations
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type.RAW,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        if state in ("e", "f"):
            qop.x180(qubit, transition="ge")
        if state == "f":
            qop.x180(qubit, transition="ef")
        qop.measure(qubit, dsl.handles.result_handle(qubit.uid, state))
        qop.passive_reset(qubit)
