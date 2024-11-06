# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the raw-traces measurement.

In this measurement, raw traces are acquired for qubits in different states in order to
compute the optimal integration kernels for qubit readout, which allow to maximally
distinguish between the qubit states (typically, g, e, f).

The raw-traces measurement has the following pulse sequence

    qb --- [prepare state] --- [measure]

The corresponding traces are acquired for all combinations of the qubits and states
given by the user.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from laboneq import workflow
from laboneq.simple import AcquisitionType, Experiment, dsl
from laboneq.workflow.tasks import (
    append_result,
    combine_results,
    compile_experiment,
    run_experiment,
)

from laboneq_applications.analysis.time_traces import analysis_workflow
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import (
    BaseExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks.parameter_updating import (
    temporary_modify,
    update_qubits,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.dsl.quantum import TransmonParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import Qubits


@workflow.options
class TimeTracesExperimentOptions(BaseExperimentOptions):
    """Options for the time-traces experiment.

    This class is needed to change the default value of acquisition_type compared with
    the value in BaseExperimentOptions.

    Attributes:
        acquisition_type:
            Acquisition type to use for the experiment.
            Default: `AcquisitionType.RAW`.
    """

    acquisition_type: str | AcquisitionType = workflow.option_field(
        AcquisitionType.RAW, description="Acquisition type to use for the experiment"
    )


@workflow.workflow(name="time_traces")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    states: Sequence[Literal["g", "e", "f"]],
    temporary_parameters: dict[str, dict | TransmonParameters] | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The raw-traces experiment workflow.

    The workflow consists of the following tasks:

    - [validate_and_convert_qubits_sweeps]()
    - `with for_(qubits) as qubit`:
        - [create_experiment]()
        - [compile_experiment]()
        - [run_experiment]()
        - [append_result]()
    - [combine_results]()
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
        states:
            The qubit states for which to acquire the raw traces. Must be a
            list of strings containing g, e or f.
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
        options.create_experiment.count(10)
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubit=temp_qubits[0],
            states="gef",
            options=options,
        ).run()
        ```
    """
    qubits = temporary_modify(qubits, temporary_parameters)
    qubits = validation.validate_and_convert_qubits_sweeps(qubits)
    results = []
    with workflow.for_(qubits, lambda q: q.uid) as qubit:
        with workflow.for_(states, lambda s: s) as state:
            exp = create_experiment(qpu, qubit, state)
            compiled_exp = compile_experiment(session, exp)
            # Below, we disable saving for the inputs and outputs of the run_experiment
            # task. The interface for doing this will be cleaned up later.
            result = workflow.task(run_experiment, save=False)(session, compiled_exp)
            append_result(results, result)
    combined_results = combine_results(results)
    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(combined_results, qubits, states)
        qubit_parameters = analysis_results.output
        with workflow.if_(options.update):
            update_qubits(qpu, qubit_parameters["new_parameter_values"])
    workflow.return_(combined_results)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubit: QuantumElement,
    state: Literal["g", "e", "f"],
    options: TimeTracesExperimentOptions | None = None,
) -> Experiment:
    """Creates a raw-traces Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubit:
            The qubit for which to create the Experiment.
        state:
            The state in which to prepare the qubit. Must be 'g', 'e' or 'f'.
        options:
            The options for building the experiment as an instance of
            [TimeTracesExperimentOptions]. See the docstring of this class for more
            details.

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Raises:
        ValueError:
            If the state is not 'g', 'e' or 'f'.

        ValueError:
            If options.acquisition_type is not AcquisitionType.RAW or "raw".

    Example:
        ```python
        options = TimeTracesExperimentOptions()
        options.count(10)
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        create_experiment(
            qpu=qpu,
            qubits=temp_qubits,
            states="gef"
            options=options,
        )
        ```
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
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        if state in ("e", "f"):
            qop.x180(qubit, transition="ge")
        if state == "f":
            qop.x180(qubit, transition="ef")
        qop.measure(qubit, dsl.handles.result_handle(qubit.uid, suffix=state))
        qop.passive_reset(qubit)
