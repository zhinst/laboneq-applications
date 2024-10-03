"""This module defines the drag beta parameter experiment.

In this experiment, we sweep the drag beta parameter of all drive
pulses on a given qubit transition. In order to determine the optimal beta for
compensating phase errors, we apply a pulse sequence that is sensitive to phase
errors.

The drag_beta_parameter experiment has the following pulse sequence:

    qb --- [ prep transition ] --- [ x90_transition ]
    --- [ y180_transition ] --- [ measure ]

    qb --- [ prep transition ] --- [ x90_transition ]
    --- [ my180_transition ] --- [ measure ]

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from laboneq.simple import Experiment, SweepParameter

from laboneq_applications import dsl
from laboneq_applications.contrib.analysis.drag_beta_parameter import analysis_workflow
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks import compile_experiment, run_experiment, update_qubits
from laboneq_applications.workflow import if_, task, workflow

if TYPE_CHECKING:
    from laboneq.dsl.session import Session

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import Qubits, QubitSweepPoints


@workflow
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    beta_values: QubitSweepPoints,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Drag Beta Parameter Workflow.

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
        beta_values:
            The beta_values to sweep over for each qubit. If `qubits` is a
            single qubit, `beta_values` must be a list of numbers or an array. Otherwise
            it must be a list of lists of numbers or arrays.
        options:
            The options for building the workflow.
            In addition to options from [WorkflowOptions], the following
            custom options are supported:
                - create_experiment: The options for creating the experiment.

    Returns:
        WorkflowBuilder:
            The builder of the experiment workflow.

    Example:
        ```python
        options = TuneUpExperimentWorkflowOptions()
        options.create_experiment.count = 10
        options.create_experiment.transition = "ge"
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            qop=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=temp_qubits,
            beta_values=[
                np.linspace(-0.05, 0.05, 11),
                np.linspace(-0.04, 0.04, 11),
            ],
            options=options,
        ).run()
        ```
    """
    exp = create_experiment(
        qpu,
        qubits,
        beta_values=beta_values,
    )
    compiled_exp = compile_experiment(session, exp)
    _result = run_experiment(session, compiled_exp)
    with if_(options.do_analysis):
        analysis_results = analysis_workflow(_result, qubits, beta_values)
        qubit_parameters = analysis_results.tasks["extract_qubit_parameters"].output
        with if_(options.update):
            update_qubits(qpu, qubit_parameters["new_parameter_values"])


@task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: Qubits,
    beta_values: QubitSweepPoints,
    options: TuneupExperimentOptions | None = None,
) -> Experiment:
    """Creates an Drag Beta Parameter experiment Workflow.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        beta_values:
            The beta_values to sweep over for each qubit. If `qubits` is a
            single qubit, `beta_values` must be a list of numbers or an array. Otherwise
            it must be a list of lists of numbers or arrays.
        options:
            The options for building the experiment.
            See [TuneupExperimentOptions] and [BaseExperimentOptions] for
            accepted options.
            Overwrites the options from [TuneupExperimentOptions] and
            [BaseExperimentOptions].

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Raises:
        ValueError:
            If the qubits and beta_values are not of the same length.

        ValueError:
            If beta_values is not a list of numbers when a single qubit is passed.

        ValueError:
            If beta_values is not a list of lists of numbers.

    Example:
        ```python
        options = {
            "count": 10,
            "transition": "ge",
            "averaging_mode": "cyclic",
            "acquisition_type": "integration_trigger",
            "cal_traces": True,
        }
        options = TuneupExperimentOptions(**options)
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            qop=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        create_experiment(
            qpu=qpu,
            qubits=temp_qubits,
            beta_values=[
                np.linspace(-0.05, 0.05, 11),
                np.linspace(-0.05, 0.05, 11),
            ],
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = TuneupExperimentOptions() if options is None else options
    qubits, beta_values = dsl.validation.validate_and_convert_qubits_sweeps(
        qubits, beta_values
    )
    pulse_ids = ["y180", "my180"]
    pulse_phase = [np.pi / 2, -np.pi / 2]
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for q, q_betas in zip(qubits, beta_values):
            with dsl.sweep(
                name=f"amps_{q.uid}",
                parameter=SweepParameter(f"beta_{q.uid}", q_betas),
            ) as beta:
                for pulse_id, phase in zip(pulse_ids, pulse_phase):
                    qpu.qop.prepare_state(q, opts.transition[0])
                    qpu.qop.x90(q, pulse={"beta": beta}, transition=opts.transition)
                    qpu.qop.y180(
                        q, pulse={"beta": beta}, phase=phase, transition=opts.transition
                    )
                    qpu.qop.measure(q, dsl.handles.result_handle(f"{q.uid}_{pulse_id}"))
                    qpu.qop.passive_reset(q)
            if opts.use_cal_traces:
                with dsl.section(
                    name=f"cal_{q.uid}",
                ):
                    for state in opts.cal_states:
                        qpu.qop.prepare_state(q, state)
                        qpu.qop.measure(
                            q,
                            dsl.handles.calibration_trace_handle(q.uid, state),
                        )
                        qpu.qop.passive_reset(q)
