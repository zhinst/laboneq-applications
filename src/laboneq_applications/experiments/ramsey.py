"""This module defines the Ramsey experiment.

In this experiment, we sweep the wait time between two x90 pulses on a given qubit
transition in order to determine the T2 time of the qubit.

The Ramsey experiment has the following pulse sequence:

    qb --- [ prep transition ] --- [ x90_transition ] --- [ delay ] ---
    [ x90_transition ] --- [ measure ]

The second x90 pulse has a delay dependent phase that generates an oscillation of the
qubit population at the frequency `detuning`.

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from laboneq.simple import Experiment, SectionAlignment, SweepParameter

from laboneq_applications.core import handles
from laboneq_applications.core.build_experiment import qubit_experiment
from laboneq_applications.core.options import (
    TuneupExperimentOptions,
)
from laboneq_applications.core.quantum_operations import dsl
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.tasks import compile_experiment, run_experiment
from laboneq_applications.workflow import TuneUpWorkflowOptions, task, workflow

if TYPE_CHECKING:
    from laboneq.dsl.session import Session

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import Qubits, QubitSweepPoints
    from laboneq_applications.workflow.engine.core import WorkflowBuilder


options = TuneUpWorkflowOptions


@workflow
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    delays: QubitSweepPoints,
    detunings: dict[str, float] | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> WorkflowBuilder:
    """The Ramsey Workflow.

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
        delays:
            The delays (in seconds) of the second x90 pulse to sweep over for each
            qubit. If `qubits` is a single qubit, `delays` must be a list of numbers
            or an array. Otherwise it must be a list of lists of numbers or arrays.
        detunings:
            The detunings in Hz to generate oscillating qubit occupations. `detunings`
            is a dictionary of float values for different qubits.
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
        options = TuneUpWorkflowOptions()
        options.create_experiment.count = 10
        options.create_experiment.transition = "ge"
        qpu = QPU(
            setup=DeviceSetup("my_device"),
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            qop=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=temp_qubits,
            delays=[[0.1, 0.5, 1], [0.1, 0.5, 1]],
            detunings = {'q0':1e6,'q1':1.346e6},
            options=options,
        ).run()
        ```
    """
    exp = create_experiment(
        qpu,
        qubits,
        delays=delays,
        detunings=detunings,
    )
    compiled_exp = compile_experiment(session, exp)
    _result = run_experiment(session, compiled_exp)


@task
@qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: Qubits,
    delays: QubitSweepPoints,
    detunings: dict[str, float] | None = None,
    options: TuneupExperimentOptions | None = None,
) -> Experiment:
    """Creates a Ramsey Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        delays:
            The delays (in seconds) of the second x90 pulse to sweep over for each
            qubit. If `qubits` is a single qubit, `delays` must be a list of numbers
            or an array. Otherwise it must be a list of lists of numbers or arrays.
        detunings:
            The detuning in Hz to generate oscillating qubit occupations. `detunings`
            is a dictionary of float values for different qubits.
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
            If the lengths of `qubits` and `delays` do not match.

        ValueError:
            If `delays` is not a list of numbers when a single qubit is passed.

        ValueError:
            If `delays` is not a list of lists of numbers when a list of qubits
            is passed.

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
        setup = DeviceSetup()
        qpu = QPU(
            setup=DeviceSetup("my_device"),
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            qop=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        create_experiment(
            qpu=qpu,
            qubits=temp_qubits,
            delays=[[0.1, 0.5, 1], [0.1, 0.5, 1]],
            detunings = {'q0':1e6,'q1':1.346e6},
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = TuneupExperimentOptions() if options is None else options
    qubits, delays = validate_and_convert_qubits_sweeps(qubits, delays)

    if detunings is None:
        detunings = {qb.uid: 0 for qb in qubits}

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        swp_delays = []
        swp_phases = []
        for q, q_delays in zip(qubits, delays):
            swp_delays += [
                SweepParameter(
                    uid=f"wait_time_{q.uid}",
                    values=q_delays,
                ),
            ]
            swp_phases += [
                SweepParameter(
                    uid=f"x90_phases_{q.uid}",
                    values=[
                        ((wait_time - q_delays[0]) * detunings[q.uid] * 2 * np.pi)
                        % (2 * np.pi)
                        for wait_time in q_delays
                    ],
                ),
            ]

        with dsl.sweep(
            name="sweep_delays_phases",
            parameter=swp_delays + swp_phases,
            alignment = SectionAlignment.RIGHT,
        ):
            for q, wait_time, phase in zip(qubits, swp_delays, swp_phases):
                qpu.qop.prepare_state(q, opts.transition[0])
                qpu.qop.ramsey(q, wait_time, phase, transition=opts.transition)
                qpu.qop.measure(q, handles.result_handle(q.uid))
                qpu.qop.passive_reset(q)

        if opts.use_cal_traces:
            for q in qubits:
                with dsl.section(
                    name=f"cal_{q.uid}",
                ):
                    for state in opts.cal_states:
                        qpu.qop.prepare_state(q, state)
                        qpu.qop.measure(
                            q,
                            handles.calibration_trace_handle(q.uid, state),
                        )
                        qpu.qop.passive_reset(q)
