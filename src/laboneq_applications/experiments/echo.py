"""This module defines the Hahn echo experiment.

In the Hahn echo experiment, we perform a Ramsey experiment and place one extra y180
pulse between the two x90 pulses. Due to the additional y180 pulse, the quasi-static
contributions to dephasing can be “refocused” and by that the experiment is less
sensitive to quasi-static noise.
The pulses are generally chosen to be resonant with the qubit transition for a
Hahn echo, since any frequency detuning would be nominally refocused anyway.

The Hahn echo experiment has the following pulse sequence:

    qb --- [ prep transition ] --- [ x90_transition ] --- [ delay/2 ] ---
    [ y180_transition ] --- [ delay/2 ] --- [ x90_transition ] --- [ measure ]

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.simple import Experiment, SweepParameter

from laboneq_applications.core import handles
from laboneq_applications.core.build_experiment import qubit_experiment
from laboneq_applications.core.options import (
    TuneupExperimentOptions,
)
from laboneq_applications.core.quantum_operations import dsl
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
)
from laboneq_applications.tasks import compile_experiment, run_experiment
from laboneq_applications.workflow import WorkflowOptions, task, workflow

if TYPE_CHECKING:
    from laboneq.dsl.session import Session

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import Qubits, QubitSweepPoints
    from laboneq_applications.workflow.engine.core import WorkflowBuilder


class EchoExperimentOptions(TuneupExperimentOptions):
    """Base options for the resonator spectroscopy experiment.

    Additional attributes:
        refocus_pulse:
            String to define the quantum operation in-betweeen the x90 pulses.
            Default: "y180".
    """

    refocus_qop: str = "y180"


class EchoWorkflowOptions(WorkflowOptions):
    """Option for spectroscopy workflow.

    Attributes:
        create_experiment (EchoExperimentOptions):
            The options for creating the experiment.
    """

    create_experiment: EchoExperimentOptions = EchoExperimentOptions()


options = EchoWorkflowOptions


@workflow
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    delays: QubitSweepPoints,
    options: EchoWorkflowOptions | None = None,
) -> WorkflowBuilder:
    """The Hahn echo Workflow.

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
            The delays to sweep over for each qubit. Note that `delays` must be
            identical for qubits that use the same measure port.
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
        options = EchoWorkflowOptions()
        options.create_experiment.count = 10
        options.create_experiment.transition = "ge"
        qpu = QPU(
            setup=DeviceSetup("my_device"),
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            qop=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        result = run(
            session=session,
            qpu=qpu,
            qubits=temp_qubits,
            delays=[[1e-6, 5e-6, 10e-6]], [1e-6, 5e-6, 10e-6]],
            options=options,
        )
        ```
    """
    exp = create_experiment(
        qpu,
        qubits,
        delays=delays,
    )
    compiled_exp = compile_experiment(session, exp)
    _result = run_experiment(session, compiled_exp)


@task
@qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: Qubits,
    delays: QubitSweepPoints,
    options: EchoExperimentOptions | None = None,
) -> Experiment:
    """Creates a Hahn echo Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        delays:
            The delays to sweep over for each qubit. Note that `delays` must be
            identical for qubits that use the same measure port.
        options:
            The options for building the experiment.
            See [EchoExperimentOptions] and [BaseExperimentOptions] for
            accepted options.
            Overwrites the options from [TuneupExperimentOptions] and
            [BaseExperimentOptions].

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Raises:
        ValueError:
            If delays is not a list of numbers or array when a single qubit is passed.

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
            delays=[[1e-6, 5e-6, 10e-6], [1e-6, 5e-6, 10e-6]]
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = EchoExperimentOptions() if options is None else options

    qubits, delays = validate_and_convert_qubits_sweeps(qubits, delays)

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for q, q_delays in zip(qubits, delays):
            with dsl.sweep(
                name=f"delays_{q.uid}",
                parameter=SweepParameter(f"delay_{q.uid}", q_delays),
            ) as delay:
                qpu.qop.prepare_state(q, opts.transition[0])
                qpu.qop.ramsey(
                    q, delay, 0, echo_pulse=opts.refocus_qop, transition=opts.transition
                )
                qpu.qop.measure(q, handles.result_handle(q.uid))
                qpu.qop.passive_reset(q)
            if opts.use_cal_traces:
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
