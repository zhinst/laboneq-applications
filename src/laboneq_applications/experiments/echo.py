"""This module defines the Hahn echo experiment.

In the Hahn echo experiment, we perform a Ramsey experiment and place one extra
refocusing pulse, typically y180, between the two x90 pulses. Due to this additional
pulse, the quasi-static contributions to dephasing can be “refocused” and by that the
experiment is less sensitive to quasi-static noise.

The pulses are generally chosen to be resonant with the qubit transition for a
Hahn echo, since any frequency detuning would be nominally refocused anyway.

The Hahn echo experiment has the following pulse sequence:

    qb --- [ prep transition ] --- [ x90_transition ] --- [ delay/2 ] ---
    [ refocusing pulse ] --- [ delay/2 ] --- [ x90_transition ] --- [ measure ]

If multiple qubits are passed to the experiment workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.simple import AveragingMode, Experiment, SweepParameter

from laboneq_applications import dsl, workflow
from laboneq_applications.analysis.echo import analysis_workflow
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks import compile_experiment, run_experiment, update_qubits
from laboneq_applications.tasks.parameter_updating import temporary_modify
from laboneq_applications.workflow import option_field, options

if TYPE_CHECKING:
    from laboneq.dsl.quantum import (
        TransmonParameters,
    )
    from laboneq.dsl.session import Session

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import Qubits, QubitSweepPoints


@options
class EchoExperimentOptions(TuneupExperimentOptions):
    """Options for the Hahn echo experiment.

    Additional attributes:
        refocus_pulse:
            String to define the quantum operation in-between the x90 pulses.
            Default: "y180".
    """

    refocus_qop: str = option_field(
        "y180",
        description="String to define the quantum operation in-between the x90 pulses",
    )


@workflow.workflow(name="echo")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    delays: QubitSweepPoints,
    temporary_parameters: dict[str, dict | TransmonParameters] | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Hahn echo experiment workflow.

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
            The qubits on which to run the experiments. May be either a single
            qubit or a list of qubits.
        delays:
            The delays to sweep over for each qubit. The delays between the two x90
            pulses and the refocusing pulse are `delays / 2`; see the schematic of
            the pulse sequence at the top of the file. Note that `delays` must be
            identical for qubits that use the same measure port.
        temporary_parameters:
            The temporary parameters to update the qubits with.
        options:
            The options for building the workflow as an instance of
            [TuneUpWorkflowOptions]. See the docstrings of this class for more details.

    Returns:
        WorkflowBuilder:
            The builder for the experiment workflow.

    Example:
        ```python
        options = EchoWorkflowOptions()
        options.count(10)
        options.transition("ge")
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        result = run(
            session=session,
            qpu=qpu,
            qubits=temp_qubits,
            delays=[np.linspace(0, 30e-6, 51), np.linspace(0, 30e-6, 51)],
            options=options,
        )
        ```
    """
    qubits = temporary_modify(qubits, temporary_parameters)
    exp = create_experiment(
        qpu,
        qubits,
        delays=delays,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(result, qubits, delays)
        qubit_parameters = analysis_results.output
        with workflow.if_(options.update):
            update_qubits(qpu, qubit_parameters["new_parameter_values"])
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
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
            The qubits on which to run the experiments. May be either a single
            qubit or a list of qubits.
        delays:
            The delays to sweep over for each qubit. The delays between the two x90
            pulses and the refocusing pulse are `delays / 2`; see the schematic of
            the pulse sequence at the top of the file. Note that `delays` must be
            identical for qubits that use the same measure port.
        options:
            The options for building the workflow as an instance of
            [EchoExperimentOptions], inheriting from [TuneupExperimentOptions].
            See the docstrings of these classes for more details.

    Returns:
        Experiment:
            The generated LabOne Q Experiment instance to be compiled and executed.

    Raises:
        ValueError:
            If the conditions in validation.validate_and_convert_qubits_sweeps are not
            fulfilled.

        ValueError:
            If the experiment uses calibration traces and the averaging mode is
            sequential.

    Example:
        ```python
        options = TuneupExperimentOptions()
        options.count = 10
        options.cal_traces = True
        setup = DeviceSetup()
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        create_experiment(
            qpu=qpu,
            qubits=temp_qubits,
            delays=[np.linspace(0, 30e-6, 51), np.linspace(0, 30e-6, 51)],
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = EchoExperimentOptions() if options is None else options

    qubits, delays = dsl.validation.validate_and_convert_qubits_sweeps(qubits, delays)
    if (
        opts.use_cal_traces
        and AveragingMode(opts.averaging_mode) == AveragingMode.SEQUENTIAL
    ):
        raise ValueError(
            "'AveragingMode.SEQUENTIAL' (or {AveragingMode.SEQUENTIAL}) cannot be used "
            "with calibration traces because the calibration traces are added "
            "outside the sweep."
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
        for q, q_delays in zip(qubits, delays):
            with dsl.sweep(
                name=f"delays_{q.uid}",
                parameter=SweepParameter(f"delay_{q.uid}", q_delays),
            ) as delay:
                qop.prepare_state(q, opts.transition[0])
                qop.ramsey(
                    q, delay, 0, echo_pulse=opts.refocus_qop, transition=opts.transition
                )
                qop.measure(q, dsl.handles.result_handle(q.uid))
                qop.passive_reset(q)
            if opts.use_cal_traces:
                qop.calibration_traces(q, states=opts.cal_states)
