"""This module defines the signal propagation delay experiment.

In this experiment, we sweep the readout integration delay
of a measurement to optimize the time delay between measure and acquire.

The signal propagation delay experiment has the following pulse sequence:

    [ measure ] --- [ readout integration delay ] --- [acquire]

This experiment only supports 1 qubit at the time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.simple import Experiment, SweepParameter, dsl
from laboneq.workflow import if_, task, workflow
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.contrib.analysis.signal_propagation_delay import (
    analysis_workflow,
)
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks.parameter_updating import update_qubits

if TYPE_CHECKING:
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import Qubit, QubitSweepPoints


@workflow
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit: Qubit,
    delays: QubitSweepPoints,
    measure_delay: float | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Signal Propagation Delay Workflow.

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
        qubit:
            The qubit to run the experiment on. It can be only a single qubit
            coupled to a resonator.
        delays:
            The readout integration delays to sweep over for the readout pulse.
            Must be a list of numbers or an array.
        measure_delay:
            Delay between subsequent measurements.
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
            qubit=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        result = run(
            session=session,
            qpu=qpu,
            qubit=temp_qubits[0],
            delays=np.linspace(0e-9, 100e-9, 51),
            options=options,
        )
        ```
    """
    exp = create_experiment(
        qpu,
        qubit,
        delays=delays,
        measure_delay=measure_delay,
    )
    compiled_exp = compile_experiment(session, exp)
    _result = run_experiment(session, compiled_exp)
    with if_(options.do_analysis):
        analysis_results = analysis_workflow(_result, qubit, delays)
        qubit_parameters = analysis_results.tasks["extract_qubit_parameters"].output
        with if_(options.update):
            update_qubits(qpu, qubit_parameters["new_parameter_values"])


@task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubit: Qubit,
    delays: QubitSweepPoints,
    measure_delay: float | None = None,
    options: TuneupExperimentOptions | None = None,
) -> Experiment:
    """Creates an Signal Propagation Delay Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubit:
            The qubit to run the experiments on.
        delays:
            The readout integration delays to sweep over for the readout pulse.
            Must be a list of numbers or an array.
        measure_delay:
            Delay between subsequent measurements.
        options:
            The options for building the experiment.
            See [TuneupExperimentOptions] and [BaseExperimentOptions] for
            accepted options.
            Overwrites the options from [TuneupExperimentOptions] and
            [BaseExperimentOptions].

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Example:
        ```python
        options = {
            "count": 10,
        }
        options = TuneupExperimentOptions(**options)
        setup = DeviceSetup()
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        create_experiment(
            qpu=qpu,
            qubit=temp_qubits[0],
            delays=np.linspace(0e9, 100e9, 51),
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = TuneupExperimentOptions() if options is None else options
    qop = qpu.quantum_operations
    if measure_delay is None:
        measure_delay = 1e-6
    calibration = dsl.experiment_calibration()
    signal_calibration = calibration[qubit.signals["acquire"]]
    with dsl.sweep(
        name=f"freq_{qubit.uid}",
        parameter=SweepParameter(f"port_delays_{qubit.uid}", delays),
    ) as delay:
        with dsl.acquire_loop_rt(
            count=opts.count,
            averaging_mode=opts.averaging_mode,
            acquisition_type=opts.acquisition_type,
            repetition_mode=opts.repetition_mode,
            repetition_time=opts.repetition_time,
            reset_oscillator_phase=opts.reset_oscillator_phase,
        ):
            signal_calibration.port_delay = delay
            qop.measure(qubit, dsl.handles.result_handle(qubit.uid))
            qop.delay(qubit, measure_delay)
