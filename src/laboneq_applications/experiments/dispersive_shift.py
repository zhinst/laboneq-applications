"""This module defines the dispersive shift experiment.

In this experiment, we prepare the qubit at defined states and then perform resonator
spectroscopy to characterize the dispersive shift. Note that the qubits states to be
prepared are defined in 'cal_states' of the option.

The dispersive shift experiment has the following pulse sequence:

    qb --- [ prep state ] --- [ measure (swept frequency) ]

This experiment only supports 1 qubit at the time, and involves only
its coupled resonator.

The Acquisition mode is restricted to SPECTROSCOPY mode, utilizing Hardware modulation.
Once LRT option becomes more generally available, this example can be improved to
support a multiplexed version for multiple qubits.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.dsl.enums import AcquisitionType
from laboneq.simple import Experiment, SweepParameter

from laboneq_applications import dsl
from laboneq_applications.experiments.options import (
    SpectroscopyExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks import compile_experiment, run_experiment
from laboneq_applications.workflow import task, workflow

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.dsl.session import Session
    from numpy.typing import ArrayLike

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import QubitSweepPoints


@workflow
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit: QuantumElement,
    frequencies: QubitSweepPoints,
    states: Sequence[str],
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Dispersive Shift Workflow.

    The workflow consists of the following steps:

    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubit:
            The qubit to run the experiments on. It can be only a single qubit
            coupled to a resonator.
        frequencies:
            The resonator frequencies to sweep over for the readout pulse (or CW)
            sent to the resonator. Must be a list of numbers or an array.
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
        options = SpectroscopyExperimentOptions()
        options.create_experiment.count = 10
        options.create_experiment.acquisition_type = AcquisitionType.SPECTROSCOPY
        qpu = QPU(
            setup=DeviceSetup("my_device"),
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            qop=SpectroscopyExperimentOptions(),
        )
        temp_qubits = qpu.copy_qubits()
        result = run(
            session=session,
            qpu=qpu,
            qubits=temp_qubits[0],
            frequencies=np.linspace(1.8e9, 2.2e9, 101),
            states="ge"
            options=options,
        )
        ```
    """
    exp = create_experiment(
        qpu,
        qubit,
        frequencies=frequencies,
        states=states,
    )
    compiled_exp = compile_experiment(session, exp)
    _result = run_experiment(session, compiled_exp)


@task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubit: QuantumElement,
    frequencies: ArrayLike,
    states: Sequence[str],
    options: SpectroscopyExperimentOptions | None = None,
) -> Experiment:
    """Creates a Dispersive Shift Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubit:
            The qubit to run the experiments on. It can be only a single qubit
            coupled to a resonator.
        frequencies:
            The resonator frequencies to sweep over for the readout pulse (or CW)
            sent to the resonator. Must be a list of numbers or an array.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].
        options:
            The options for building the experiment.
            See [SpectroscopyExperimentOptions] and [BaseExperimentOptions] for
            accepted options.
            Overwrites the options from [SpectroscopyExperimentOptions] and
            [BaseExperimentOptions].

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Raises:
        ValueError:
            If Acquisition type is not set to SPECTROSCOPY.

        ValueError:
            If frequencies is not a list of lists of numbers.

    Example:
        ```python
        options = {
            "count": 10,
            "acquisition_type": "spectroscopy",
        }
        options = TuneupExperimentOptions(**options)
        setup = DeviceSetup()
        qpu = QPU(
            setup=DeviceSetup("my_device"),
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            qop=SpectroscopyExperimentOptions(),
        )
        temp_qubits = qpu.copy_qubits()
        create_experiment(
            qpu=qpu,
            qubits=temp_qubits[0],
            frequencies=np.linspace(1.8e9, 2.2e9, 101),
            states="ge"
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = SpectroscopyExperimentOptions() if options is None else options
    if AcquisitionType(opts.acquisition_type) != AcquisitionType.SPECTROSCOPY:
        raise ValueError(
            "The only allowed acquisition_type for this experiment"
            f"is 'AcquisitionType.SPECTROSCOPY' (or {AcquisitionType.SPECTROSCOPY})"
            "because it sweeps frequency of a hardware oscillator.",
        )

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        with dsl.sweep(
            name=f"freqs_{qubit.uid}",
            parameter=SweepParameter(f"frequency_{qubit.uid}", frequencies),
        ) as frequency:
            qpu.qop.set_frequency(qubit, frequency=frequency, readout=True)
            for state in states:
                qpu.qop.prepare_state(qubit, state)
                qpu.qop.measure(
                    qubit,
                    dsl.handles.result_handle(qubit.uid, state),
                )
                qpu.qop.passive_reset(qubit)
