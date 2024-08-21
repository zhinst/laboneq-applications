"""This module defines the resonator spectroscopy experiment.

In this experiment, we sweep the resonator frequency and optionally the amplitude
of a measure pulse to characterize the resonator coupled to the qubit.

The resonator spectroscopy experiment has the following pulse sequence:

    qb --- [ measure ]

This experiment only supports 1 qubit at the time, and involves only
its coupled resonator.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

from laboneq.dsl.enums import AcquisitionType
from laboneq.simple import Experiment, SweepParameter

from laboneq_applications.core import handles
from laboneq_applications.core.build_experiment import qubit_experiment
from laboneq_applications.core.options import (
    BaseExperimentOptions,
)
from laboneq_applications.core.quantum_operations import dsl
from laboneq_applications.tasks import compile_experiment, run_experiment
from laboneq_applications.workflow import (
    WorkflowOptions,
    task,
    workflow,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.dsl.session import Session
    from numpy.typing import ArrayLike

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.workflow.engine.core import WorkflowBuilder


class SpectroscopyExperimentOptions(BaseExperimentOptions):
    """Base options for the resonator spectroscopy experiment.

    Additional attributes:
        use_cw:
            If true, perform a CW spectroscopy.
            No pulse is emitted on measure.
            Default: False.
        spectroscopy_reset_delay:
            How long to wait after an acquisition in seconds.
            Default: 1e-6.
        acquisition_type:
            Acquisition type to use for the experiment.
            Default: `AcquisitionType.SPECTROSCOPY`.
    """

    use_cw: bool = False
    spectroscopy_reset_delay: float = 1e-6
    acquisition_type: AcquisitionType = AcquisitionType.SPECTROSCOPY


class SpectroscopyWorkflowOptions(WorkflowOptions):
    """Option for spectroscopy workflow.

    Attributes:
        create_experiment (SpectroscopyExperimentOptions):
            The options for creating the experiment.
    """

    create_experiment: SpectroscopyExperimentOptions = SpectroscopyExperimentOptions()


options = SpectroscopyWorkflowOptions


@workflow
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit: QuantumElement,
    frequencies: ArrayLike,
    amplitudes: ArrayLike,
    options: SpectroscopyWorkflowOptions | None = None,
) -> WorkflowBuilder:
    """The Resonator Spectroscopy Workflow.

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
        amplitudes:
            The amplitudes of the readout pulses to sweep over.
            Must be a list of numbers or an array.
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
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            qop=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        result = run(
            session=session,
            qpu=qpu,
            qubit=temp_qubits[0],
            frequencies=np.linspace(7.1e9, 7.6e9, 501),
            amplitudes=np.linspace(0.1, 1, 10),
            options=options,
        )
        ```
    """
    exp = create_experiment(
        qpu,
        qubit,
        frequencies=frequencies,
        amplitudes=amplitudes,
    )
    compiled_exp = compile_experiment(session, exp)
    _result = run_experiment(session, compiled_exp)


@task
@qubit_experiment
def create_experiment(
    qpu: QPU,
    qubit: QuantumElement,
    frequencies: ArrayLike,
    amplitudes: ArrayLike | None = None,
    options: SpectroscopyExperimentOptions | None = None,
) -> Experiment:
    """Creates an Resonator Spectroscopy Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubit:
            The qubit to run the experiments on. May be either a single
            qubit or a list of qubits.
        frequencies:
            The resonator frequencies to sweep over for each qubit.
            It must be a list of lists of numbers or arrays.
        amplitudes:
            The amplitudes to sweep over for each resonator. If this is left to None,
            no amplitudes sweep will be performed, and the power will exclusively be
            defined by the qubit calibration
            it must be a list of lists of numbers or arrays.
        options:
            The options for building the experiment.
            See [SpectroscopyExperimentOptions] and [BaseExperimentOptions] for
            accepted options.
            Overwrites the options from [TuneupExperimentOptions] and
            [BaseExperimentOptions].

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Raises:
        ValueError:
            If the qubit and qubit_amplitudes are not of the same length.

        ValueError:
            If qubit_amplitudes or qubit_amplitudes is not a list of numbers.

    Example:
        ```python
        options = {
            "count": 10,
            "spectroscopy_reset_delay": 3e-6
        }
        options = TuneupExperimentOptions(**options)
        setup = DeviceSetup()
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            qop=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        create_experiment(
            qpu=qpu,
            qubit=temp_qubits[0],
            frequencies=np.linspace(7.1e9, 7.6e9, 501),
            amplitudes=np.linspace(0.1, 1, 10),
            options=options,
        )
        ```
    """
    opts = SpectroscopyExperimentOptions() if options is None else options
    if AcquisitionType(opts.acquisition_type) != AcquisitionType.SPECTROSCOPY:
        raise ValueError(
            "The only allowed acquisition_type for this experiment"
            f"is 'AcquisitionType.SPECTROSCOPY' (or {AcquisitionType.SPECTROSCOPY})"
            "because it sweeps frequency of a hardware oscillator.",
        )
    amplitudes_sweep = (
        dsl.sweep(parameter=SweepParameter(f"amplitudes_{qubit.uid}", amplitudes))
        if amplitudes is not None
        else nullcontext()
    )

    with amplitudes_sweep as amplitude:
        with dsl.acquire_loop_rt(
            count=opts.count,
            averaging_mode=opts.averaging_mode,
            acquisition_type=opts.acquisition_type,
            repetition_mode=opts.repetition_mode,
            repetition_time=opts.repetition_time,
            reset_oscillator_phase=opts.reset_oscillator_phase,
        ):
            with dsl.sweep(
                name=f"freq_{qubit.uid}",
                parameter=SweepParameter(f"frequencies_{qubit.uid}", frequencies),
            ) as frequency:
                if amplitudes is not None:
                    qpu.qop.set_readout_amplitude(qubit, amplitude=amplitude)
                qpu.qop.set_frequency(qubit, frequency=frequency, readout=True)
                if opts.use_cw:
                    qpu.qop.acquire(qubit, handles.result_handle(qubit.uid))
                else:
                    qpu.qop.measure(qubit, handles.result_handle(qubit.uid))
                qpu.qop.delay(qubit, opts.spectroscopy_reset_delay)