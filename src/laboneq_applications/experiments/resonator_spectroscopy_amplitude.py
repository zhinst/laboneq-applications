"""This module defines the resonator spectroscopy amplitude sweep experiment.

In this experiment, we sweep the resonator frequency and the amplitude
of a measure pulse in a 2D fashion to characterize the resonator coupled to the qubit.

The resonator spectroscopy amplitude sweep experiment has the following pulse sequence:

    qb --- [ measure ]
         sweep amplitude

This experiment only supports 1 qubit at the time, and involves only
its coupled resonator
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.dsl.enums import AcquisitionType
from laboneq.simple import Experiment, SweepParameter

from laboneq_applications import dsl
from laboneq_applications.experiments.options import (
    ResonatorSpectroscopyExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks import compile_experiment, run_experiment
from laboneq_applications.tasks.parameter_updating import temporary_modify

if TYPE_CHECKING:
    from laboneq.dsl.quantum import (
        TransmonParameters,
    )
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.dsl.session import Session
    from numpy.typing import ArrayLike

    from laboneq_applications.qpu_types import QPU


@workflow.workflow(name="resonator_spectroscopy_amplitude")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit: QuantumElement,
    frequencies: ArrayLike,
    amplitudes: ArrayLike,
    temporary_parameters: dict[str, dict | TransmonParameters] | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Workflow for a resonator spectroscopy with a readout-amplitude sweep.

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
        temporary_parameters:
            The temporary parameters to update the qubits with.
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
            quantum_operations=TunableTransmonOperations(),
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
    qubit = temporary_modify(qubit, temporary_parameters)
    exp = create_experiment(
        qpu,
        qubit,
        frequencies=frequencies,
        amplitudes=amplitudes,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubit: QuantumElement,
    frequencies: ArrayLike,
    amplitudes: ArrayLike,
    options: ResonatorSpectroscopyExperimentOptions | None = None,
) -> Experiment:
    """A Resonator Spectroscopy where the measure-pulse amplitude is also swept.

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
            The amplitudes to sweep over for each resonator.
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
            quantum_operations=TunableTransmonOperations(),
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
    # Define the custom options for the experiment
    opts = ResonatorSpectroscopyExperimentOptions() if options is None else options
    qubit, frequencies = dsl.validation.validate_and_convert_single_qubit_sweeps(
        qubit, frequencies
    )
    # guard against wrong options for the acquisition type
    if AcquisitionType(opts.acquisition_type) != AcquisitionType.SPECTROSCOPY:
        raise ValueError(
            "The only allowed acquisition_type for this experiment"
            "is 'AcquisitionType.SPECTROSCOPY' (or 'spectrsocopy')"
            "because it contains a sweep"
            "of the frequency of a hardware oscillator.",
        )

    qop = qpu.quantum_operations
    with dsl.sweep(
        parameter=SweepParameter(f"amplitudes_{qubit.uid}", amplitudes),
    ) as amplitude:
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
                qop.set_readout_amplitude(qubit, amplitude=amplitude)
                qop.set_frequency(qubit, frequency=frequency, readout=True)
                if opts.use_cw:
                    qop.acquire(qubit, dsl.handles.result_handle(qubit.uid))
                else:
                    qop.measure(qubit, dsl.handles.result_handle(qubit.uid))
                qop.delay(qubit, opts.spectroscopy_reset_delay)
