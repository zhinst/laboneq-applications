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

from laboneq import workflow
from laboneq.dsl.enums import AcquisitionType
from laboneq.simple import Experiment, SweepParameter
from laboneq.workflow import option_field, options

from laboneq_applications import dsl
from laboneq_applications.analysis.dispersive_shift import analysis_workflow
from laboneq_applications.experiments.options import (
    BaseExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks import compile_experiment, run_experiment, update_qubits
from laboneq_applications.tasks.parameter_updating import temporary_modify

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.dsl.quantum import (
        TransmonParameters,
    )
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.dsl.session import Session
    from numpy.typing import ArrayLike

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import QubitSweepPoints


@options
class DispersiveShiftExperimentOptions(BaseExperimentOptions):
    """Options for the dispersive-shift experiment.

    This class is needed only to change the default value of acquisition_type compared
    to the one in BaseExperimentOptions.

    Attributes:
        acquisition_type:
            Acquisition type to use for the experiment.
            Default: `AcquisitionType.SPECTROSCOPY`.
    """

    acquisition_type: AcquisitionType = option_field(
        AcquisitionType.SPECTROSCOPY, description="Acquisition type to use."
    )


@workflow.workflow(name="dispersive_shift")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit: QuantumElement,
    frequencies: QubitSweepPoints,
    states: Sequence[str],
    temporary_parameters: dict[str, dict | TransmonParameters] | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Dispersive Shift Workflow.

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
            The qubit to run the experiments on. It can be only a single qubit
            coupled to a resonator.
        frequencies:
            The resonator frequencies to sweep over for the readout pulse (or CW)
            sent to the resonator. Must be a list of numbers or an array.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].
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
        options.count(10)
        options.acquisition_type(AcquisitionType.SPECTROSCOPY)
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=SpectroscopyExperimentOptions(),
        )
        temp_qubits = qpu.copy_qubits()
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=temp_qubits[0],
            frequencies=np.linspace(1.8e9, 2.2e9, 101),
            states="ge"
            options=options,
        ).run()
        ```
    """
    qubit = temporary_modify(qubit, temporary_parameters)
    exp = create_experiment(
        qpu,
        qubit,
        frequencies=frequencies,
        states=states,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(result, qubit, frequencies, states)
        qubit_parameters = analysis_results.output
        with workflow.if_(options.update):
            update_qubits(qpu, qubit_parameters["new_parameter_values"])
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubit: QuantumElement,
    frequencies: ArrayLike,
    states: Sequence[str],
    options: DispersiveShiftExperimentOptions | None = None,
) -> Experiment:
    """Creates a Dispersive Shift Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubit:
            The qubit to run the experiments on. It can be only a single qubit
            coupled to a resonator.
        frequencies:
            The resonator frequencies to sweep over for the readout pulse
            sent to the resonator. Must be a list of numbers or an array.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].
        options:
            The options for building the experiment as an instance of
            [BaseExperimentOptions]. See docstring of this class for more details.

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
        options = ResonatorSpectroscopyExperimentOptions()
        options.count(10)
        options.acquisition_type(AcquisitionType.SPECTROSCOPY)
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
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
    opts = DispersiveShiftExperimentOptions() if options is None else options
    qubit, frequencies = dsl.validation.validate_and_convert_single_qubit_sweeps(
        qubit, frequencies
    )
    if AcquisitionType(opts.acquisition_type) != AcquisitionType.SPECTROSCOPY:
        raise ValueError(
            "The only allowed acquisition_type for this experiment"
            f"is 'AcquisitionType.SPECTROSCOPY' (or {AcquisitionType.SPECTROSCOPY})"
            "because it sweeps frequency of a hardware oscillator.",
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
        with dsl.sweep(
            name=f"frequency_sweep_{qubit.uid}",
            parameter=SweepParameter(f"frequency_{qubit.uid}", frequencies),
        ) as frequency:
            qop.set_frequency(qubit, frequency=frequency, readout=True)
            for state in states:
                qop.prepare_state(qubit, state)
                qop.measure(
                    qubit,
                    dsl.handles.result_handle(qubit.uid, suffix=state),
                )
                qop.passive_reset(qubit)
