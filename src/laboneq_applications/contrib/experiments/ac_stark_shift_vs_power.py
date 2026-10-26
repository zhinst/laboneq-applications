# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the AC stark shift spectroscopy experiment.

In this experiment, qubit spectroscopy is performed while the amplitude of the readout
tone is varied. The photons in the readout resonator cause a shift in the qubit
frequency (AC Stark shift). By fitting the relationship between the frequency shift
and the readout amplitude, we can determine the qubit frequency with greater precision.

The experiment has the following pulse sequence, where the qubit spectroscopy drive
and the measure pulse are played simultaneously:

    qb --- [ spect ]
       --- [ measure (swept amplitude) ]

If multiple qubits are passed to the `experiment_workflow`, the above sequence
is applied in parallel on all the qubits. Since this experiment sweeps the
amplitude of the readout line and uses its hardware oscillator, it can only be run
in parallel on qubits whose measure lines use distinct QA channels.

The Acquisition mode is restricted to SPECTROSCOPY mode on QCCS or INTEGRATION mode on
ZQCS, both utilizing Hardware modulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.simple import AcquisitionType, Experiment, SweepParameter, dsl
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.contrib.analysis import ac_stark_shift_vs_power
from laboneq_applications.conversion import (
    as_quantum_element_list,
    sweeps_per_quantum_element,
)
from laboneq_applications.experiments.options import (
    QubitSpectroscopyExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
    update_qpu,
)
from laboneq_applications.tasks.verify_qa_channels import _verify_unique_qa_channels

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


# create additional options for the AC Stark vs Power experiment
@workflow.task_options(base_class=QubitSpectroscopyExperimentOptions)
class AcStarkExperimentOptions:
    """Base options for the AC Stark shift vs power experiment.

    Attributes:
        acquisition_type:
            Acquisition type to use for the experiment. Use
            `AcquisitionType.SPECTROSCOPY` on QCCS or `AcquisitionType.INTEGRATION` on
            ZQCS.
            Default: `AcquisitionType.SPECTROSCOPY`.
    """

    acquisition_type: AcquisitionType = workflow.option_field(
        AcquisitionType.SPECTROSCOPY,
        description="Acquisition type to use for the experiment.",
    )


@workflow.workflow(name="ac_stark_vs_power")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: list[str] | str,
    frequencies: QubitSweepPoints,
    amplitudes: QubitSweepPoints,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The AC Stark Shift vs Power Workflow.

    The workflow consists of the following steps:

    - [create_experiment][]
    - [compile_experiment][]
    - [run_experiment][]
    - [analysis_workflow][]
    - [update_qpu][]

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on, passed by UID. May be either a
            single qubit or a list of qubits. The measure lines of the qubits
            must use distinct QA channels, because this experiment sweeps the
            amplitude of the readout line and uses its hardware oscillator, which
            are shared by all qubits multiplexed on the same QA channel.
        frequencies:
            The qubit frequencies to sweep over for the qubit spectroscopy drive of
            each qubit. If `qubits` is a single qubit, `frequencies` must be a list
            of numbers or an array. Otherwise it must be a list of lists of numbers
            or arrays.
        amplitudes:
            The readout amplitudes to sweep over for the measure pulse of each
            qubit. If `qubits` is a single qubit, `amplitudes` must be a list of
            numbers or an array. Otherwise it must be a list of lists of numbers
            or arrays.
        temporary_parameters:
            The temporary parameters with which to update the quantum elements and
            topology edges. For quantum elements, the dictionary key is the quantum
            element UID. For topology edges, the dictionary key is the edge tuple
            `(tag, source node UID, target node UID)`.
        options:
            The options for building the workflow as an instance of
            [TuneUpWorkflowOptions][]. See the docstring of this class for more details.

    Returns:
        WorkflowBuilder:
            The builder for the experiment workflow.

    Example:
        ```python
        options = experiment_workflow.options()
        options.count(10)
        # QPU from a single-qubit device setup
        qpu = QPU(
            quantum_elements=TunableTransmonQubit.from_device_setup(setup),
            quantum_operations=TunableTransmonOperations(),
        )
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits="q0",
            frequencies=np.linspace(6.0e9, 6.3e9, 101),
            amplitudes=np.linspace(0.2, 0.6, 21),
            options=options,
        ).run()
        ```
    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    temp_qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    _verify_unique_qa_channels(session, temp_qubits)
    exp = create_experiment(
        temp_qpu,
        temp_qubits,
        frequencies=frequencies,
        amplitudes=amplitudes,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_results = ac_stark_shift_vs_power.analysis_workflow(
            result, temp_qubits, frequencies, amplitudes
        )
        qubit_parameters = analysis_results.output
        with workflow.if_(options.update):
            update_qpu(qpu, qubit_parameters["new_parameter_values"])
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    frequencies: QubitSweepPoints,
    amplitudes: QubitSweepPoints,
    options: AcStarkExperimentOptions | None = None,
) -> Experiment:
    """Creates an AC Stark shift vs. power experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        frequencies:
            The qubit frequencies to sweep over for the qubit spectroscopy drive of
            each qubit. If `qubits` is a single qubit, `frequencies` must be a list
            of numbers or an array. Otherwise it must be a list of lists of numbers
            or arrays. The frequencies of all the qubits must contain the same
            number of sweep points.
        amplitudes:
            The readout amplitudes to sweep over for the measure pulse of each
            qubit. If `qubits` is a single qubit, `amplitudes` must be a list of
            numbers or an array. Otherwise it must be a list of lists of numbers
            or arrays. The amplitudes of all the qubits must contain the same
            number of sweep points.
        options:
            The options for building the experiment as an instance of
            [AcStarkExperimentOptions][]. See the docstring of this class for more
            details.

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Raises:
        ValueError:
            If the qubits, frequencies and amplitudes are not of the same length.

        ValueError:
            If frequencies or amplitudes is not a list of numbers or an array
            when a single qubit is passed, or a list of lists of numbers or
            arrays of the same length when multiple qubits are passed.

        ValueError:
            If the acquisition_type is neither AcquisitionType.SPECTROSCOPY
            nor AcquisitionType.INTEGRATION.


    Example:
        ```python
        options = AcStarkExperimentOptions()
        options.count = 10
        # QPU from a single-qubit device setup
        qpu = QPU(
            quantum_elements=TunableTransmonQubit.from_device_setup(setup),
            quantum_operations=TunableTransmonOperations(),
        )
        create_experiment(
            qpu=qpu,
            qubits=qpu.quantum_elements[0],
            frequencies=np.linspace(6.0e9, 6.3e9, 101),
            amplitudes=np.linspace(0.2, 0.6, 21),
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = AcStarkExperimentOptions() if options is None else options

    qubits = as_quantum_element_list(qubits)

    # Guard against wrong options for the acquisition type
    if opts.acquisition_type not in (
        AcquisitionType.SPECTROSCOPY,
        AcquisitionType.INTEGRATION,
    ):
        raise ValueError(
            "The only allowed acquisition types for this experiment are "
            "'AcquisitionType.SPECTROSCOPY' (on QCCS) and 'AcquisitionType.INTEGRATION'"
            " (on ZQCS) because it sweeps the frequency of a hardware oscillator.",
        )

    amp_sweep_pars = []
    freq_sweep_pars = []
    for q, q_amplitudes, q_frequencies in sweeps_per_quantum_element(
        qubits, amplitudes, frequencies
    ):
        amp_sweep_pars.append(SweepParameter(f"amplitude_{q.uid}", q_amplitudes))
        freq_sweep_pars.append(SweepParameter(f"frequency_{q.uid}", q_frequencies))

    qop = qpu.quantum_operations

    with dsl.sweep(
        name="readout_amplitude_sweep",
        parameter=amp_sweep_pars,
    ):
        with dsl.acquire_loop_rt(
            count=opts.count,
            averaging_mode=opts.averaging_mode,
            acquisition_type=opts.acquisition_type,
            repetition_mode=opts.repetition_mode,
            repetition_time=opts.repetition_time,
            reset_oscillator_phase=opts.reset_oscillator_phase,
        ):
            with dsl.sweep(
                name="qubit_frequency_sweep",
                parameter=freq_sweep_pars,
            ):
                for q, q_amplitude, q_frequency in zip(
                    qubits, amp_sweep_pars, freq_sweep_pars, strict=False
                ):
                    qop.set_frequency(q, q_frequency)
                    qop.set_readout_amplitude(q, amplitude=q_amplitude)
                    # omit_reserves lets the spectroscopy drive and the measure pulse
                    # play simultaneously, such that the readout photons Stark shift
                    # the qubit frequency during the spectroscopy drive
                    qop.qubit_spectroscopy_drive.omit_reserves(q)
                    qop.measure.omit_reserves(q, dsl.handles.result_handle(q.uid))
                    qop.passive_reset(q, delay=opts.spectroscopy_reset_delay)
