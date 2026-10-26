# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the resonator spectroscopy versus flux experiment.

In this experiment, we sweep the resonator frequency and the amplitude of a flux
pulse played on the flux line of the qubit, in a 2D fashion, to measure how the
frequency of the resonator depends on the flux bias of the qubit. In contrast to
the resonator spectroscopy amplitude sweep, the swept amplitude is the one of the
flux pulse and not the one of the measure pulse.

The resonator spectroscopy versus flux experiment has the following pulse sequence:

    qb   --- [ measure ]
    flux --- [ flux pulse (swept amplitude) ]

If multiple qubits are passed to the `experiment_workflow`, the above sequence is
applied in parallel on all the qubits. Since this experiment sweeps the frequency
of the hardware oscillator of the readout line, it can only be run in parallel on
qubits whose measure lines use distinct QA channels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.simple import AcquisitionType, Experiment, SweepParameter, dsl
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.analysis.spectroscopy_two_dimensional_plotting import (
    analysis_workflow,
)
from laboneq_applications.conversion import (
    as_quantum_element_list,
    sweeps_per_quantum_element,
)
from laboneq_applications.experiments.options import (
    ResonatorSpectroscopyExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
)
from laboneq_applications.tasks.verify_qa_channels import _verify_unique_qa_channels

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.workflow(name="resonator_spectroscopy_amplitude_qflux")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: QuantumElements,
    frequencies: QubitSweepPoints,
    amplitudes: QubitSweepPoints,
    temporary_parameters: dict[str | tuple[str, str, str], dict | QuantumParameters]
    | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Workflow for a resonator spectroscopy with a qflux-amplitude sweep.

    The workflow consists of the following steps:

    - [create_experiment][]
    - [compile_experiment][]
    - [run_experiment][]
    - [analysis_workflow][]

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single qubit or
            a list of qubits. The measure lines of the qubits must use distinct QA
            channels, because this experiment sweeps the frequency of the hardware
            oscillator of the readout line, which is shared by all qubits
            multiplexed on the same QA channel.
        frequencies:
            The resonator frequencies to sweep over for the readout pulse (or CW)
            sent to the resonator of each qubit. If `qubits` is a single qubit,
            `frequencies` must be a list of numbers or an array. Otherwise, it must
            be a list of lists of numbers or arrays.
        amplitudes:
            The amplitudes of the flux pulses to sweep over for each qubit. If
            `qubits` is a single qubit, `amplitudes` must be a list of numbers or
            an array. Otherwise, it must be a list of lists of numbers or arrays.
        temporary_parameters:
            The temporary parameters with which to update the quantum elements and
            topology edges. For quantum elements, the dictionary key is the quantum
            element UID. For topology edges, the dictionary key is the edge tuple
            `(tag, source node UID, target node UID)`.
        options:
            The options for building the workflow as an instance of
            [TuneUpWorkflowOptions][]. See the docstring of
            [TuneUpWorkflowOptions][] for more details.

    Returns:
        result:
            The result of the workflow.

    Example:
        ```python
        options = experiment_workflow.options()
        options.count(10)
        qpu = QPU(
            quantum_elements=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_quantum_elements()
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=temp_qubits,
            frequencies=[
                np.linspace(7.1e9, 7.6e9, 501),
                np.linspace(7.2e9, 7.7e9, 501),
            ],
            amplitudes=[np.linspace(0.1, 1, 10), np.linspace(0.1, 1, 10)],
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
        analysis_workflow(
            result=result,
            qubits=temp_qubits,
            sweep_points_1d=frequencies,
            sweep_points_2d=amplitudes,
            label_sweep_points_1d="Readout Frequency,\n$f_{\\mathrm{RO}}$ (GHz)",
            label_sweep_points_2d="Flux-Pulse Amplitude,\n$A$ (a.u.)",
        )
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    frequencies: QubitSweepPoints,
    amplitudes: QubitSweepPoints,
    options: ResonatorSpectroscopyExperimentOptions | None = None,
) -> Experiment:
    """Creates a resonator spectroscopy versus flux experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiment on. May be either a single qubit or
            a list of qubits.
        frequencies:
            The resonator frequencies to sweep over for the readout pulse (or CW)
            sent to the resonator of each qubit. If `qubits` is a single qubit,
            `frequencies` must be a list of numbers or an array. Otherwise, it must
            be a list of lists of numbers or arrays. The frequencies of all the
            qubits must contain the same number of sweep points.
        amplitudes:
            The amplitudes of the flux pulses to sweep over for each qubit. If
            `qubits` is a single qubit, `amplitudes` must be a list of numbers or
            an array. Otherwise, it must be a list of lists of numbers or arrays.
            The amplitudes of all the qubits must contain the same number of sweep
            points.
        options:
            The options for building the experiment.
            See [SpectroscopyExperimentOptions][] and [BaseExperimentOptions][] for
            accepted options.
            Overwrites the options from [TuneupExperimentOptions][] and
            [BaseExperimentOptions][].

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Raises:
        ValueError:
            If the frequencies are not a list of numbers or an array.

        ValueError:
            If the acquisition_type is neither AcquisitionType.SPECTROSCOPY
            nor AcquisitionType.INTEGRATION.

        ValueError:
            If the frequencies or the amplitudes of the qubits do not contain the
            same number of sweep points.

    Example:
        ```python
        options = {"count": 10, "spectroscopy_reset_delay": 3e-6}
        options = ResonatorSpectroscopyExperimentOptions(**options)
        setup = DeviceSetup("my_device")
        qpu = QPU(
            quantum_elements=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_quantum_elements()
        create_experiment(
            qpu=qpu,
            qubits=temp_qubits,
            frequencies=[
                np.linspace(7.1e9, 7.6e9, 501),
                np.linspace(7.2e9, 7.7e9, 501),
            ],
            amplitudes=[np.linspace(0.1, 1, 10), np.linspace(0.1, 1, 10)],
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = ResonatorSpectroscopyExperimentOptions() if options is None else options
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
        amp_sweep_pars.append(SweepParameter(f"amplitudes_{q.uid}", q_amplitudes))
        freq_sweep_pars.append(SweepParameter(f"frequencies_{q.uid}", q_frequencies))
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
            name="flux_pulse_amplitude_sweep",
            parameter=amp_sweep_pars,
        ):
            with dsl.sweep(
                name="resonator_frequency_sweep",
                parameter=freq_sweep_pars,
            ):
                for q, q_amplitude, q_frequency in zip(
                    qubits, amp_sweep_pars, freq_sweep_pars, strict=False
                ):
                    qop.set_frequency(q, frequency=q_frequency, readout=True)
                    # the flux pulse lasts for as long as the qubit is measured
                    flux_length = q.parameters.readout_integration_length
                    if not opts.use_cw:
                        flux_length = max(flux_length, q.parameters.readout_length)
                    with dsl.section(name=f"measure_{q.uid}"):
                        qop.flux_pulse.omit_section(q, q_amplitude, flux_length)
                        if opts.use_cw:
                            qop.acquire.omit_section(
                                q, dsl.handles.result_handle(q.uid)
                            )
                        else:
                            qop.measure.omit_section(
                                q, dsl.handles.result_handle(q.uid)
                            )
                    qop.delay(q, opts.spectroscopy_reset_delay)
