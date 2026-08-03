# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the displacement calibration experiment for bosonic qubits.

The displacement calibration experiment has the following pulse sequence:

    qb --- [ memory drive (swept amplitude) ] ---
    --- [ transmon x180 (swept frequency) ] --- [ measure ]

The idea is to calibrate the displacement operation on the memory cavity
by sweeping the amplitude of the memory drive pulse and the frequency of
the transmon pi pulse. For each displacement amplitude, the transmon
spectroscopy reveals peaks at frequencies separated by the dispersive
shift chi, corresponding to different photon number states |n⟩ in the
memory cavity. The relative heights of these peaks encode the photon
number distribution, allowing one to calibrate the displacement amplitude
to a desired mean photon number.

If multiple qubits are passed to the `run` workflow, the above pulses are
applied in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.dsl.quantum import QuantumParameters
from laboneq.simple import Experiment, SweepParameter, dsl
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.contrib.analysis.bosonic_qubits import (
    displacement_calibration as displacement_calibration_analysis,
)
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
)
from laboneq_applications.tasks import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.task_options(base_class=TuneupExperimentOptions)
class DisplacementCalibrationExperimentOptions:
    """Options for the displacement calibration experiment.

    Additional attributes:
        memory_drive_pulse:
            Dictionary of pulse overrides for the memory drive
            (displacement) pulse. By default, the pulse parameters from
            the qubit's `memory_drive_pulse` are used.
        memory_drive_length:
            Override for the memory drive pulse length. If None,
            the qubit parameter `memory_drive_length` is used.
        transmon_pi_pulse:
            Dictionary of pulse overrides for the transmon pi pulse.
            By default, the pulse parameters from the qubit's
            `transmon_drive_pulse_when_memory_in_zero_state` are used.
    """

    memory_drive_pulse: dict | None = None
    memory_drive_length: float | None = None
    transmon_pi_pulse: dict | None = None


@workflow.workflow_options
class DisplacementCalibrationWorkflowOptions:
    """Options for the displacement calibration workflow.

    Attributes:
        do_analysis (bool):
            The option for performing the analysis.
    """

    do_analysis: bool = True


@workflow.workflow(name="displacement_calibration")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: list[str] | str,
    *,
    amplitudes: QubitSweepPoints,
    frequencies: QubitSweepPoints,
    temporary_parameters: dict[str, dict | QuantumParameters] | None = None,
    options: DisplacementCalibrationWorkflowOptions | None = None,
) -> None:
    """The displacement calibration workflow.

    The workflow consists of the following steps:

    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()
    - [analysis_workflow]()

    Steps that can be added in the future include:

    - [update_qpu]()

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on, passed by UID. May be
            either a single qubit or a list of qubits.
        amplitudes:
            The displacement amplitudes to sweep over for the memory
            cavity drive for each qubit.
        frequencies:
            The RF frequencies to sweep over for the transmon pi pulse
            for each qubit.
        temporary_parameters:
            The temporary parameters to update the qubits with.
        options:
            The options for building the workflow.
            In addition to options from
            [WorkflowOptions][laboneq.workflow.WorkflowOptions], the following
            custom options are supported:
                - create_experiment: The options for creating the experiment.

    Returns:
        result:
            The result of the workflow.

    Example:
        ```python
        options = experiment_workflow.options()
        options.count(10)
        # QPU from a single-qubit device setup
        qpu = QPU(
            quantum_elements=BosonicQubit.from_device_setup(setup),
            quantum_operations=BosonicQubitOperations(),
        )
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits="q0",
            amplitudes=np.linspace(0.0, 1.0, 51),
            frequencies=np.linspace(4.9e9, 5.1e9, 101),
            options=options,
        ).run()
        ```
    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    exp = create_experiment(
        qpu,
        qubits,
        amplitudes=amplitudes,
        frequencies=frequencies,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        displacement_calibration_analysis.analysis_workflow(
            result, qpu, qubits, amplitudes=amplitudes, frequencies=frequencies
        )
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    amplitudes: QubitSweepPoints,
    frequencies: QubitSweepPoints,
    options: DisplacementCalibrationExperimentOptions | None = None,
) -> Experiment:
    """Creates a displacement calibration experiment.

    The pulse sequence for each qubit is:

    1. Memory drive (displacement) pulse with swept amplitude.
    2. Transmon pi pulse (x180) with swept frequency.
    3. Readout pulse and acquisition.

    This is a 2D sweep: for each displacement amplitude, the transmon
    pi pulse frequency is swept. The resulting transmon spectrum shows
    peaks at frequencies f_transmon - n * chi, where n is the photon
    number in the memory cavity. The relative peak heights reveal the
    photon number distribution produced by the displacement, enabling
    calibration of the displacement amplitude.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        amplitudes:
            The displacement amplitudes to sweep over for the memory
            cavity drive for each qubit.
        frequencies:
            The RF frequencies to sweep over for the transmon pi pulse
            for each qubit.
        options:
            The options for building the experiment.
            See [DisplacementCalibrationExperimentOptions] and
            [BaseExperimentOptions][laboneq_applications.experiments.options.BaseExperimentOptions]
            for accepted options.

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled
            and executed.

    Raises:
        ValueError:
            If amplitudes or frequencies is not a list of numbers or
            array when a single qubit is passed.

    Example:
        ```python
        options = {
            "count": 10,
            "averaging_mode": "cyclic",
            "acquisition_type": "integration_trigger",
            "use_cal_traces": True,
        }
        options = DisplacementCalibrationExperimentOptions(**options)
        # QPU from a single-qubit device setup
        qpu = QPU(
            quantum_elements=BosonicQubit.from_device_setup(setup),
            quantum_operations=BosonicQubitOperations(),
        )
        q0 = qpu["q0"]
        create_experiment(
            qpu=qpu,
            qubits=q0,
            amplitudes=np.linspace(0.0, 1.0, 51),
            frequencies=np.linspace(4.9e9, 5.1e9, 101),
            options=options,
        )
        ```
    """
    opts = DisplacementCalibrationExperimentOptions() if options is None else options
    qubits, amplitudes = validation.validate_and_convert_qubits_sweeps(
        qubits, amplitudes
    )
    # Validate frequencies separately: must have same number of entries
    # as qubits. We rely on the user to pass one frequency array per qubit.
    if not isinstance(frequencies, list):
        frequencies = [frequencies]
    if len(frequencies) == 1 and len(qubits) > 1:
        frequencies = frequencies * len(qubits)

    # One SweepParameter per qubit for each axis.
    amplitude_params = [
        SweepParameter(f"amplitude_{q.uid}", q_amplitudes)
        for q, q_amplitudes in zip(qubits, amplitudes, strict=False)
    ]
    frequency_params = [
        SweepParameter(f"frequency_{q.uid}", q_frequencies)
        for q, q_frequencies in zip(qubits, frequencies, strict=False)
    ]

    qop = qpu.quantum_operations
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        # Outer: all qubit amplitudes swept in lockstep.
        with dsl.sweep(
            name="amplitude_sweep",
            parameter=amplitude_params,
        ):
            # Inner: all qubit frequencies swept in lockstep.
            with dsl.sweep(
                name="frequency_sweep",
                parameter=frequency_params,
            ):
                for q, amplitude, frequency in zip(
                    qubits, amplitude_params, frequency_params, strict=False
                ):
                    qop.set_frequency(q, frequency, target="transmon", n=0)
                    qop.displacement(
                        q,
                        amplitude=amplitude,
                        length=opts.memory_drive_length,
                        pulse=opts.memory_drive_pulse,
                    )
                    qop.x180(
                        q,
                        pulse=opts.transmon_pi_pulse,
                        n=0,
                    )
                    qop.measure(q, dsl.handles.result_handle(q.uid))
                    qop.passive_reset(q)
