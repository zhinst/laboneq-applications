# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""SNAP gate calibration workflow.

The SNAP (Selective Number-Arbitrary Phase) gate applies independent
phases to each Fock state of a bosonic memory:

    S(θ⃗) = Σₙ exp(iθₙ) |n⟩⟨n|

For each photon number n, the SNAP operation is decomposed into
two selective π pulses at the number-dependent transmon frequency
f(n) = f_transmon(0) + n · χ:

    SNAP(θₙ)  =  Rx(π, n)  ·  R_φ(π, n)

where the drive phase φ of the second pulse controls the acquired
cavity phase θₙ.

This workflow calibrates in two parts:

**Part 1 — Amplitude calibration:**

Verifies the π-pulse amplitude in the two-pulse SNAP context.
Two identical selective pulses with the same swept amplitude are
applied; the correct π amplitude corresponds to the minimum
transmon excitation (transmon fully returns to |g⟩).

Pulse sequence per (qubit, photon number n)::

    qb --- [D(n_bar=n)] --- [Rx(amp,n)] --- [Rx(amp,n)] --- [measure]

**Part 2 — Phase calibration:**

Maps the drive phase φ of the second π pulse to the acquired
cavity phase θₙ. The amplitude from Part 1 is used for both
pulses; only the phase of the second pulse is swept.

Pulse sequence per (qubit, photon number n)::

    qb --- [D(n_bar=n)] --- [Rx(π,n,φ=0)] --- [Rx(π,n,φ=sweep)]
                                             --- [W(β=√n)]

The Wigner function at β = √n oscillates sinusoidally with φ.
The phase offset of the oscillation is the calibration constant.

Prerequisites:
    - Selective transmon Rx calibration (amp_pi per n).
    - Displacement calibration (displacement_amp_per_unit_beta).
    - Dispersive shift χ.

Note:
    The ``rx`` quantum operation must accept a ``phase`` keyword
    argument for Part 2. This argument sets the drive oscillator
    phase of the selective pulse.

If multiple qubits are passed, pulse sequences are applied in
parallel on all qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from laboneq import workflow
from laboneq.dsl.quantum import QuantumParameters
from laboneq.simple import CompiledExperiment, Experiment, SweepParameter, dsl
from laboneq.workflow.tasks import compile_experiment as _compile_experiment
from laboneq.workflow.tasks import run_experiment as _run_experiment

from laboneq_applications.contrib.analysis.bosonic_qubits import (
    snap_calibration as snap_calibration_analysis,
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
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import (
        QuantumElements,
        QubitSweepPoints,
    )


# ═════════════════════════════════════════════════════════════
# Options
# ═════════════════════════════════════════════════════════════


@workflow.task_options(base_class=TuneupExperimentOptions)
class SnapAmplitudeExperimentOptions:
    """Options for the SNAP amplitude calibration experiment (Part 1).

    Attributes:
        selective_drive_pulse:
            Dictionary of pulse overrides for the selective
            transmon drive pulse.
        selective_drive_length:
            Override for the selective drive pulse length in
            seconds.
        memory_drive_pulse:
            Dictionary of pulse overrides for the displacement
            pulse used for state preparation.
        memory_drive_length:
            Override for the displacement pulse length in seconds.
        chunk_count:
            Number of sweep points per chunk. Default: 0 (auto
            chunking).
    """

    selective_drive_pulse: dict | None = None
    selective_drive_length: float | None = None
    memory_drive_pulse: dict | None = None
    memory_drive_length: float | None = None
    chunk_count: int = 1  # 0 = auto chunking


@workflow.task_options(base_class=TuneupExperimentOptions)
class SnapPhaseExperimentOptions:
    """Options for the SNAP phase calibration experiment (Part 2).

    Attributes:
        selective_drive_pulse:
            Dictionary of pulse overrides for the selective
            transmon drive pulse.
        selective_drive_length:
            Override for the selective drive pulse length in
            seconds.
        memory_drive_pulse:
            Dictionary of pulse overrides for the displacement
            pulse used for state preparation.
        memory_drive_length:
            Override for the displacement pulse length in seconds.
        chunk_count:
            Number of sweep points per chunk. Default: 0 (auto
            chunking).
    """

    selective_drive_pulse: dict | None = None
    selective_drive_length: float | None = None
    memory_drive_pulse: dict | None = None
    memory_drive_length: float | None = None
    chunk_count: int = 1  # 0 = auto chunking


@workflow.workflow_options
class SnapCalibrationWorkflowOptions:
    """Options for the SNAP calibration workflow.

    Attributes:
        do_analysis (bool):
            The option for performing the analysis.
        do_phase_calibration:
            Whether to run Part 2 (phase calibration) after
            Part 1 (amplitude calibration). Part 2 uses the
            π amplitude extracted by the Part 1 analysis, so it
            only runs when ``do_analysis`` is also True.
    """

    do_phase_calibration: bool = True
    do_analysis: bool = True


# ═════════════════════════════════════════════════════════════
# Workflow
# ═════════════════════════════════════════════════════════════


@workflow.workflow(name="snap_calibration")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: list[str] | str,
    *,
    amplitudes_transmon: list[ArrayLike],
    phases_transmon: list[ArrayLike | None] | None = None,
    photon_number_memory: int,
    temporary_parameters: dict[str, dict | QuantumParameters] | None = None,
    options: SnapCalibrationWorkflowOptions | None = None,
) -> None:
    """The SNAP gate calibration workflow.

    Runs Part 1 (amplitude calibration) and optionally Part 2
    (phase calibration). Part 2 uses the calibrated π amplitudes
    from Part 1.

    The workflow consists of the following steps:

    - [create_amplitude_experiment]()
    - [compile_amplitude_experiment]()
    - [run_amplitude_experiment]()
    - [analyze_amplitude]()

    - [create_phase_experiment]()
    - [compile_phase_experiment]()
    - [run_phase_experiment]()
    - [analyze_phase]()

    Steps that can be added in the future include:

    - [update_qpu]()

    Arguments:
        session:
            The connected session to use for running the
            experiment.
        qpu:
            The qpu consisting of the original qubits and
            quantum operations.
        qubits:
            The qubits to run the experiments on, passed by UID.
        amplitudes_transmon:
            The selective drive amplitudes to sweep for Part 1.
            Should be centered around the amp_pi values from
            the selective Rx calibration. Either a single array
            (broadcast to all qubits) or a list of arrays.
        photon_number_memory:
            The cavity photon number to calibrate.
            For example, ``3``.
        phases_transmon:
            The drive phases to sweep for Part 2 (radians).
            If None, defaults to ``np.linspace(0, 2π, 51)``.
        temporary_parameters:
            Temporary parameters to update the qubits with.
        options:
            The options for building the workflow.

    Returns:
        result:
            The result of the workflow.

    Example:
        ```python
        # QPU from a single-qubit device setup
        qpu = QPU(
            quantum_elements=BosonicQubit.from_device_setup(setup),
            quantum_operations=BosonicQubitOperations(),
        )
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits="q0",
            amplitudes_transmon=np.linspace(0.1, 0.9, 51),
            photon_number_memory=3,
        ).run()
        ```
    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(
        temp_qpu,
        qubits,
    )

    # ── Part 1: amplitude calibration ──────────────────────
    exp_amp = create_amplitude_experiment(
        qpu,
        qubits,
        amplitudes_transmon=amplitudes_transmon,
        photon_number_memory=photon_number_memory,
    )
    compiled_amp = compile_amplitude_experiment(session, exp_amp)
    result_amp = run_amplitude_experiment(session, compiled_amp)
    # Part 2 (phase calibration) consumes the π amplitude extracted by the
    # Part 1 analysis, so it can only run when do_analysis is enabled.
    with workflow.if_(options.do_analysis):
        amp_analysis = snap_calibration_analysis.analyze_amplitude(
            result_amp,
            qpu,
            qubits,
            amplitudes_transmon=amplitudes_transmon,
            photon_number_memory=photon_number_memory,
        )

        # ── Part 2: phase calibration ──────────────────────────
        with workflow.if_(options.do_phase_calibration):
            exp_phase = create_phase_experiment(
                qpu,
                qubits,
                amp_pi_snap=amp_analysis,
                photon_number_memory=photon_number_memory,
                phases_transmon=phases_transmon,
            )
            compiled_phase = compile_phase_experiment(
                session,
                exp_phase,
            )
            result_phase = run_phase_experiment(
                session,
                compiled_phase,
            )
            snap_calibration_analysis.analyze_phase(
                result_phase,
                qpu,
                qubits,
                photon_number_memory=photon_number_memory,
                phases_transmon=phases_transmon,
            )
    workflow.return_(result_amp)


# ═════════════════════════════════════════════════════════════
# Part 1: amplitude calibration
# ═════════════════════════════════════════════════════════════


@workflow.task
@dsl.qubit_experiment
def create_amplitude_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    amplitudes_transmon: QubitSweepPoints,
    photon_number_memory: int,
    options: SnapAmplitudeExperimentOptions | None = None,
) -> Experiment:
    """Creates the SNAP amplitude calibration experiment (Part 1).

    For one shared photon number ``n`` across all qubits, prepares a
    coherent state with n̄ = n and sweeps the amplitude of two
    identical selective π pulses *simultaneously* on every qubit.
    The correct π amplitude is where the transmon excitation is
    minimised.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum
            operations.
        qubits:
            The qubits to run the experiment on (calibrated in
            parallel).
        amplitudes_transmon:
            The selective drive amplitudes to sweep, one array per
            qubit. All arrays must share the same length.
        photon_number_memory:
            The cavity photon number to calibrate (shared across
            qubits).
        options:
            The experiment options.

    Returns:
        The LabOne Q experiment.
    """
    opts = SnapAmplitudeExperimentOptions() if options is None else options
    qubits, amplitudes_transmon = validation.validate_and_convert_qubits_sweeps(
        qubits,
        amplitudes_transmon,
    )

    n = photon_number_memory
    qop = qpu.quantum_operations

    # One SweepParameter per qubit; all swept in lockstep.
    amp_params = [
        SweepParameter(
            uid=f"snap_amp_{q.uid}_n{n}",
            values=q_amplitudes,
        )
        for q, q_amplitudes in zip(
            qubits,
            amplitudes_transmon,
            strict=False,
        )
    ]

    sweep_kwargs = {
        "uid": f"snap_amp_n{n}",
        "parameter": amp_params,
    }
    if opts.chunk_count > 0:
        sweep_kwargs["chunk_count"] = opts.chunk_count
    else:
        sweep_kwargs["auto_chunking"] = True

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        with dsl.sweep(**sweep_kwargs):
            for q, amplitude in zip(
                qubits,
                amp_params,
                strict=False,
            ):
                # Prepare coherent state with n̄ = n
                if n > 0:
                    qop.displacement(
                        q,
                        n_bar=n,
                        length=opts.memory_drive_length,
                        pulse=opts.memory_drive_pulse,
                    )
                # First π pulse
                qop.rx(
                    q,
                    angle=None,
                    n=n,
                    amplitude=amplitude,
                    length=opts.selective_drive_length,
                    pulse=opts.selective_drive_pulse,
                )
                # Second π pulse (same amplitude, opposite sign,
                # mirroring the original behaviour)
                qop.rx(
                    q,
                    angle=None,
                    n=n,
                    amplitude=-amplitude,
                    length=opts.selective_drive_length,
                    pulse=opts.selective_drive_pulse,
                )
                qop.measure(q, snap_amp_handle(q.uid, n))
                qop.passive_reset(q)


# ═════════════════════════════════════════════════════════════
# Part 2: phase calibration
# ═════════════════════════════════════════════════════════════


@workflow.task
@dsl.qubit_experiment
def create_phase_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    amp_pi_snap: dict,
    phases_transmon: QubitSweepPoints,
    photon_number_memory: int,
    options: SnapPhaseExperimentOptions | None = None,
) -> Experiment:
    """Creates the SNAP phase calibration experiment (Part 2)."""
    opts = SnapPhaseExperimentOptions() if options is None else options
    n = photon_number_memory

    if not isinstance(qubits, list):
        qubits = [qubits]

    # Default phase array, broadcast to all qubits if not provided.
    if phases_transmon is None:
        default_phases = np.linspace(0, 2 * np.pi, 51)
        phases_transmon = [default_phases for _ in qubits]
    elif isinstance(phases_transmon, np.ndarray) or (
        isinstance(phases_transmon, list)
        and not (
            len(phases_transmon) > 0
            and isinstance(phases_transmon[0], (list, np.ndarray))
        )
    ):
        # A single 1-D array was passed: broadcast to every qubit.
        single = np.asarray(phases_transmon)
        phases_transmon = [single for _ in qubits]
    else:
        phases_transmon = [np.asarray(p) for p in phases_transmon]

    if len(phases_transmon) != len(qubits):
        raise ValueError(
            "phases_transmon must have one array per qubit "
            f"(got {len(phases_transmon)}, expected {len(qubits)})."
        )

    qop = qpu.quantum_operations

    # Per-qubit phase sweep parameters; all swept in lockstep.
    phase_params = [
        SweepParameter(
            uid=f"snap_phase_{q.uid}_n{n}",
            values=q_phases,
        )
        for q, q_phases in zip(
            qubits,
            phases_transmon,
            strict=False,
        )
    ]

    sweep_kwargs = {
        "uid": f"snap_phase_n{n}",
        "parameter": phase_params,
    }
    if opts.chunk_count > 0:
        sweep_kwargs["chunk_count"] = opts.chunk_count
    else:
        sweep_kwargs["auto_chunking"] = True

    # Per-qubit calibrated π amplitudes; resolved at compile time.
    amp_pi_per_qubit = [amp_pi_snap[q.uid]["amp_pi_snap"][str(n)] for q in qubits]
    beta_amp = float(np.sqrt(n)) if n > 0 else 0.0

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        with dsl.sweep(**sweep_kwargs):
            for q, phase, amp_pi_n in zip(
                qubits,
                phase_params,
                amp_pi_per_qubit,
                strict=False,
            ):
                if n > 0:
                    qop.displacement(
                        q,
                        n_bar=n,
                        length=opts.memory_drive_length,
                        pulse=opts.memory_drive_pulse,
                    )

                qop.rx(
                    q,
                    angle=None,
                    n=n,
                    amplitude=amp_pi_n,
                    length=opts.selective_drive_length,
                    pulse=opts.selective_drive_pulse,
                )

                qop.rx(
                    q,
                    angle=None,
                    n=n,
                    amplitude=-amp_pi_n,
                    phase=phase,
                    length=opts.selective_drive_length,
                    pulse=opts.selective_drive_pulse,
                )

                qop.wigner_point(
                    q,
                    beta_amplitude=beta_amp,
                    beta_phase=0.0,
                    handle=snap_phase_handle(q.uid, n),
                )
                qop.passive_reset(q)


@workflow.task(name="compile_amplitude_experiment")
def compile_amplitude_experiment(
    session: Session,
    experiment: Experiment,
) -> CompiledExperiment:
    """Compile the SNAP amplitude experiment."""
    return _compile_experiment.func(session, experiment)


@workflow.task(name="run_amplitude_experiment")
def run_amplitude_experiment(
    session: Session,
    compiled_experiment: CompiledExperiment,
) -> RunExperimentResults:
    """Run the SNAP amplitude experiment."""
    return _run_experiment.func(session, compiled_experiment)


@workflow.task(name="compile_phase_experiment")
def compile_phase_experiment(
    session: Session,
    experiment: Experiment,
) -> CompiledExperiment:
    """Compile the SNAP phase experiment."""
    return _compile_experiment.func(session, experiment)


@workflow.task(name="run_phase_experiment")
def run_phase_experiment(
    session: Session,
    compiled_experiment: CompiledExperiment,
) -> RunExperimentResults:
    """Run the SNAP phase experiment."""
    return _run_experiment.func(session, compiled_experiment)


# ═════════════════════════════════════════════════════════════
# Handles
# ═════════════════════════════════════════════════════════════


def snap_amp_handle(qubit_uid: str, n: int) -> str:
    """Return the result handle for a SNAP amplitude measurement.

    Arguments:
        qubit_uid: The qubit UID.
        n: The photon number.

    Returns:
        The result handle string.
    """
    return f"{qubit_uid}/snap_amp_n{n}"


def snap_phase_handle(qubit_uid: str, n: int) -> str:
    """Return the result handle for a SNAP phase measurement.

    Arguments:
        qubit_uid: The qubit UID.
        n: The photon number.

    Returns:
        The result handle string.
    """
    return f"{qubit_uid}/snap_phase_n{n}"
