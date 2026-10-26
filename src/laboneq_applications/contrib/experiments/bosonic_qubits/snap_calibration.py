# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""SNAP gate calibration workflow.

The SNAP (Selective Number-Arbitrary Phase) gate applies independent
phases to each Fock state of a bosonic memory:

    S(theta_vec) = sum_n exp(i*theta_n) |n><n|

Part 2 of this workflow maps the drive phase phi of the second
selective pulse to the acquired cavity phase theta_n.

The Part 2 reads the *angular position of the Wigner fringe*.
For each SNAP drive phase phi we trace the Wigner function around a
circle of fixed radius |beta| = sqrt(n) by sweeping the Wigner
displacement phase arg(beta). The SNAP-imprinted cavity phase shows
up as an angular shift of that fringe, which is exactly the quantity
we want and is robust to the multi-Fock background.

Pulse sequence per (qubit, photon number n), 2-D sweep
(phi outer, arg(beta) inner)::

    qb --- [D(n_bar=n)]
        --- [Rx(pi, n, phase=0)]
        --- [Rx(pi, n, phase=phi)]
        --- [W(|beta|=sqrt(n), arg(beta)=beta_phase)]

The analysis fits, for each phi, the fringe W(arg beta) to extract its
angular position theta(phi), then fits theta(phi) vs phi (expected
linear, slope ~ +/-1). The intercept is the calibration constant.

Prerequisites:
    - Selective transmon Rx calibration (amp_pi per n).
    - Displacement calibration (displacement_amp_per_unit_beta).
    - Dispersive shift chi.

Note:
    The ``rx`` quantum operation must accept a ``phase`` keyword
    argument for Part 2. The ``wigner_point`` operation must accept a
    swept ``beta_phase``.

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
from laboneq_applications.conversion import (
    as_quantum_element_list,
    sweeps_per_quantum_element,
)
from laboneq_applications.experiments.options import (
    BaseExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
    update_qpu,
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


@workflow.task_options(base_class=BaseExperimentOptions)
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
            Number of chunks the real-time sweep is split into.
            Default: 1 (0 = auto chunking).
    """

    selective_drive_pulse: dict | None = None
    selective_drive_length: float | None = None
    memory_drive_pulse: dict | None = None
    memory_drive_length: float | None = None
    chunk_count: int = 1  # 0 = auto chunking


@workflow.task_options(base_class=BaseExperimentOptions)
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
        wigner_displacement_pulse:
            Dictionary of pulse overrides for the Wigner-tomography
            displacement pulse.
        wigner_displacement_length:
            Override for the Wigner displacement pulse length in
            seconds.
        chunk_count:
            Number of chunks the real-time sweep is split into.
            Default: 1 (0 = auto chunking). Applies to the inner
            (beta-phase) sweep.
    """

    selective_drive_pulse: dict | None = None
    selective_drive_length: float | None = None
    memory_drive_pulse: dict | None = None
    memory_drive_length: float | None = None
    wigner_displacement_pulse: dict | None = None
    wigner_displacement_length: float | None = None
    chunk_count: int = 1  # 0 = auto chunking


@workflow.workflow_options(base_class=TuneUpWorkflowOptions)
class SnapCalibrationWorkflowOptions:
    """Options for the SNAP calibration workflow.

    Attributes:
        do_phase_calibration:
            Whether to run Part 2 (phase calibration) after
            Part 1 (amplitude calibration). Part 2 uses the
            π amplitude extracted by the Part 1 analysis, so it
            only runs when ``do_analysis`` is also True.
            Default: True.
        do_analysis:
            Whether to run the analysis workflows.
            Default: True.
        update:
            Whether to store the calibrated SNAP parameters in the qpu.
            Default: False.
    """

    do_phase_calibration: bool = workflow.option_field(
        True, description="Whether to run Part 2 (phase calibration)."
    )


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
    phases_wigner: list[ArrayLike | None] | None = None,
    photon_number_memory: int,
    temporary_parameters: dict[str, dict | QuantumParameters] | None = None,
    options: SnapCalibrationWorkflowOptions | None = None,
) -> None:
    """The SNAP gate calibration workflow.

    Runs Part 1 (amplitude calibration) and optionally Part 2
    (phase calibration). Part 2 uses the calibrated pi amplitudes
    from Part 1 and reads out the Wigner fringe angle (see module
    docstring).

    The workflow consists of the following steps:

    - [create_amplitude_experiment][]
    - [compile_amplitude_experiment][]
    - [run_amplitude_experiment][]
    - [amplitude_analysis_workflow][]
    - [update_qpu][]

    - [create_phase_experiment][]
    - [compile_phase_experiment][]
    - [run_phase_experiment][]
    - [phase_analysis_workflow][]
    - [update_qpu][]

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum
            operations.
        qubits:
            The qubits to run the experiments on, passed by UID.
        amplitudes_transmon:
            The selective drive amplitudes to sweep for Part 1.
            Should be centered around the amp_pi values from the
            selective Rx calibration. A single array for a single
            qubit, or a list of arrays (one per qubit).
        phases_transmon:
            The SNAP drive phases to sweep for Part 2 (radians).
            If None, defaults to ``np.linspace(0, 2*pi, 13,
            endpoint=False)`` -- a coarse outer axis is enough since
            theta(phi) is expected linear.
        phases_wigner:
            The Wigner displacement phases arg(beta) to sweep for the
            inner tomography axis (radians). If None, defaults to
            ``np.linspace(0, 2*pi, 48, endpoint=False)``.
        photon_number_memory:
            The cavity photon number to calibrate. For example, ``3``.
        temporary_parameters:
            Temporary parameters to update the qubits with.
        options:
            The options for building the workflow, passed as an instance of
            [SnapCalibrationWorkflowOptions][]. Setting `update` to True stores
            the calibrated `snap_drive_amp_per_n`, `snap_drive_phase_per_n`
            and `snap_drive_phase_slope_sign_per_n` in the qpu.

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
    temp_qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)

    # -------- Part 1: amplitude calibration --------
    amp_exp = create_amplitude_experiment(
        qpu=temp_qpu,
        qubits=temp_qubits,
        amplitudes_transmon=amplitudes_transmon,
        photon_number_memory=photon_number_memory,
    )
    amp_compiled = compile_amplitude_experiment(session, amp_exp)
    amp_result = run_amplitude_experiment(session, amp_compiled)

    # Part 2 (phase calibration) consumes the π amplitude extracted by the
    # Part 1 analysis, so it can only run when do_analysis is enabled.
    with workflow.if_(options.do_analysis):
        amp_analysis = snap_calibration_analysis.amplitude_analysis_workflow(
            amp_result, temp_qubits, amplitudes_transmon, photon_number_memory
        )
        amp_parameters = amp_analysis.output
        with workflow.if_(options.update):
            update_qpu(qpu, amp_parameters["new_parameter_values"])

        # -------- Part 2: phase calibration --------
        with workflow.if_(options.do_phase_calibration):
            phase_exp = create_phase_experiment(
                qpu=temp_qpu,
                qubits=temp_qubits,
                snap_amplitudes=amp_parameters["new_parameter_values"],
                phases_transmon=phases_transmon,
                phases_wigner=phases_wigner,
                photon_number_memory=photon_number_memory,
            )
            phase_compiled = compile_phase_experiment(session, phase_exp)
            phase_result = run_phase_experiment(session, phase_compiled)
            phase_analysis = snap_calibration_analysis.phase_analysis_workflow(
                phase_result,
                temp_qubits,
                photon_number_memory,
                phases_transmon,
                phases_wigner,
            )
            phase_parameters = phase_analysis.output
            with workflow.if_(options.update):
                update_qpu(qpu, phase_parameters["new_parameter_values"])
    workflow.return_(amp_result)


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
    coherent state with n_bar = n and sweeps the amplitude of two
    selective pi pulses *simultaneously* on every qubit.
    The correct pi amplitude is where the transmon excitation is
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
    qubits = as_quantum_element_list(qubits)

    n = photon_number_memory
    qop = qpu.quantum_operations

    amp_params = [
        SweepParameter(
            uid=f"snap_amp_{q.uid}_n{n}",
            values=q_amplitudes,
        )
        for q, q_amplitudes in sweeps_per_quantum_element(qubits, amplitudes_transmon)
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
                # Prepare coherent state with n_bar = n
                if n > 0:
                    qop.displacement(
                        q,
                        n_bar=n,
                        length=opts.memory_drive_length,
                        pulse=opts.memory_drive_pulse,
                    )
                # First pi pulse
                qop.rx(
                    q,
                    angle=None,
                    n=n,
                    amplitude=amplitude,
                    length=opts.selective_drive_length,
                    pulse=opts.selective_drive_pulse,
                )
                # Second pi pulse — same amplitude and direction as the
                # first. Together the two pulses complete a 2π rotation,
                # returning the transmon to |g⟩ at the π amplitude.
                qop.rx(
                    q,
                    angle=None,
                    n=n,
                    amplitude=amplitude,
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
    snap_amplitudes: dict,
    phases_transmon: QubitSweepPoints,
    phases_wigner: QubitSweepPoints,
    photon_number_memory: int,
    options: SnapPhaseExperimentOptions | None = None,
) -> Experiment:
    """Creates the SNAP phase calibration experiment (Part 2).

    Two-dimensional sweep:
        outer  -> SNAP drive phase phi (phases_transmon)
        inner  -> Wigner displacement phase arg(beta) (phases_wigner)

    For each phi we trace a full Wigner fringe around the circle
    |beta| = sqrt(n). The analysis extracts the fringe angle theta(phi)
    and fits theta(phi) vs phi.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum
            operations.
        qubits:
            The qubits to run the experiment on (calibrated in
            parallel).
        snap_amplitudes:
            The new parameter values extracted by the Part 1 analysis,
            in the form ``{q.uid: {"snap_drive_amp_per_n": {str(n): amp}}}``.
            For a qubit without a new value (e.g. a failed fit), the
            stored ``snap_drive_amp_per_n`` qubit parameter is used.
        phases_transmon:
            The SNAP drive phases phi to sweep (outer axis). Either
            None (default grid), a single array (broadcast to all
            qubits), or one array per qubit.
        phases_wigner:
            The Wigner displacement phases arg(beta) to sweep (inner
            axis). Either None (default grid), a single array
            (broadcast to all qubits), or one array per qubit.
        photon_number_memory:
            The cavity photon number to calibrate (shared across
            qubits).
        options:
            The experiment options.

    Returns:
        The LabOne Q experiment.

    Raises:
        ValueError:
            If no SNAP amplitude is available for a qubit at the
            requested photon number.
    """
    opts = SnapPhaseExperimentOptions() if options is None else options
    n = photon_number_memory

    qubits = as_quantum_element_list(qubits)

    # ---- normalise SNAP drive-phase (outer) axis ----
    phases_transmon = snap_calibration_analysis.broadcast_phases(
        phases_transmon,
        len(qubits),
        default=snap_calibration_analysis.DEFAULT_PHASES_TRANSMON,
    )

    # ---- normalise Wigner displacement-phase (inner) axis ----
    phases_wigner = snap_calibration_analysis.broadcast_phases(
        phases_wigner,
        len(qubits),
        default=snap_calibration_analysis.DEFAULT_PHASES_WIGNER,
    )

    qop = qpu.quantum_operations

    # Outer sweep params: SNAP drive phase phi.
    phase_params = [
        SweepParameter(uid=f"snap_phase_{q.uid}_n{n}", values=p)
        for q, p in zip(qubits, phases_transmon, strict=False)
    ]
    # Inner sweep params: Wigner displacement phase arg(beta).
    phase_wigner_params = [
        SweepParameter(uid=f"snap_betaphase_{q.uid}_n{n}", values=b)
        for q, b in zip(qubits, phases_wigner, strict=False)
    ]

    outer_kwargs = {
        "uid": f"snap_phase_n{n}",
        "parameter": phase_params,
    }
    inner_kwargs = {
        "uid": f"snap_betaphase_n{n}",
        "parameter": phase_wigner_params,
    }
    # Chunk the inner (typically larger) axis only.
    if opts.chunk_count > 0:
        inner_kwargs["chunk_count"] = opts.chunk_count
    else:
        inner_kwargs["auto_chunking"] = True

    # Per-qubit calibrated pi amplitudes; resolved at compile time.
    amp_pi_per_qubit = []
    for q in qubits:
        amps = snap_amplitudes.get(q.uid, {}).get(
            "snap_drive_amp_per_n", q.parameters.snap_drive_amp_per_n
        )
        if str(n) not in amps:
            raise ValueError(
                f"Qubit {q.uid!r}: no SNAP amplitude for photon number n={n}. "
                "The Part 1 fit failed and snap_drive_amp_per_n has no stored "
                "value for this photon number."
            )
        amp_pi_per_qubit.append(amps[str(n)])
    beta_amp = float(np.sqrt(n)) if n > 0 else 0.0

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        # Outer: SNAP drive phase phi.
        with dsl.sweep(**outer_kwargs):
            # Inner: Wigner displacement phase arg(beta).
            with dsl.sweep(**inner_kwargs):
                for q, phase, beta_phase, amp_pi_n in zip(
                    qubits,
                    phase_params,
                    phase_wigner_params,
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

                    # First selective pi pulse (reference phase 0).
                    qop.rx(
                        q,
                        angle=None,
                        n=n,
                        amplitude=amp_pi_n,
                        length=opts.selective_drive_length,
                        pulse=opts.selective_drive_pulse,
                    )

                    # Second selective pi pulse: same amplitude as the
                    # first, but with swept drive phase phi. The relative
                    # phase between the two pulses imprints the cavity
                    # phase theta_n = phi on |n⟩.
                    qop.rx(
                        q,
                        angle=None,
                        n=n,
                        amplitude=amp_pi_n,
                        phase=phase,
                        length=opts.selective_drive_length,
                        pulse=opts.selective_drive_pulse,
                    )

                    # Wigner tomography: sample the fringe around the
                    # circle of radius sqrt(n) by sweeping arg(beta).
                    qop.wigner_point(
                        q,
                        beta_amplitude=beta_amp,
                        beta_phase=beta_phase,
                        handle=snap_phase_handle(q.uid, n),
                        displacement_length=opts.wigner_displacement_length,
                        displacement_pulse=opts.wigner_displacement_pulse,
                    )
                    qop.passive_reset(q)


# ═════════════════════════════════════════════════════════════
# Compile / run tasks
# ═════════════════════════════════════════════════════════════


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
    """Return the result handle for a SNAP amplitude measurement."""
    return f"{qubit_uid}/snap_amp_n{n}"


def snap_phase_handle(qubit_uid: str, n: int) -> str:
    """Return the result handle for a SNAP phase measurement."""
    return f"{qubit_uid}/snap_phase_n{n}"
