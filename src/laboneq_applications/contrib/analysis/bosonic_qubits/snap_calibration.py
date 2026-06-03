# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Analysis for the SNAP gate calibration experiment.

Part 1 — Amplitude analysis:
    Fits the transmon response vs. selective drive amplitude to
    a sin² model. The π amplitude is where the transmon returns
    to |g⟩ after two sequential selective pulses:

        P(e) = C · sin²(π · amp / amp_π) + offset

Part 2 — Phase analysis:
    Fits the Wigner signal vs. drive phase to a sinusoidal model.
    The phase offset gives the calibration constant mapping
    drive phase → acquired cavity phase:

        W(φ) = A · cos(φ + φ₀) + B
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.workflow.tasks.run_experiment import (
        RunExperimentResults,
    )

    from laboneq_applications.typing import (
        QuantumElements,
        QubitSweepPoints,
    )


# ═════════════════════════════════════════════════════════════
# Options
# ═════════════════════════════════════════════════════════════


@workflow.task_options
class SnapAmplitudeAnalysisOptions:
    """Options for SNAP amplitude analysis (Part 1).

    Attributes:
        generate_plot: Whether to generate plots. Default: True.
        show_plot: Whether to show plots. Default: True.
        save_figures: Whether to save figures. Default: False.
        save_path: Path to save figures. Default: "./"
        num_fit_points: Number of points for the fit curve.
    """

    generate_plot: bool = True
    show_plot: bool = True
    save_figures: bool = False
    save_path: str = "./"
    num_fit_points: int = 200


@workflow.task_options
class SnapPhaseAnalysisOptions:
    """Options for SNAP phase analysis (Part 2).

    Attributes:
        generate_plot: Whether to generate plots. Default: True.
        show_plot: Whether to show plots. Default: True.
        save_figures: Whether to save figures. Default: False.
        save_path: Path to save figures. Default: "./"
        num_fit_points: Number of points for the fit curve.
    """

    generate_plot: bool = True
    show_plot: bool = True
    save_figures: bool = False
    save_path: str = "./"
    num_fit_points: int = 200


# ═════════════════════════════════════════════════════════════
# Analysis tasks
# ═════════════════════════════════════════════════════════════


@workflow.task
def analyze_amplitude(
    result: RunExperimentResults,
    qpu: QPU,
    qubits: QuantumElements,
    amplitudes_transmon: QubitSweepPoints,
    photon_number_memory: int,
    options: SnapAmplitudeAnalysisOptions | None = None,
) -> dict:
    """Analyze Part 1: extract π amplitude for SNAP.

    Fits a sin² model to the two-pulse Rabi data and extracts
    the π amplitude for the given (qubit, photon number).

    Arguments:
        result: Experiment results.
        qpu: The QPU.
        qubits: The qubits that were measured.
        amplitudes_transmon: The amplitude sweeps that were used.
        photon_number_memory: The photon number that was
            calibrated.
        options: Analysis options.

    Returns:
        Dictionary keyed by qubit UID with the structure::

            {
                "q0": {
                    "amp_pi_snap": {"3": 0.45},
                    "rabi_data": {...},
                    "amplitudes": array,
                    "photon_number": 3,
                },
            }
    """
    opts = SnapAmplitudeAnalysisOptions() if options is None else options

    qubits = _ensure_list(qubits)
    amplitudes_transmon = _broadcast(
        amplitudes_transmon,
        len(qubits),
    )
    n = photon_number_memory

    analysis_results = {}

    for q, q_amps_raw in zip(
        qubits,
        amplitudes_transmon,
        strict=False,
    ):
        q_amps = np.asarray(q_amps_raw)

        handle = _snap_amp_handle(q.uid, n)
        data_raw = result.get_data(handle)
        data = np.real(np.asarray(data_raw))

        fit_result = _fit_snap_rabi(
            q_amps,
            data,
            opts.num_fit_points,
        )

        rabi_data = {
            "raw_data": data,
            "fit_params": list(fit_result["params"]),
            "fit_success": bool(fit_result["success"]),
            "amp_pi": float(fit_result["amp_pi"]),
            "fit_x": fit_result["fit_x"],
            "fit_y": fit_result["fit_y"],
        }

        analysis_results[q.uid] = {
            "amp_pi_snap": {str(n): float(fit_result["amp_pi"])},
            "rabi_data": rabi_data,
            "amplitudes": q_amps,
            "photon_number": n,
        }

        if opts.generate_plot:
            _plot_amplitude_calibration(
                q.uid,
                q_amps,
                rabi_data,
                n,
                opts,
            )

    return analysis_results


@workflow.task
def analyze_phase(
    result: RunExperimentResults,
    qpu: QPU,
    qubits: QuantumElements,
    photon_number_memory: int,
    phases_transmon: (np.ndarray | list[np.ndarray] | list[list[float]] | None) = None,
    options: SnapPhaseAnalysisOptions | None = None,
) -> dict:
    """Analyze Part 2: extract SNAP phase offset.

    Accepts ``phases_transmon`` in any of:
        - ``None``: defaults to ``np.linspace(0, 2π, 51)`` for every qubit.
        - 1-D array: broadcast to every qubit.
        - list of 1-D arrays (one per qubit): used as-is.
    """
    opts = SnapPhaseAnalysisOptions() if options is None else options

    qubits = _ensure_list(qubits)
    n = photon_number_memory

    # Normalise phases_transmon into a per-qubit list of 1-D arrays.
    phases_per_qubit = _broadcast_phases(phases_transmon, len(qubits))

    analysis_results = {}

    for q, phases_q_raw in zip(
        qubits,
        phases_per_qubit,
        strict=False,
    ):
        phases_q = np.asarray(phases_q_raw).ravel()

        handle = _snap_phase_handle(q.uid, n)
        data_raw = result.get_data(handle)
        data = np.real(np.asarray(data_raw)).ravel()

        fit_result = _fit_sinusoid(
            phases_q,
            data,
            opts.num_fit_points,
        )

        phase_data = {
            "raw_data": data,
            "fit_params": list(fit_result["params"]),
            "fit_success": bool(fit_result["success"]),
            "phase_offset": float(fit_result["phase_offset"]),
            "fit_x": fit_result["fit_x"],
            "fit_y": fit_result["fit_y"],
        }

        analysis_results[q.uid] = {
            "phase_offsets": {
                str(n): float(fit_result["phase_offset"]),
            },
            "phase_data": phase_data,
            "phases": phases_q,
            "photon_number": n,
        }

        if opts.generate_plot:
            _plot_phase_calibration(
                q.uid,
                phases_q,
                phase_data,
                n,
                opts,
            )

    return analysis_results


# ═════════════════════════════════════════════════════════════
# Fitting helpers
# ═════════════════════════════════════════════════════════════


def _snap_rabi_model(
    x: np.ndarray,
    amp_pi: float,
    contrast: float,
    offset: float,
) -> np.ndarray:
    """Two-pulse SNAP Rabi model.

    P(e) = contrast · sin²(π · x / amp_pi) + offset

    At x = amp_pi, sin² = 0 → minimum excitation.

    Arguments:
        x: Drive amplitude array.
        amp_pi: π-pulse amplitude.
        contrast: Peak-to-peak signal contrast.
        offset: Baseline offset.

    Returns:
        Model values.
    """
    return contrast * np.sin(np.pi * x / amp_pi) ** 2 + offset


def _fit_snap_rabi(
    amplitudes: np.ndarray,
    data: np.ndarray,
    num_fit_points: int = 200,
) -> dict:
    """Fit two-pulse SNAP Rabi and extract π amplitude.

    Arguments:
        amplitudes: Sweep amplitudes.
        data: Measured signal.
        num_fit_points: Points for fit curve.

    Returns:
        Dictionary with fit results.
    """
    try:
        offset_guess = float(np.min(data))
        contrast_guess = float(np.max(data) - np.min(data))
        # Maximum of sin² is at amp_pi / 2
        max_idx = np.argmax(data)
        amp_pi_guess = float(2 * amplitudes[max_idx])
        amp_pi_guess = max(amp_pi_guess, amplitudes[1])

        popt, _ = curve_fit(
            _snap_rabi_model,
            amplitudes,
            data,
            p0=[amp_pi_guess, contrast_guess, offset_guess],
            bounds=(
                [amplitudes[1], 0, -np.inf],
                [np.inf, np.inf, np.inf],
            ),
            maxfev=10000,
        )
        success = True
    except (RuntimeError, ValueError):
        popt = np.array(
            [
                amplitudes[-1] / 2,
                0.0,
                float(np.mean(data)),
            ]
        )
        success = False

    amp_pi = float(popt[0])

    fit_x = np.linspace(
        amplitudes[0],
        amplitudes[-1],
        num_fit_points,
    )
    fit_y = _snap_rabi_model(fit_x, *popt)

    return {
        "params": tuple(popt),
        "success": success,
        "amp_pi": amp_pi,
        "fit_x": fit_x,
        "fit_y": fit_y,
    }


def _sinusoid_model(
    phi: np.ndarray,
    amplitude: float,
    phase_offset: float,
    offset: float,
) -> np.ndarray:
    """Sinusoidal model for phase calibration.

    W(φ) = amplitude · cos(φ + φ₀) + offset

    Arguments:
        phi: Drive phase array (radians).
        amplitude: Oscillation amplitude.
        phase_offset: Phase offset φ₀.
        offset: Baseline offset.

    Returns:
        Model values.
    """
    return amplitude * np.cos(phi + phase_offset) + offset


def _fit_sinusoid(
    phases: np.ndarray,
    data: np.ndarray,
    num_fit_points: int = 200,
) -> dict:
    """Fit sinusoidal oscillation and extract phase offset.

    Arguments:
        phases: Phase sweep (radians).
        data: Measured Wigner signal.
        num_fit_points: Points for fit curve.

    Returns:
        Dictionary with fit results.
    """
    try:
        offset_guess = float(np.mean(data))
        amp_guess = float(
            (np.max(data) - np.min(data)) / 2,
        )
        phase_guess = float(-phases[np.argmax(data)])

        popt, _ = curve_fit(
            _sinusoid_model,
            phases,
            data,
            p0=[amp_guess, phase_guess, offset_guess],
            maxfev=10000,
        )
        success = True
    except (RuntimeError, ValueError):
        popt = np.array([0.0, 0.0, float(np.mean(data))])
        success = False

    # Normalize phase to [0, 2π)
    phase_offset = float(popt[1] % (2 * np.pi))

    fit_x = np.linspace(
        phases[0],
        phases[-1],
        num_fit_points,
    )
    fit_y = _sinusoid_model(fit_x, *popt)

    return {
        "params": tuple(popt),
        "success": success,
        "phase_offset": phase_offset,
        "fit_x": fit_x,
        "fit_y": fit_y,
    }


# ═════════════════════════════════════════════════════════════
# Plotting helpers
# ═════════════════════════════════════════════════════════════


def _plot_amplitude_calibration(
    q_uid: str,
    amplitudes: np.ndarray,
    rabi_data: dict,
    photon_number: int,
    opts: SnapAmplitudeAnalysisOptions,
) -> plt.Figure:
    """Plot Part 1: two-pulse Rabi for a single photon number."""
    fig, ax = plt.subplots(
        figsize=(6, 4),
        constrained_layout=True,
    )

    ax.plot(
        amplitudes,
        rabi_data["raw_data"],
        "o",
        markersize=3,
        alpha=0.6,
        label="Data",
    )
    ax.plot(
        rabi_data["fit_x"],
        rabi_data["fit_y"],
        "-",
        linewidth=1.5,
        label="Fit",
    )
    ax.axvline(
        rabi_data["amp_pi"],
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"amp_π = {rabi_data['amp_pi']:.4f}",
    )
    ax.set_xlabel("Selective drive amplitude")
    ax.set_ylabel("Signal")
    ax.set_title(
        f"{q_uid} — SNAP amplitude calibration (Part 1) — n = {photon_number}",
        fontweight="bold",
    )
    ax.legend(fontsize=8)

    if opts.save_figures:
        fig.savefig(
            f"{opts.save_path}/snap_amp_cal_{q_uid}_n{photon_number}.png",
            dpi=150,
            bbox_inches="tight",
        )
    if opts.show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _plot_phase_calibration(
    q_uid: str,
    phases: np.ndarray,
    phase_data: dict,
    photon_number: int,
    opts: SnapPhaseAnalysisOptions,
) -> plt.Figure:
    """Plot Part 2: Wigner signal vs drive phase."""
    fig, ax = plt.subplots(
        figsize=(6, 4),
        constrained_layout=True,
    )

    ax.plot(
        np.degrees(phases),
        phase_data["raw_data"],
        "o",
        markersize=3,
        alpha=0.6,
        label="Data",
    )
    ax.plot(
        np.degrees(phase_data["fit_x"]),
        phase_data["fit_y"],
        "-",
        linewidth=1.5,
        label="Fit",
    )
    phi_deg = np.degrees(phase_data["phase_offset"])
    ax.axvline(
        phi_deg,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"φ₀ = {phi_deg:.1f}°",
    )
    ax.set_xlabel("Drive phase (deg)")
    ax.set_ylabel("Wigner signal")
    ax.set_title(
        f"{q_uid} — SNAP phase calibration (Part 2) — n = {photon_number}",
        fontweight="bold",
    )
    ax.legend(fontsize=8)

    if opts.save_figures:
        fig.savefig(
            f"{opts.save_path}/snap_phase_cal_{q_uid}_n{photon_number}.png",
            dpi=150,
            bbox_inches="tight",
        )
    if opts.show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ═════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════


def _ensure_list(qubits: QuantumElements | list) -> list:
    """Ensure qubits is a list."""
    if not isinstance(qubits, list):
        return [qubits]
    return qubits


def _broadcast(points: list | object, n_qubits: int) -> list:
    """Broadcast sweep points to match qubit count."""
    if not isinstance(points, list):
        points = [points]
    if len(points) == 1 and n_qubits > 1:
        points = points * n_qubits
    return points


def _broadcast_phases(
    phases_transmon: np.ndarray | list | None,
    n_qubits: int,
) -> list[np.ndarray]:
    """Normalise ``phases_transmon`` into a per-qubit list of 1-D arrays.

    Accepts:
        - ``None``: default ``np.linspace(0, 2π, 51)`` broadcast to all qubits.
        - 1-D array / 1-D list: broadcast to all qubits.
        - List of 1-D arrays (one per qubit): used as-is.
    """
    if phases_transmon is None:
        default = np.linspace(0, 2 * np.pi, 51)
        return [default for _ in range(n_qubits)]

    # Detect "list of per-qubit arrays" vs "single 1-D array/list".
    is_list_of_arrays = (
        isinstance(phases_transmon, list)
        and len(phases_transmon) > 0
        and isinstance(phases_transmon[0], (list, tuple, np.ndarray))
    )

    if is_list_of_arrays:
        out = [np.asarray(p).ravel() for p in phases_transmon]
        if len(out) == 1 and n_qubits > 1:
            out = out * n_qubits
        if len(out) != n_qubits:
            raise ValueError(
                "phases_transmon must have one array per qubit "
                f"(got {len(out)}, expected {n_qubits})."
            )
        return out

    # Single 1-D array/list → broadcast.
    single = np.asarray(phases_transmon).ravel()
    return [single for _ in range(n_qubits)]


def _snap_amp_handle(qubit_uid: str, n: int) -> str:
    """Result handle for SNAP amplitude measurement."""
    return f"{qubit_uid}/snap_amp_n{n}"


def _snap_phase_handle(qubit_uid: str, n: int) -> str:
    """Result handle for SNAP phase measurement."""
    return f"{qubit_uid}/snap_phase_n{n}"
