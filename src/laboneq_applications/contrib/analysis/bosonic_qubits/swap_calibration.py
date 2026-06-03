# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Analysis for the SWAP gate calibration experiment.

Supports three modes:
    - "length":     1D fit of a damped cosine vs. time.
    - "amplitude":  1D fit of a damped cosine vs. amplitude.
    - "2d":         Chevron pattern; extracts the optimal
                    (amplitude, length) by finding the row
                    (fixed amplitude) with the highest
                    oscillation frequency and taking the
                    corresponding ``t_swap``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    import matplotlib as mpl
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
class SwapAnalysisOptions:
    """Options for SWAP calibration analysis.

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
    num_fit_points: int = 400


@workflow.workflow_options
class SwapAnalysisWorkflowOptions:
    """Options for the SWAP calibration analysis workflow."""


@workflow.workflow(name="swap_calibration_analysis")
def analysis_workflow(
    result: RunExperimentResults,
    qpu: QPU,
    qubits: QuantumElements,
    *,
    swap_durations: QubitSweepPoints | None = None,
    swap_amplitudes: QubitSweepPoints | None = None,
    options: SwapAnalysisWorkflowOptions | None = None,
) -> dict:
    """Workflow wrapper for SWAP calibration analysis."""
    options = SwapAnalysisWorkflowOptions() if options is None else options

    return analyze_swap(
        result=result,
        qpu=qpu,
        qubits=qubits,
        swap_durations=swap_durations,
        swap_amplitudes=swap_amplitudes,
    )


# ═════════════════════════════════════════════════════════════
# Analysis task
# ═════════════════════════════════════════════════════════════


@workflow.task
def analyze_swap(
    result: RunExperimentResults,
    qpu: QPU,
    qubits: QuantumElements,
    swap_durations: QubitSweepPoints | None = None,
    swap_amplitudes: QubitSweepPoints | None = None,
    options: SwapAnalysisOptions | None = None,
) -> dict:
    """Analyze the SWAP calibration.

    Mode is inferred from which inputs are provided:
        - ``swap_durations`` only → length sweep.
        - ``swap_amplitudes`` only → amplitude sweep.
        - Both → 2D chevron sweep.

    Arguments:
        result: Experiment results.
        qpu: The QPU.
        qubits: The qubits that were measured.
        swap_durations: The duration sweep (if any).
        swap_amplitudes: The amplitude sweep (if any).
        options: Analysis options.

    Returns:
        Dictionary keyed by qubit UID. Structure depends on
        mode — each dict always contains the key ``"mode"`` and
        the calibrated ``"t_swap"`` and/or ``"amp_swap"``.
    """
    opts = SwapAnalysisOptions() if options is None else options

    mode = _get_sweep_mode(swap_durations, swap_amplitudes)

    qubits = _ensure_list(qubits)
    if swap_durations is not None:
        swap_durations = _broadcast(
            swap_durations,
            len(qubits),
        )
    if swap_amplitudes is not None:
        swap_amplitudes = _broadcast(
            swap_amplitudes,
            len(qubits),
        )

    analysis_results = {}

    for i, q in enumerate(qubits):
        handle = _swap_handle(q.uid)
        data_raw = result.get_data(handle)
        data = np.real(np.asarray(data_raw))

        if mode == "length":
            q_durs = np.asarray(swap_durations[i])
            fit = _fit_damped_cosine(
                q_durs,
                data,
                "time",
                opts.num_fit_points,
            )
            analysis_results[q.uid] = {
                "mode": "length",
                "t_swap": float(fit["x_min"]),
                "frequency": float(fit["frequency"]),
                "raw_data": data,
                "durations": q_durs,
                "fit_params": list(fit["params"]),
                "fit_success": bool(fit["success"]),
                "fit_x": fit["fit_x"],
                "fit_y": fit["fit_y"],
            }

        elif mode == "amplitude":
            q_amps = np.asarray(swap_amplitudes[i])
            fit = _fit_damped_cosine(
                q_amps,
                data,
                "amplitude",
                opts.num_fit_points,
            )
            analysis_results[q.uid] = {
                "mode": "amplitude",
                "amp_swap": float(fit["x_min"]),
                "rabi_rate_per_unit_amp": float(fit["frequency"]),
                "raw_data": data,
                "amplitudes": q_amps,
                "fit_params": list(fit["params"]),
                "fit_success": bool(fit["success"]),
                "fit_x": fit["fit_x"],
                "fit_y": fit["fit_y"],
            }

        else:  # "2d"
            q_amps = np.asarray(swap_amplitudes[i])
            q_durs = np.asarray(swap_durations[i])
            # Data shape: (n_amps, n_durations)
            data_2d = data.reshape(len(q_amps), len(q_durs))
            chevron = _fit_chevron(
                q_amps,
                q_durs,
                data_2d,
                opts.num_fit_points,
            )
            analysis_results[q.uid] = {
                "mode": "2d",
                "t_swap": float(chevron["t_swap"]),
                "amp_swap": float(chevron["amp_swap"]),
                "raw_data": data_2d,
                "amplitudes": q_amps,
                "durations": q_durs,
                "row_fits": chevron["row_fits"],
                "frequencies_per_amp": chevron["frequencies"],
            }

        if opts.generate_plot:
            _plot_swap_calibration(
                q.uid,
                analysis_results[q.uid],
                opts,
            )

    return analysis_results


# ═════════════════════════════════════════════════════════════
# Fitting helpers
# ═════════════════════════════════════════════════════════════


def _damped_cosine_model(
    x: np.ndarray,
    amplitude: float,
    frequency: float,
    phase: float,
    decay: float,
    offset: float,
) -> np.ndarray:
    """Damped cosine model.

    f(x) = A · cos(2π · f · x + φ) · exp(-x / τ) + B

    For "time" sweeps, ``x`` is seconds and ``f`` is Hz.
    For "amplitude" sweeps, ``x`` is dimensionless amplitude
    and ``f`` is oscillation rate per unit amplitude.
    """
    return (
        amplitude * np.cos(2 * np.pi * frequency * x + phase) * np.exp(-x / decay)
        + offset
    )


def _fit_damped_cosine(
    x: np.ndarray,
    data: np.ndarray,
    sweep_type: str,
    num_fit_points: int = 400,
) -> dict:
    """Fit damped cosine and extract location of first minimum.

    The first minimum corresponds to full |e,0⟩ → |g,1⟩
    transfer. For cos(2π·f·x + φ):
        - If amplitude >= 0: first min at 2π·f·x + φ = π.
        - If amplitude  < 0: first min at 2π·f·x + φ = 0.

    Arguments:
        x: Sweep axis (seconds or amplitude units).
        data: Measured transmon signal.
        sweep_type: "time" or "amplitude" (for bounds/guesses).
        num_fit_points: Points for fit curve.

    Returns:
        Dictionary with fit results including ``"x_min"``.
    """
    try:
        offset_guess = float(np.mean(data))
        amp_guess = float((np.max(data) - np.min(data)) / 2)

        # FFT-based frequency guess.
        dx = float(x[1] - x[0])
        n = len(data)
        freqs = np.fft.rfftfreq(n, d=dx)
        spectrum = np.abs(
            np.fft.rfft(data - np.mean(data)),
        )
        freq_guess = float(
            freqs[1 + int(np.argmax(spectrum[1:]))],
        )
        if freq_guess <= 0:
            freq_guess = 1.0 / (x[-1] - x[0])

        decay_guess = float((x[-1] - x[0]) * 2.0)

        popt, _ = curve_fit(
            _damped_cosine_model,
            x,
            data,
            p0=[
                amp_guess,
                freq_guess,
                0.0,
                decay_guess,
                offset_guess,
            ],
            bounds=(
                [0, 0, -2 * np.pi, 0, -np.inf],
                [np.inf, np.inf, 2 * np.pi, np.inf, np.inf],
            ),
            maxfev=10000,
        )
        success = True
    except (RuntimeError, ValueError):
        popt = np.array(
            [
                0.0,
                1.0 / (x[-1] - x[0]),
                0.0,
                x[-1],
                float(np.mean(data)),
            ]
        )
        success = False

    amplitude, frequency, phase, _decay, _offset = popt

    # First minimum (first point of maximum |e⟩ population
    # transferred away, assuming data starts near |e⟩).
    target_arg = np.pi if amplitude >= 0 else 0.0

    x_min = (target_arg - phase) / (2 * np.pi * frequency)
    period = 1.0 / frequency
    while x_min <= 0:
        x_min += period / 2

    fit_x = np.linspace(x[0], x[-1], num_fit_points)
    fit_y = _damped_cosine_model(fit_x, *popt)

    return {
        "params": tuple(popt),
        "success": success,
        "x_min": float(x_min),
        "frequency": float(frequency),
        "fit_x": fit_x,
        "fit_y": fit_y,
    }


def _fit_chevron(
    amplitudes: np.ndarray,
    durations: np.ndarray,
    data_2d: np.ndarray,
    num_fit_points: int = 400,
) -> dict:
    """Fit a 2D chevron by fitting each amplitude row.

    The optimal operating point is the amplitude that gives
    the highest oscillation frequency (fastest SWAP). The
    corresponding ``t_swap`` is the first minimum of that row.

    Arguments:
        amplitudes: Amplitude axis (outer).
        durations: Duration axis (inner, seconds).
        data_2d: Data of shape (n_amps, n_durations).
        num_fit_points: Fit curve resolution.

    Returns:
        Dictionary with:
            - ``t_swap``: best duration.
            - ``amp_swap``: best amplitude.
            - ``frequencies``: fit frequencies per amplitude.
            - ``row_fits``: list of per-row fit dicts.
    """
    frequencies = []
    row_fits = []

    for row in data_2d:
        row_fit = _fit_damped_cosine(
            durations,
            row,
            "time",
            num_fit_points,
        )
        row_fits.append(row_fit)
        # If a fit failed, mark frequency as NaN so it is
        # excluded from the max search.
        freq = row_fit["frequency"] if row_fit["success"] else np.nan
        frequencies.append(freq)

    frequencies = np.array(frequencies)

    best_idx = 0 if np.all(np.isnan(frequencies)) else int(np.nanargmax(frequencies))

    amp_swap = float(amplitudes[best_idx])
    t_swap = float(row_fits[best_idx]["x_min"])

    return {
        "t_swap": t_swap,
        "amp_swap": amp_swap,
        "frequencies": frequencies,
        "row_fits": row_fits,
    }


# ═════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════


def _plot_swap_calibration(
    q_uid: str,
    result: dict,
    opts: SwapAnalysisOptions,
) -> plt.Figure:
    """Dispatch plot by sweep mode."""
    if result["mode"] == "length":
        return _plot_1d_length(q_uid, result, opts)
    if result["mode"] == "amplitude":
        return _plot_1d_amplitude(q_uid, result, opts)
    return _plot_2d_chevron(q_uid, result, opts)


def _plot_1d_length(
    q_uid: str,
    result: dict,
    opts: SwapAnalysisOptions,
) -> mpl.figure.Figure:
    fig, ax = plt.subplots(
        figsize=(7, 4),
        constrained_layout=True,
    )
    durations = result["durations"]
    ax.plot(
        durations * 1e9,
        result["raw_data"],
        "o",
        markersize=3,
        alpha=0.6,
        label="Data",
    )
    ax.plot(
        result["fit_x"] * 1e9,
        result["fit_y"],
        "-",
        linewidth=1.5,
        label="Fit",
    )
    ax.axvline(
        result["t_swap"] * 1e9,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"t_swap = {result['t_swap'] * 1e9:.2f} ns",
    )
    ax.set_xlabel("SWAP duration (ns)")
    ax.set_ylabel("Transmon signal")
    ax.set_title(
        f"{q_uid} — SWAP length calibration (f = {result['frequency'] * 1e-6:.3f} MHz)",
        fontweight="bold",
    )
    ax.legend(fontsize=8)
    _save_and_show(fig, f"swap_cal_length_{q_uid}", opts)
    return fig


def _plot_1d_amplitude(
    q_uid: str,
    result: dict,
    opts: SwapAnalysisOptions,
) -> mpl.figure.Figure:
    fig, ax = plt.subplots(
        figsize=(7, 4),
        constrained_layout=True,
    )
    amplitudes = result["amplitudes"]
    ax.plot(
        amplitudes,
        result["raw_data"],
        "o",
        markersize=3,
        alpha=0.6,
        label="Data",
    )
    ax.plot(
        result["fit_x"],
        result["fit_y"],
        "-",
        linewidth=1.5,
        label="Fit",
    )
    ax.axvline(
        result["amp_swap"],
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"amp_swap = {result['amp_swap']:.4f}",
    )
    ax.set_xlabel("SWAP drive amplitude")
    ax.set_ylabel("Transmon signal")
    ax.set_title(
        f"{q_uid} — SWAP amplitude calibration",
        fontweight="bold",
    )
    ax.legend(fontsize=8)
    _save_and_show(fig, f"swap_cal_amplitude_{q_uid}", opts)
    return fig


def _plot_2d_chevron(
    q_uid: str,
    result: dict,
    opts: SwapAnalysisOptions,
) -> mpl.figure.Figure:
    fig, ax = plt.subplots(
        figsize=(7.5, 5),
        constrained_layout=True,
    )
    amplitudes = result["amplitudes"]
    durations = result["durations"]
    data_2d = result["raw_data"]

    # Pixel extent (corner coordinates).
    da = amplitudes[1] - amplitudes[0] if len(amplitudes) > 1 else 1e-3
    dt = durations[1] - durations[0] if len(durations) > 1 else 1e-9
    extent = [
        (durations[0] - dt / 2) * 1e9,
        (durations[-1] + dt / 2) * 1e9,
        amplitudes[0] - da / 2,
        amplitudes[-1] + da / 2,
    ]

    im = ax.imshow(
        data_2d,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
    )
    fig.colorbar(im, ax=ax, label="Transmon signal")

    ax.axhline(
        result["amp_swap"],
        color="k",
        linestyle="--",
        alpha=0.7,
    )
    ax.axvline(
        result["t_swap"] * 1e9,
        color="k",
        linestyle="--",
        alpha=0.7,
    )
    ax.plot(
        result["t_swap"] * 1e9,
        result["amp_swap"],
        "x",
        color="yellow",
        markersize=12,
        markeredgewidth=2,
        label=(
            f"Optimum: "
            f"t = {result['t_swap'] * 1e9:.2f} ns, "
            f"amp = {result['amp_swap']:.4f}"
        ),
    )
    ax.set_xlabel("SWAP duration (ns)")
    ax.set_ylabel("SWAP drive amplitude")
    ax.set_title(
        f"{q_uid} — SWAP chevron (2D)",
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right")
    _save_and_show(fig, f"swap_cal_chevron_{q_uid}", opts)
    return fig


def _save_and_show(
    fig: mpl.figure.Figure,
    name: str,
    opts: SwapAnalysisOptions,
) -> None:
    if opts.save_figures:
        fig.savefig(
            f"{opts.save_path}/{name}.png",
            dpi=150,
            bbox_inches="tight",
        )
    if opts.show_plot:
        plt.show()
    else:
        plt.close(fig)


# ═════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════


def _get_sweep_mode(
    swap_durations: QubitSweepPoints | None,
    swap_amplitudes: QubitSweepPoints | None,
) -> str:
    """Infer sweep mode from which inputs are provided."""
    if swap_durations is not None and swap_amplitudes is not None:
        return "2d"
    if swap_durations is not None:
        return "length"
    if swap_amplitudes is not None:
        return "amplitude"
    raise ValueError(
        "At least one of swap_durations or swap_amplitudes must be provided.",
    )


def _ensure_list(qubits: QuantumElements | list) -> list:
    if not isinstance(qubits, list):
        return [qubits]
    return qubits


def _broadcast(points: list | object, n_qubits: int) -> list:
    if not isinstance(points, list):
        points = [points]
    if len(points) == 1 and n_qubits > 1:
        points = points * n_qubits
    return points


def _swap_handle(qubit_uid: str) -> str:
    return f"{qubit_uid}/swap_cal"
