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

from laboneq_applications.analysis.fitting_helpers import fit_data_lmfit
from laboneq_applications.analysis.options import BasePlottingOptions
from laboneq_applications.conversion import (
    as_quantum_element_list,
    sweeps_per_quantum_element,
)

if TYPE_CHECKING:
    import matplotlib as mpl
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


@workflow.workflow_options
class SwapAnalysisWorkflowOptions:
    """Options for the SWAP calibration analysis workflow.

    Attributes:
        do_plotting:
            Whether to create plots.
            Default: True.
    """

    do_plotting: bool = workflow.option_field(
        True, description="Whether to create plots."
    )


@workflow.task_options
class SwapFitDataOptions:
    """Options for the `fit_data` task of the SWAP calibration analysis.

    Attributes:
        num_fit_points:
            Number of points for the fit curve. Default: 400.
    """

    num_fit_points: int = workflow.option_field(
        400, description="Number of points for the fit curve."
    )


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    swap_durations: QubitSweepPoints | None = None,
    swap_amplitudes: QubitSweepPoints | None = None,
    options: SwapAnalysisWorkflowOptions | None = None,
) -> None:
    """The SWAP calibration analysis workflow.

    The workflow consists of the following steps:

    - [fit_data][]
    - [extract_qubit_parameters][]
    - [plot_swap_calibration][]

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        swap_durations:
            The SWAP durations that were swept for each qubit, if any.
        swap_amplitudes:
            The SWAP amplitudes that were swept for each qubit, if any.
        options:
            The options for building the workflow, passed as an instance of
            [SwapAnalysisWorkflowOptions][].

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        analysis_result = analysis_workflow(
            result=result,
            qubits=q0,
            swap_durations=np.linspace(100e-9, 600e-9, 51),
        ).run()
        ```
    """
    processed_data_dict = fit_data(qubits, result, swap_durations, swap_amplitudes)
    qubit_parameters = extract_qubit_parameters(qubits, processed_data_dict)
    with workflow.if_(options.do_plotting):
        plot_swap_calibration(qubits, processed_data_dict)
    workflow.return_(qubit_parameters)


# ═════════════════════════════════════════════════════════════
# Analysis tasks
# ═════════════════════════════════════════════════════════════


@workflow.task
def fit_data(
    qubits: QuantumElements,
    result: RunExperimentResults,
    swap_durations: QubitSweepPoints | None = None,
    swap_amplitudes: QubitSweepPoints | None = None,
    options: SwapFitDataOptions | None = None,
) -> dict[str, dict]:
    """Fit the SWAP calibration data.

    Mode is inferred from which inputs are provided:
        - ``swap_durations`` only → length sweep.
        - ``swap_amplitudes`` only → amplitude sweep.
        - Both → 2D chevron sweep.

    Arguments:
        qubits:
            The qubits on which to run the analysis.
        result:
            The experiment results returned by the run_experiment task.
        swap_durations:
            The SWAP durations that were swept for each qubit, if any.
        swap_amplitudes:
            The SWAP amplitudes that were swept for each qubit, if any.
        options:
            The options for this task as an instance of [SwapFitDataOptions][].

    Returns:
        Dictionary keyed by qubit UID. Structure depends on
            mode — each dict always contains the key ``"mode"`` and
            the calibrated ``"t_swap"`` and/or ``"amp_swap"``.
    """
    opts = SwapFitDataOptions() if options is None else options

    mode = _get_sweep_mode(swap_durations, swap_amplitudes)
    qubits = as_quantum_element_list(qubits)
    if swap_durations is not None:
        swap_durations = [
            d for _, d in sweeps_per_quantum_element(qubits, swap_durations)
        ]
    if swap_amplitudes is not None:
        swap_amplitudes = [
            a for _, a in sweeps_per_quantum_element(qubits, swap_amplitudes)
        ]

    processed_data_dict = {}

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
            processed_data_dict[q.uid] = {
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
            processed_data_dict[q.uid] = {
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
            # The durations are the outer sweep, so the data comes back as
            # (n_durations, n_amps). Transpose to (n_amps, n_durations).
            data_2d = data.reshape(len(q_durs), len(q_amps)).T
            chevron = _fit_chevron(
                q_amps,
                q_durs,
                data_2d,
                opts.num_fit_points,
            )
            processed_data_dict[q.uid] = {
                "mode": "2d",
                "t_swap": float(chevron["t_swap"]),
                "amp_swap": float(chevron["amp_swap"]),
                "raw_data": data_2d,
                "amplitudes": q_amps,
                "durations": q_durs,
                "row_fits": chevron["row_fits"],
                "frequencies_per_amp": chevron["frequencies"],
                "fit_success": bool(chevron["success"]),
            }

    return processed_data_dict


@workflow.task
def extract_qubit_parameters(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict],
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Extract the SWAP length and/or amplitude from the fit results.

    Arguments:
        qubits:
            The qubits on which to run the analysis.
        processed_data_dict:
            The processed data dictionary returned by fit_data.

    Returns:
        dict with the extracted qubit parameters and their previous values:

            ```python
            {
                "new_parameter_values": {
                    q.uid: {"swap_length": value, "swap_amplitude": value},
                },
                "old_parameter_values": {
                    q.uid: {"swap_length": value, "swap_amplitude": value},
                },
            }
            ```

            Only the parameters calibrated by the sweep mode are included
            (`swap_length` for a length sweep, `swap_amplitude` for an amplitude
            sweep, both for a 2D sweep). Parameters whose fit failed are left out
            of new_parameter_values.
    """
    qubits = as_quantum_element_list(qubits)
    qubit_parameters = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
    }
    for q in qubits:
        res = processed_data_dict[q.uid]
        fit_ok = res.get("fit_success", True)
        candidates = {}
        if res["mode"] in ("length", "2d"):
            candidates["swap_length"] = res["t_swap"]
        if res["mode"] in ("amplitude", "2d"):
            candidates["swap_amplitude"] = res["amp_swap"]
        for name, value in candidates.items():
            qubit_parameters["old_parameter_values"][q.uid][name] = getattr(
                q.parameters, name
            )
            if fit_ok and np.isfinite(value):
                qubit_parameters["new_parameter_values"][q.uid][name] = value
    return qubit_parameters


@workflow.task
def plot_swap_calibration(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict],
    options: BasePlottingOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot the SWAP calibration data and fit for the sweep mode that was run.

    Arguments:
        qubits:
            The qubits on which to run the analysis.
        processed_data_dict:
            The processed data dictionary returned by fit_data.
        options:
            The options for this task as an instance of [BasePlottingOptions][].

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = BasePlottingOptions() if options is None else options
    qubits = as_quantum_element_list(qubits)
    figures = {}
    for q in qubits:
        res = processed_data_dict[q.uid]
        if res["mode"] == "length":
            fig = _plot_1d_length(q.uid, res, opts)
        elif res["mode"] == "amplitude":
            fig = _plot_1d_amplitude(q.uid, res, opts)
        else:
            fig = _plot_2d_chevron(q.uid, res, opts)
        figures[q.uid] = fig
    return figures


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
    fit_x = np.linspace(x[0], x[-1], num_fit_points)
    try:
        _validate_sweep(x)
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

        param_hints = {
            "amplitude": {"value": amp_guess, "min": 0},
            "frequency": {"value": freq_guess, "min": 0},
            "phase": {"value": 0.0, "min": -2 * np.pi, "max": 2 * np.pi},
            # A decay shorter than the sweep step is not resolvable; the
            # positive bound also keeps the model from dividing by zero.
            "decay": {"value": decay_guess, "min": abs(dx) / 10},
            "offset": {"value": offset_guess},
        }
        fit_res = fit_data_lmfit(
            _damped_cosine_model,
            x,
            data,
            param_hints=param_hints,
        )
        popt = np.array(
            [
                fit_res.best_values[k]
                for k in ("amplitude", "frequency", "phase", "decay", "offset")
            ]
        )
        success = bool(fit_res.success)
        fit_y = _damped_cosine_model(fit_x, *popt)
    except (RuntimeError, ValueError, TypeError):
        popt = np.array([0.0, np.nan, 0.0, np.inf, float(np.mean(data))])
        success = False
        fit_y = np.full_like(fit_x, np.nan)

    amplitude, frequency, phase, _decay, _offset = popt
    x_min = _first_minimum(amplitude, frequency, phase)

    return {
        "params": tuple(popt),
        "success": success,
        "x_min": float(x_min),
        "frequency": float(frequency),
        "fit_x": fit_x,
        "fit_y": fit_y,
    }


def _validate_sweep(x: np.ndarray) -> None:
    """Check that the sweep supports the FFT guess and the fallback values.

    Raises:
        ValueError: If the first, second and last points are not distinct.
    """
    if len(x) < 2 or x[1] == x[0] or x[-1] == x[0]:  # noqa: PLR2004
        raise ValueError("The sweep needs at least two distinct points.")


def _first_minimum(amplitude: float, frequency: float, phase: float) -> float:
    """Return the first x > 0 at which the cosine A·cos(2π·f·x + φ) is minimal.

    This is the first point of maximum |e⟩ population transferred away,
    assuming the data starts near |e⟩.
    """
    target_arg = np.pi if amplitude >= 0 else 0.0
    x_min = (target_arg - phase) / (2 * np.pi * frequency)
    # Minima repeat every full period.
    period = 1.0 / frequency
    while x_min <= 0:
        x_min += period
    return float(x_min)


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

            - ``success``: whether at least one row fit succeeded.
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

    success = not np.all(np.isnan(frequencies))
    best_idx = int(np.nanargmax(frequencies)) if success else 0

    amp_swap = float(amplitudes[best_idx])
    t_swap = float(row_fits[best_idx]["x_min"])

    return {
        "success": success,
        "t_swap": t_swap,
        "amp_swap": amp_swap,
        "frequencies": frequencies,
        "row_fits": row_fits,
    }


# ═════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════


def _plot_1d_length(
    q_uid: str,
    result: dict,
    opts: BasePlottingOptions,
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
    _save_and_close(fig, f"swap_cal_length_{q_uid}", opts)
    return fig


def _plot_1d_amplitude(
    q_uid: str,
    result: dict,
    opts: BasePlottingOptions,
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
    _save_and_close(fig, f"swap_cal_amplitude_{q_uid}", opts)
    return fig


def _plot_2d_chevron(
    q_uid: str,
    result: dict,
    opts: BasePlottingOptions,
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
    _save_and_close(fig, f"swap_cal_chevron_{q_uid}", opts)
    return fig


def _save_and_close(
    fig: mpl.figure.Figure,
    name: str,
    opts: BasePlottingOptions,
) -> None:
    if opts.save_figures:
        workflow.save_artifact(name, fig)
    if opts.close_figures:
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


def _swap_handle(qubit_uid: str) -> str:
    return f"{qubit_uid}/swap_cal"
