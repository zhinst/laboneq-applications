# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Analysis for the selective transmon Rx calibration experiment.

Fits Rabi oscillations for each (qubit, photon number) pair and
extracts calibrated pi and pi/2 pulse amplitudes.

The Rabi oscillation model is:

    signal(x) = offset + amplitude * cos(2π * freq * x + phase)

From the fitted frequency:
    amp_pi  = 1 / (2 * |freq|)
    amp_pi/2 = 1 / (4 * |freq|)

For photon numbers n > 0, the Rabi oscillation contrast is
reduced by the population P(n|beta) of the prepared coherent state
at photon number n. This does not affect the oscillation period,
so the extracted pi amplitude remains accurate.
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
class SelectiveTransmonRxAnalysisWorkflowOptions:
    """Options for the selective transmon Rx analysis workflow.

    Attributes:
        do_plotting:
            Whether to create plots.
            Default: True.
    """

    do_plotting: bool = workflow.option_field(
        True, description="Whether to create plots."
    )


@workflow.task_options
class SelectiveTransmonRxFitDataOptions:
    """Options for the `fit_data` task of the selective transmon Rx analysis.

    Attributes:
        num_fit_points:
            Number of points for the smooth fit curve in plots.
            Default: 201.
    """

    num_fit_points: int = workflow.option_field(
        201, description="Number of points for the smooth fit curve."
    )


# ═════════════════════════════════════════════════════════════
# Workflow
# ═════════════════════════════════════════════════════════════


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    amplitudes_transmon: QubitSweepPoints,
    photon_numbers_memory: list[int],
    options: SelectiveTransmonRxAnalysisWorkflowOptions | None = None,
) -> None:
    """The selective transmon Rx analysis workflow.

    The workflow consists of the following steps:

    - [fit_data][]
    - [extract_qubit_parameters][]
    - [plot_rabi_oscillations][]
    - [plot_amplitudes_vs_photon_number][]

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        amplitudes_transmon:
            The selective drive amplitudes that were swept for each qubit.
        photon_numbers_memory:
            List of cavity photon numbers that were calibrated.
        options:
            The options for building the workflow, passed as an instance of
            [SelectiveTransmonRxAnalysisWorkflowOptions][].

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        analysis_result = analysis_workflow(
            result=result,
            qubits=q0,
            amplitudes_transmon=np.linspace(0.0, 1.0, 51),
            photon_numbers_memory=[0, 1, 2, 3],
        ).run()
        ```
    """
    processed_data_dict = fit_data(
        qubits, result, amplitudes_transmon, photon_numbers_memory
    )
    qubit_parameters = extract_qubit_parameters(qubits, processed_data_dict)
    with workflow.if_(options.do_plotting):
        plot_rabi_oscillations(qubits, processed_data_dict)
        plot_amplitudes_vs_photon_number(qubits, processed_data_dict)
    workflow.return_(qubit_parameters)


# ═════════════════════════════════════════════════════════════
# Analysis tasks
# ═════════════════════════════════════════════════════════════


@workflow.task
def fit_data(
    qubits: QuantumElements,
    result: RunExperimentResults,
    amplitudes_transmon: QubitSweepPoints,
    photon_numbers_memory: list[int],
    options: SelectiveTransmonRxFitDataOptions | None = None,
) -> dict[str, dict]:
    """Fit the selective Rabi oscillations.

    For each (qubit, photon number) pair:

    1. Extracts the Rabi oscillation data.
    2. Fits a cosine model to determine the oscillation period.
    3. Extracts the pi and pi/2 pulse amplitudes.

    Arguments:
        qubits:
            The qubits on which to run the analysis.
        result:
            The experiment results returned by the run_experiment task.
        amplitudes_transmon:
            The selective drive amplitudes that were swept for each qubit.
        photon_numbers_memory:
            List of cavity photon numbers that were calibrated.
        options:
            The options for this task as an instance of
            [SelectiveTransmonRxFitDataOptions][].

    Returns:
        Dictionary keyed by qubit UID. Each entry contains:

            - ``"amplitudes"``: The amplitude sweep array.
            - ``"photon_numbers"``: The photon numbers that were calibrated.
            - ``"rabi_data"``: Dict keyed by ``str(n)``, each containing:
              ``"raw_data"``, ``"fit_params"``, ``"fit_success"``,
              ``"amp_pi"``, ``"amp_pi2"``, ``"fit_x"``, ``"fit_y"``.
              ``"fit_params"`` is empty if the fit failed.
            - ``"amp_pi_vs_n"``: Array of π amplitudes, one per photon number.
            - ``"amp_pi2_vs_n"``: Array of π/2 amplitudes, one per photon number.
    """
    opts = SelectiveTransmonRxFitDataOptions() if options is None else options
    qubits = as_quantum_element_list(qubits)

    processed_data_dict = {}
    for q, q_amplitudes_raw in sweeps_per_quantum_element(qubits, amplitudes_transmon):
        processed_data_dict[q.uid] = _analyze_single_qubit(
            q,
            result,
            np.asarray(q_amplitudes_raw),
            photon_numbers_memory,
            opts,
        )
    return processed_data_dict


@workflow.task
def extract_qubit_parameters(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict],
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Extract the selective X180 and X90 amplitudes from the fit results.

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
                    q.uid: {
                        "selective_transmon_X180_amplitudes": [...],
                        "selective_transmon_X90_amplitudes": [...],
                    },
                },
                "old_parameter_values": {...},  # same form
            }
            ```

            The new amplitude lists are copies of the old ones with the entries
            for the successfully fitted photon numbers replaced. If no photon
            number was fitted successfully for a qubit, its new_parameter_values
            entry is left empty.
    """
    qubits = as_quantum_element_list(qubits)
    qubit_parameters = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
    }
    for q in qubits:
        old_x180 = list(q.parameters.selective_transmon_X180_amplitudes)
        old_x90 = list(q.parameters.selective_transmon_X90_amplitudes)
        qubit_parameters["old_parameter_values"][q.uid] = {
            "selective_transmon_X180_amplitudes": old_x180,
            "selective_transmon_X90_amplitudes": old_x90,
        }
        new_x180, new_x90 = list(old_x180), list(old_x90)
        updated = False
        for n_str, rabi in processed_data_dict[q.uid]["rabi_data"].items():
            n = int(n_str)
            if not (rabi["fit_success"] and np.isfinite(rabi["amp_pi"])):
                continue
            new_x180[n] = rabi["amp_pi"]
            new_x90[n] = rabi["amp_pi2"]
            updated = True
        if updated:
            qubit_parameters["new_parameter_values"][q.uid] = {
                "selective_transmon_X180_amplitudes": new_x180,
                "selective_transmon_X90_amplitudes": new_x90,
            }
    return qubit_parameters


@workflow.task
def plot_rabi_oscillations(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict],
    options: BasePlottingOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot the selective Rabi oscillations and their fits per photon number.

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
        qubit_result = processed_data_dict[q.uid]
        fig = _plot_rabi_oscillations(
            q.uid,
            qubit_result["amplitudes"],
            qubit_result["rabi_data"],
            qubit_result["photon_numbers"],
        )
        _save_and_close(fig, f"selective_transmon_rx_{q.uid}_rabi_oscillations", opts)
        figures[q.uid] = fig
    return figures


@workflow.task
def plot_amplitudes_vs_photon_number(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict],
    options: BasePlottingOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot the fitted pi and pi/2 amplitudes vs. photon number.

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
        qubit_result = processed_data_dict[q.uid]
        fig = _plot_amplitude_summary(
            q.uid,
            qubit_result["photon_numbers"],
            qubit_result["amp_pi_vs_n"],
            qubit_result["amp_pi2_vs_n"],
        )
        _save_and_close(fig, f"selective_transmon_rx_{q.uid}_amplitude_summary", opts)
        figures[q.uid] = fig
    return figures


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
# Single-qubit analysis
# ═════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════
# Result-shape helper
# ═════════════════════════════════════════════════════════════


def _extract_1d_trace(
    data_raw: np.ndarray,
    n: int,
    photon_numbers_memory: list[int],
    expected_len: int,
) -> np.ndarray:
    """Reduce a result array to the 1-D amplitude trace.

    Handles three layouts produced by different sweep structures:

    1. **1-D array** of length ``expected_len`` — the legacy layout
       when the photon number was a Python loop. Returned as-is.
    2. **2-D array** of shape ``(len(photon_numbers_memory),
       expected_len)`` — produced when the photon number is a real
       ``dsl.sweep`` axis with ``dsl.match`` / ``dsl.case``. Only the
       row whose sweep index equals ``n`` is populated; the rest are
       NaN. We pick that row by its index in ``photon_numbers_memory``.
    3. **2-D array with one fully-NaN row per inactive case.** Same as
       (2); selecting by index already handles this.

    Arguments:
        data_raw: The array returned by ``result.get_data(handle)``.
        n: The photon number this handle corresponds to.
        photon_numbers_memory: The list of photon numbers swept,
            in sweep order.
        expected_len: Expected length of the amplitude axis.

    Returns:
        A 1-D ``np.ndarray`` of length ``expected_len``.
    """
    data = np.real(np.asarray(data_raw))

    if data.ndim == 1:
        return data

    if data.ndim == 2:  # noqa: PLR2004
        # Case where photon number is the leading sweep axis.
        if (
            data.shape[0] == len(photon_numbers_memory)
            and data.shape[1] == expected_len
        ):
            n_index = list(photon_numbers_memory).index(n)
            return data[n_index]
        # Case where the axes happen to be transposed.
        if (
            data.shape[1] == len(photon_numbers_memory)
            and data.shape[0] == expected_len
        ):
            n_index = list(photon_numbers_memory).index(n)
            return data[:, n_index]
        # Fallback: drop fully-NaN rows/cols and squeeze.
        squeezed = data.squeeze()
        if squeezed.ndim == 1:
            return squeezed
        # Last resort: keep the row with the most finite values.
        finite_per_row = np.sum(np.isfinite(data), axis=1)
        return data[int(np.argmax(finite_per_row))]

    # Higher-dim: squeeze singletons and recurse on the result.
    squeezed = data.squeeze()
    if squeezed.ndim == 1:
        return squeezed
    raise ValueError(
        f"Unexpected result shape {data.shape} for n={n}; "
        f"expected 1-D of length {expected_len} or 2-D with one axis "
        f"of length {len(photon_numbers_memory)}."
    )


def _analyze_single_qubit(
    q: QuantumElements,
    result: RunExperimentResults,
    q_amplitudes_transmon: np.ndarray,
    photon_numbers_memory: list[int],
    opts: SelectiveTransmonRxFitDataOptions,
) -> dict:
    """Run Rabi analysis for all photon numbers of a single qubit.

    Arguments:
        q: The qubit object.
        result: Experiment results.
        q_amplitudes_transmon: The amplitude sweep array.
        photon_numbers_memory: Photon numbers to analyze.
        opts: Analysis options.

    Returns:
        Dictionary with per-n Rabi data and summary arrays.
    """
    rabi_data = {}
    amp_pi_list = []
    amp_pi2_list = []

    expected_len = len(q_amplitudes_transmon)

    for n in photon_numbers_memory:
        handle = rabi_result_handle(q.uid, n)
        data_raw = result.get_data(handle)
        data = _extract_1d_trace(
            data_raw,
            n=n,
            photon_numbers_memory=photon_numbers_memory,
            expected_len=expected_len,
        )

        fit_result = _fit_rabi(
            q_amplitudes_transmon,
            data,
            opts.num_fit_points,
        )

        rabi_data[n] = {
            "raw_data": data,
            "fit_params": fit_result["params"],
            "fit_success": fit_result["success"],
            "amp_pi": fit_result["amp_pi"],
            "amp_pi2": fit_result["amp_pi2"],
            "fit_x": fit_result["fit_x"],
            "fit_y": fit_result["fit_y"],
        }

        amp_pi_list.append(fit_result["amp_pi"])
        amp_pi2_list.append(fit_result["amp_pi2"])

    return {
        "amplitudes": q_amplitudes_transmon,
        "photon_numbers": photon_numbers_memory,
        "rabi_data": {
            str(n): {
                "raw_data": v["raw_data"],
                "fit_params": (
                    list(v["fit_params"]) if v["fit_params"] is not None else []
                ),
                "fit_success": bool(v["fit_success"]),
                "amp_pi": float(v["amp_pi"]),
                "amp_pi2": float(v["amp_pi2"]),
                "fit_x": v["fit_x"],
                "fit_y": v["fit_y"],
            }
            for n, v in rabi_data.items()
        },
        "amp_pi_vs_n": np.array(amp_pi_list),
        "amp_pi2_vs_n": np.array(amp_pi2_list),
    }


# ═════════════════════════════════════════════════════════════
# Fitting
# ═════════════════════════════════════════════════════════════


def _rabi_model(
    x: np.ndarray,
    amplitude: float,
    frequency: float,
    phase: float,
    offset: float,
) -> np.ndarray:
    """Cosine model for Rabi oscillation.

    signal(x) = offset + amplitude * cos(2* pi * frequency * x + phase)

    Arguments:
        x: Drive amplitude array.
        amplitude: Oscillation amplitude.
        frequency: Oscillation frequency (1/amp units).
        phase: Phase offset (radians).
        offset: Vertical offset.

    Returns:
        Model values at each x.
    """
    return offset + amplitude * np.cos(2 * np.pi * frequency * x + phase)


def _estimate_frequency(
    x: np.ndarray,
    data: np.ndarray,
) -> float:
    """Estimate Rabi oscillation frequency from FFT.

    Arguments:
        x: Drive amplitude array (uniformly spaced).
        data: Measured signal.

    Returns:
        Estimated frequency in 1/amplitude units.
    """
    n = len(data)
    if n < 4:  # noqa: PLR2004
        return 1.0 / (x[-1] - x[0]) if x[-1] != x[0] else 1.0

    dx = np.mean(np.diff(x))
    fft = np.fft.rfft(data - np.mean(data))
    freqs = np.fft.rfftfreq(n, d=dx)

    # Skip DC component (index 0)
    magnitudes = np.abs(fft[1:])
    if len(magnitudes) == 0:
        return 1.0 / (x[-1] - x[0]) if x[-1] != x[0] else 1.0

    idx_peak = np.argmax(magnitudes) + 1
    return float(freqs[idx_peak])


def _fit_rabi(
    x: np.ndarray,
    data: np.ndarray,
    num_fit_points: int = 201,
) -> dict:
    """Fit a Rabi oscillation and extract pi/pi2 amplitudes.

    Fits the data to:
        signal(x) = offset + amplitude * cos(2 * pi * freq * x + phase)

    Extracts:
        amp_pi  = 1 / (2 * |freq|)
        amp_pi2 = 1 / (4 * |freq|)

    Arguments:
        x: Drive amplitude array.
        data: Measured signal.
        num_fit_points: Number of points for smooth fit curve.

    Returns:
        Dictionary containing:
            "params": fitted (amplitude, frequency, phase, offset)
                or None if fit failed.
            "success": whether the fit succeeded.
            "amp_pi": extracted pi amplitude (NaN if fit failed).
            "amp_pi2": extracted pi/2 amplitude (NaN if fit failed).
            "fit_x": smooth x array for plotting.
            "fit_y": smooth fit curve for plotting.
    """
    fit_x = np.linspace(x[0], x[-1], num_fit_points)

    # Initial guesses
    offset_guess = np.mean(data)
    amplitude_guess = (np.max(data) - np.min(data)) / 2.0
    freq_guess = _estimate_frequency(x, data)
    phase_guess = 0.0

    # Determine sign of amplitude from whether data starts
    # high (P(g), amplitude > 0) or low (P(e), amplitude < 0)  # noqa: ERA001
    if data[0] < offset_guess:
        amplitude_guess = -amplitude_guess

    param_hints = {
        "amplitude": {"value": amplitude_guess},
        "frequency": {"value": freq_guess},
        "phase": {"value": phase_guess},
        "offset": {"value": offset_guess},
    }

    try:
        fit_res = fit_data_lmfit(
            _rabi_model,
            x,
            data,
            param_hints=param_hints,
        )
        popt = np.array(
            [
                fit_res.best_values[k]
                for k in ("amplitude", "frequency", "phase", "offset")
            ]
        )

        freq_fit = abs(popt[1])
        if freq_fit > 0:
            amp_pi = 1.0 / (2.0 * freq_fit)
            amp_pi2 = 1.0 / (4.0 * freq_fit)
        else:
            amp_pi = np.nan
            amp_pi2 = np.nan

        fit_y = _rabi_model(fit_x, *popt)

        return {
            "params": tuple(popt),
            "success": bool(fit_res.success),
            "amp_pi": amp_pi,
            "amp_pi2": amp_pi2,
            "fit_x": fit_x,
            "fit_y": fit_y,
        }

    except (RuntimeError, ValueError, TypeError):
        return {
            "params": None,
            "success": False,
            "amp_pi": np.nan,
            "amp_pi2": np.nan,
            "fit_x": fit_x,
            "fit_y": np.full_like(fit_x, np.nan),
        }


# ═════════════════════════════════════════════════════════════
# Plotting helpers
# ═════════════════════════════════════════════════════════════


def _plot_rabi_oscillations(
    qubit_uid: str,
    amplitudes: np.ndarray,
    rabi_data: dict,
    photon_numbers: list[int],
) -> plt.Figure:
    """Plot Rabi oscillations for all photon numbers.

    Each photon number gets its own subplot showing the raw data
    as scatter points and the cosine fit as a smooth curve.
    Vertical dashed lines mark the extracted π and π/2
    amplitudes.

    Arguments:
        qubit_uid: Qubit UID for the plot title.
        amplitudes: The amplitude sweep array.
        rabi_data: Dict keyed by photon number n.
        photon_numbers: Photon numbers to plot.

    Returns:
        fig: The matplotlib figure.
    """
    n_panels = len(photon_numbers)
    ncols = min(n_panels, 3)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    for idx, n in enumerate(photon_numbers):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        rd = rabi_data[str(n)]
        data = rd["raw_data"]

        # Raw data
        ax.plot(
            amplitudes,
            data,
            "o",
            color="steelblue",
            markersize=3,
            alpha=0.7,
            label="Data",
        )

        # Fit curve
        if rd["fit_success"]:
            ax.plot(
                rd["fit_x"],
                rd["fit_y"],
                "-",
                color="crimson",
                linewidth=1.5,
                label="Fit",
            )

            # Mark amp_pi
            amp_pi = rd["amp_pi"]
            if np.isfinite(amp_pi) and amplitudes[0] <= amp_pi <= amplitudes[-1]:
                ax.axvline(
                    amp_pi,
                    color="green",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.8,
                    label=rf"$A_{{\pi}}$ = {amp_pi:.4f}",
                )

            # Mark amp_pi2
            amp_pi2 = rd["amp_pi2"]
            if np.isfinite(amp_pi2) and amplitudes[0] <= amp_pi2 <= amplitudes[-1]:
                ax.axvline(
                    amp_pi2,
                    color="orange",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.8,
                    label=rf"$A_{{\pi/2}}$ = {amp_pi2:.4f}",
                )
        else:
            ax.text(
                0.5,
                0.5,
                "Fit failed",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="red",
            )

        ax.set_xlabel("Selective drive amplitude")
        ax.set_ylabel("Signal")
        ax.set_title(f"n = {n}")
        ax.legend(fontsize=7, loc="upper right")

    # Hide unused axes
    for idx in range(n_panels, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"{qubit_uid} — Selective Rabi Oscillations",
        fontsize=13,
        fontweight="bold",
    )

    return fig


def _plot_amplitude_summary(
    qubit_uid: str,
    photon_numbers: list[int],
    amp_pi_vs_n: np.ndarray,
    amp_pi2_vs_n: np.ndarray,
) -> plt.Figure:
    """Plot π and π/2 amplitudes vs photon number.

    Arguments:
        qubit_uid: Qubit UID for the plot title.
        photon_numbers: Photon numbers.
        amp_pi_vs_n: Fitted π amplitude per n.
        amp_pi2_vs_n: Fitted π/2 amplitude per n.

    Returns:
        fig: The matplotlib figure.
    """
    fig, ax = plt.subplots(
        figsize=(7, 5),
        constrained_layout=True,
    )

    ns = np.array(photon_numbers)

    ax.plot(
        ns,
        amp_pi_vs_n,
        "o-",
        color="crimson",
        markersize=7,
        linewidth=1.5,
        label=r"$A_{\pi}$",
    )
    ax.plot(
        ns,
        amp_pi2_vs_n,
        "s--",
        color="steelblue",
        markersize=6,
        linewidth=1.5,
        label=r"$A_{\pi/2}$",
    )

    # Annotate values
    for i, n in enumerate(photon_numbers):
        if np.isfinite(amp_pi_vs_n[i]):
            ax.annotate(
                f"{amp_pi_vs_n[i]:.4f}",
                (n, amp_pi_vs_n[i]),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=7,
                color="crimson",
            )
        if np.isfinite(amp_pi2_vs_n[i]):
            ax.annotate(
                f"{amp_pi2_vs_n[i]:.4f}",
                (n, amp_pi2_vs_n[i]),
                textcoords="offset points",
                xytext=(8, -10),
                fontsize=7,
                color="steelblue",
            )

    ax.set_xlabel("Photon number $n$")
    ax.set_ylabel("Calibrated amplitude")
    ax.set_title(f"{qubit_uid} — Selective pulse amplitudes vs. photon number")
    ax.set_xticks(ns)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(ns[0] - 0.5, ns[-1] + 0.5)
    ax.set_ylim(0, None)

    return fig


# ═════════════════════════════════════════════════════════════
# Handles
# ═════════════════════════════════════════════════════════════


def rabi_result_handle(qubit_uid: str, n: int) -> str:
    """Return the result handle for a selective Rabi measurement.

    Arguments:
        qubit_uid: The qubit UID.
        n: The photon number.

    Returns:
        The result handle string.
    """
    return f"{qubit_uid}/rabi_n{n}"
