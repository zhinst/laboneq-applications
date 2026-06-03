# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Analysis for the displacement calibration experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.simple import dsl
from scipy.special import gammaln

from laboneq_applications.analysis.fitting_helpers import fit_data_lmfit

if TYPE_CHECKING:
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.task_options
class DisplacementCalibrationAnalysisOptions:
    """Options for the displacement calibration analysis.

    Attributes:
        generate_plot:
            Whether to generate plots. Default: True.
        show_plot:
            Whether to show plots in the terminal. Default: True.
        plot_raw_data:
            If True, plot the raw complex data (real and imaginary).
            If False, plot the magnitude. Default: False.
        save_figures:
            Whether to save the figures. Default: False.
        save_path:
            Path to save the figures. Default: "./"
        max_photon_number:
            Maximum photon number to consider for the distribution
            extraction and fitting. Default: 10.
        peak_window_width:
            Width of the frequency window (in Hz) around each expected
            peak position used for extracting peak heights. If None,
            it is set to 0.4 * |chi|.
        num_distributions_to_plot:
            Number of evenly-spaced amplitude slices to show as
            individual photon number distributions. Default: 5.
    """

    generate_plot: bool = True
    show_plot: bool = True
    plot_raw_data: bool = False
    save_figures: bool = False
    save_path: str = "./"
    max_photon_number: int = 10
    peak_window_width: float | None = None
    num_distributions_to_plot: int = 5


@workflow.workflow_options
class DisplacementCalibrationAnalysisWorkflowOptions:
    """Options for the displacement calibration analysis workflow."""


@workflow.workflow(name="displacement_calibration_analysis")
def analysis_workflow(
    result: RunExperimentResults,
    qpu: QPU,
    qubits: QuantumElements,
    *,
    amplitudes: QubitSweepPoints,
    frequencies: QubitSweepPoints,
    options: DisplacementCalibrationAnalysisWorkflowOptions | None = None,
) -> None:
    """The displacement calibration analysis workflow.

    Arguments:
        result:
            The result of the displacement calibration experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits the experiment was run on.
        amplitudes:
            The displacement amplitudes that were swept.
        frequencies:
            The transmon drive frequencies that were swept.
        options:
            The options for the analysis workflow.

    Returns:
        result:
            The result of the analysis workflow.
    """
    analyze(
        result,
        qpu,
        qubits,
        amplitudes=amplitudes,
        frequencies=frequencies,
    )


@workflow.task
def analyze(
    result: RunExperimentResults,
    qpu: QPU,
    qubits: QuantumElements,
    amplitudes: QubitSweepPoints,
    frequencies: QubitSweepPoints,
    options: DisplacementCalibrationAnalysisOptions | None = None,
) -> dict:
    """Analyze the displacement calibration results."""
    opts = DisplacementCalibrationAnalysisOptions() if options is None else options

    qubits = _ensure_list(qubits)
    amplitudes = _broadcast_sweep_points(amplitudes, len(qubits))
    frequencies = _broadcast_sweep_points(frequencies, len(qubits))

    analysis_results = {}
    for q, q_amps_raw, q_freqs_raw in zip(
        qubits, amplitudes, frequencies, strict=False
    ):
        q_amps = np.asarray(q_amps_raw)
        q_freqs = np.asarray(q_freqs_raw)

        handle = dsl.handles.result_handle(q.uid)
        data_raw = result.get_data(handle)
        data_2d = np.array(data_raw).reshape(len(q_amps), len(q_freqs))

        qubit_result = _analyze_single_qubit(q, data_2d, q_amps, q_freqs, opts)

        if opts.generate_plot:
            _generate_plots(qubit_result, q.uid, opts)

        analysis_results[q.uid] = qubit_result

    return analysis_results


def _ensure_list(qubits: QuantumElements) -> list:
    if not isinstance(qubits, list):
        return [qubits]
    return qubits


def _broadcast_sweep_points(
    points: QubitSweepPoints,
    n_qubits: int,
) -> list:
    """Ensure sweep points list matches qubit count."""
    if not isinstance(points, list):
        points = [points]
    if len(points) == 1 and n_qubits > 1:
        points = points * n_qubits
    return points


def _analyze_single_qubit(
    q: QuantumElements,
    data_2d: np.ndarray,
    q_amplitudes: np.ndarray,
    q_frequencies: np.ndarray,
    opts: DisplacementCalibrationAnalysisOptions,
) -> dict:
    """Run analysis for a single qubit."""
    chi = q.parameters.chi
    f_transmon_0 = q.parameters.transmon_resonance_frequency_at_n0

    if chi == 0:
        raise ValueError(
            f"Qubit {q.uid}: chi is 0. Cannot perform number-resolved "
            f"spectroscopy without a finite dispersive shift."
        )
    if f_transmon_0 is None:
        raise ValueError(
            f"Qubit {q.uid}: transmon_resonance_frequency_at_n0 is not set."
        )

    max_n = opts.max_photon_number
    photon_numbers = np.arange(max_n + 1)
    n_amplitudes = len(q_amplitudes)

    photon_distributions = np.zeros((n_amplitudes, max_n + 1))
    poisson_fits = np.zeros((n_amplitudes, max_n + 1))
    n_bar_fit = np.zeros(n_amplitudes)
    n_bar_direct = np.zeros(n_amplitudes)

    for i in range(n_amplitudes):
        _, weights = _extract_photon_distribution(
            spectrum=data_2d[i],
            frequencies=q_frequencies,
            f_transmon_0=f_transmon_0,
            chi=chi,
            max_photon_number=max_n,
            peak_window_width=opts.peak_window_width,
        )
        photon_distributions[i] = weights
        n_bar_i, fit_dist_i = _fit_coherent_state(photon_numbers, weights)
        n_bar_fit[i] = n_bar_i
        poisson_fits[i] = fit_dist_i
        total = np.sum(weights)
        if total > 0:
            n_bar_direct[i] = np.sum(photon_numbers * weights) / total

    # Extract displacement amplitude per unit beta
    displacement_amp_per_unit_beta = _fit_amp_per_unit_beta(q_amplitudes, n_bar_fit)

    return {
        "raw_data": data_2d,
        "amplitudes": q_amplitudes,
        "frequencies": q_frequencies,
        "photon_numbers": photon_numbers,
        "photon_distributions": photon_distributions,
        "n_bar_fit": n_bar_fit,
        "n_bar_direct": n_bar_direct,
        "poisson_fits": poisson_fits,
        "chi": chi,
        "f_transmon_0": f_transmon_0,
        "displacement_amp_per_unit_beta": displacement_amp_per_unit_beta,
        "figures": {},
    }


def _generate_plots(
    qubit_result: dict,
    q_uid: str,
    opts: DisplacementCalibrationAnalysisOptions,
) -> None:
    """Generate all plots for a single qubit."""
    qubit_result["figures"]["2d_map"] = _plot_2d_map(
        q_uid,
        qubit_result["raw_data"],
        qubit_result["amplitudes"],
        qubit_result["frequencies"],
        f_transmon_0=qubit_result["f_transmon_0"],
        chi=qubit_result["chi"],
        max_photon_number=opts.max_photon_number,
        plot_raw_data=opts.plot_raw_data,
    )
    qubit_result["figures"]["photon_distributions"] = _plot_photon_distributions(
        q_uid,
        qubit_result["photon_numbers"],
        qubit_result["photon_distributions"],
        qubit_result["poisson_fits"],
        qubit_result["n_bar_fit"],
        qubit_result["amplitudes"],
        num_to_plot=opts.num_distributions_to_plot,
    )
    qubit_result["figures"]["nbar_vs_amplitude"] = _plot_nbar_vs_amplitude(
        q_uid,
        qubit_result["amplitudes"],
        qubit_result["n_bar_fit"],
        qubit_result["n_bar_direct"],
    )
    if opts.save_figures:
        for name, fig in qubit_result["figures"].items():
            fig.savefig(
                f"{opts.save_path}/displacement_cal_{q_uid}_{name}.png",
                dpi=150,
                bbox_inches="tight",
            )
    if opts.show_plot:
        plt.show()
    else:
        for fig in qubit_result["figures"].values():
            plt.close(fig)


# =====================================================================
# Fitting helpers
# =====================================================================


def _poisson_pmf(x: np.ndarray, n_bar: float) -> np.ndarray:
    """Poisson probability mass function.

    P(n) = n_bar^n * exp(-n_bar) / n!

    Uses gammaln for numerical stability.

    Arguments:
        x: photon number array.
        n_bar: mean photon number.

    Returns:
        Poisson probabilities.
    """
    n_bar_safe = np.maximum(n_bar, 1e-15)
    return np.exp(x * np.log(n_bar_safe) - n_bar_safe - gammaln(x + 1))


def _extract_photon_distribution(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    f_transmon_0: float,
    chi: float,
    max_photon_number: int = 10,
    peak_window_width: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract photon number distribution from a single spectrum slice.

    For each photon number n, the expected peak is at:
        f_n = f_transmon_0 + n * chi

    The peak height is extracted as the maximum of |signal| in a
    frequency window around f_n, after baseline subtraction.

    Arguments:
        spectrum:
            1D complex array of the transmon spectrum.
        frequencies:
            1D frequency array corresponding to spectrum.
        f_transmon_0:
            Transmon frequency when memory is in |0>.
        chi:
            Dispersive shift (Hz). Sign convention:
            f_transmon(n) = f_transmon_0 + n * chi.
        max_photon_number:
            Maximum photon number to extract.
        peak_window_width:
            Width of the frequency window (Hz) around each peak.
            If None, defaults to 0.4 * |chi|.

    Returns:
        photon_numbers:
            Array [0, 1, ..., max_photon_number].
        weights:
            Extracted peak heights for each photon number.
    """
    signal = np.abs(spectrum)

    # Baseline subtraction
    baseline = np.median(signal)
    signal_corrected = np.maximum(signal - baseline, 0.0)

    if peak_window_width is None:
        peak_window_width = 0.4 * abs(chi)

    photon_numbers = np.arange(max_photon_number + 1)
    peak_positions = f_transmon_0 + photon_numbers * chi

    weights = np.zeros(max_photon_number + 1)
    for i, f_peak in enumerate(peak_positions):
        mask = np.abs(frequencies - f_peak) <= peak_window_width / 2
        if np.any(mask):
            weights[i] = np.max(signal_corrected[mask])

    return photon_numbers, weights


def _fit_coherent_state(
    photon_numbers: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Fit extracted photon number weights to a Poisson distribution.

    Arguments:
        photon_numbers:
            Array [0, 1, ..., N].
        weights:
            Extracted peak heights for each photon number.

    Returns:
        n_bar_fit:
            Fitted mean photon number.
        fit_distribution:
            Normalized Poisson distribution evaluated at the
            fitted n_bar.
    """
    total = np.sum(weights)
    if total <= 0:
        return 0.0, np.zeros_like(weights, dtype=float)

    probabilities = weights / total

    # Initial guess from direct mean
    n_bar_guess = float(np.sum(photon_numbers * probabilities))
    n_bar_guess = max(n_bar_guess, 0.01)

    try:
        fit_res = fit_data_lmfit(
            _poisson_pmf,
            photon_numbers,
            probabilities,
            param_hints={"n_bar": {"value": n_bar_guess, "min": 0.0}},
        )
        n_bar_fit = float(fit_res.best_values["n_bar"])
    except (RuntimeError, ValueError, TypeError):
        n_bar_fit = n_bar_guess

    fit_distribution = _poisson_pmf(photon_numbers, n_bar_fit)

    return n_bar_fit, fit_distribution


def _fit_amp_per_unit_beta(
    amplitudes: np.ndarray,
    n_bar_fit: np.ndarray,
) -> float:
    """Extract the displacement amplitude per unit beta.

    Fits amplitude = slope * |beta| where |beta| = sqrt(n_bar).

    Returns:
        displacement_amp_per_unit_beta: The linear slope.
    """
    beta = np.sqrt(n_bar_fit)

    # Only use points where n_bar > 0 (skip the origin issues)
    mask = beta > 0
    if np.sum(mask) < 2:  # noqa: PLR2004
        return np.nan

    # Fit a line through the origin: amplitude = slope * beta
    # Least squares: slope = sum(amp * beta) / sum(beta^2)
    return float(np.sum(amplitudes[mask] * beta[mask]) / np.sum(beta[mask] ** 2))


# =====================================================================
# Plotting helpers
# =====================================================================


def _plot_2d_map(
    qubit_uid: str,
    data_2d: np.ndarray,
    amplitudes: np.ndarray,
    frequencies: np.ndarray,
    f_transmon_0: float,
    chi: float,
    max_photon_number: int = 10,
    *,
    plot_raw_data: bool = False,
) -> plt.Figure:
    """Plot the 2D displacement calibration map.

    Arguments:
        qubit_uid: Qubit UID for labels.
        data_2d: 2D complex data (n_amplitudes x n_frequencies).
        amplitudes: Memory drive amplitude array.
        frequencies: Transmon drive frequency array.
        f_transmon_0: Transmon freq when memory in |0>.
        chi: Dispersive shift (Hz).
        max_photon_number: Max photon number for guide lines.
        plot_raw_data: If True, plot Re and Im; otherwise magnitude.

    Returns:
        fig: The matplotlib figure.
    """
    freq_ghz = frequencies / 1e9

    if plot_raw_data:
        fig, (ax_re, ax_im) = plt.subplots(
            1, 2, figsize=(14, 5), constrained_layout=True
        )

        im_re = ax_re.pcolormesh(
            freq_ghz,
            amplitudes,
            np.real(data_2d),
            shading="auto",
            cmap="RdBu_r",
        )
        ax_re.set_xlabel("Transmon drive frequency (GHz)")
        ax_re.set_ylabel("Memory drive amplitude")
        ax_re.set_title(f"{qubit_uid} — Real")
        fig.colorbar(im_re, ax=ax_re, label="Re(signal)")

        im_im = ax_im.pcolormesh(
            freq_ghz,
            amplitudes,
            np.imag(data_2d),
            shading="auto",
            cmap="RdBu_r",
        )
        ax_im.set_xlabel("Transmon drive frequency (GHz)")
        ax_im.set_ylabel("Memory drive amplitude")
        ax_im.set_title(f"{qubit_uid} — Imaginary")
        fig.colorbar(im_im, ax=ax_im, label="Im(signal)")

        for ax in (ax_re, ax_im):
            _add_photon_number_guides(
                ax, f_transmon_0, chi, max_photon_number, freq_ghz
            )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

        im = ax.pcolormesh(
            freq_ghz,
            amplitudes,
            np.abs(data_2d),
            shading="auto",
            cmap="viridis",
        )
        ax.set_xlabel("Transmon drive frequency (GHz)")
        ax.set_ylabel("Memory drive amplitude")
        ax.set_title(
            f"{qubit_uid} — Number-split transmon spectrum\nvs. displacement amplitude"
        )
        fig.colorbar(im, ax=ax, label="|signal|")
        _add_photon_number_guides(ax, f_transmon_0, chi, max_photon_number, freq_ghz)

    return fig


def _add_photon_number_guides(
    ax: plt.Axes,
    f_transmon_0: float,
    chi: float,
    max_photon_number: int,
    freq_ghz: np.ndarray,
) -> None:
    """Add vertical dashed lines at expected photon number peak positions.

    Arguments:
        ax: Matplotlib axes.
        f_transmon_0: Transmon frequency at n=0 (Hz).
        chi: Dispersive shift (Hz).
        max_photon_number: Max photon number to show.
        freq_ghz: Frequency array in GHz (for range checking).
    """
    freq_min, freq_max = freq_ghz.min(), freq_ghz.max()
    for n in range(max_photon_number + 1):
        f_n_ghz = (f_transmon_0 + n * chi) / 1e9
        if freq_min <= f_n_ghz <= freq_max:
            ax.axvline(
                f_n_ghz,
                color="white",
                linestyle="--",
                alpha=0.5,
                linewidth=0.8,
            )
            ax.text(
                f_n_ghz,
                ax.get_ylim()[1],
                f" n={n}",
                color="white",
                fontsize=7,
                va="top",
                ha="left",
                alpha=0.7,
            )


def _plot_photon_distributions(
    qubit_uid: str,
    photon_numbers: np.ndarray,
    photon_distributions: np.ndarray,
    poisson_fits: np.ndarray,
    n_bar_fit: np.ndarray,
    amplitudes: np.ndarray,
    num_to_plot: int = 5,
) -> plt.Figure:
    """Plot photon number distributions with Poisson fits.

    Selects evenly-spaced amplitude slices and shows bar plots
    of the extracted distribution overlaid with the Poisson fit.

    Arguments:
        qubit_uid: Qubit UID for labels.
        photon_numbers: Array [0, 1, ..., N].
        photon_distributions: Shape (n_amplitudes, N+1).
        poisson_fits: Shape (n_amplitudes, N+1).
        n_bar_fit: Fitted n_bar per amplitude.
        amplitudes: Amplitude array.
        num_to_plot: Number of slices to plot.

    Returns:
        fig: The matplotlib figure.
    """
    n_amplitudes = len(amplitudes)
    num_to_plot = min(num_to_plot, n_amplitudes)
    indices = np.linspace(0, n_amplitudes - 1, num_to_plot, dtype=int)

    ncols = min(num_to_plot, 3)
    nrows = int(np.ceil(num_to_plot / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    for panel_idx, amp_idx in enumerate(indices):
        row = panel_idx // ncols
        col = panel_idx % ncols
        ax = axes[row, col]

        weights = photon_distributions[amp_idx]
        total = np.sum(weights)
        probs = weights / total if total > 0 else weights

        fit = poisson_fits[amp_idx]

        ax.bar(
            photon_numbers,
            probs,
            width=0.6,
            alpha=0.6,
            color="steelblue",
            label="Data",
        )
        ax.plot(
            photon_numbers,
            fit,
            "o-",
            color="crimson",
            markersize=4,
            linewidth=1.5,
            label=f"Poisson ($\\bar{{n}}$={n_bar_fit[amp_idx]:.2f})",
        )
        ax.set_xlabel("Photon number $n$")
        ax.set_ylabel("$P(n)$")
        ax.set_title(f"Amp = {amplitudes[amp_idx]:.3f}")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(-0.5, photon_numbers[-1] + 0.5)
        ax.set_ylim(0, None)

    # Hide unused axes
    for panel_idx in range(num_to_plot, nrows * ncols):
        row = panel_idx // ncols
        col = panel_idx % ncols
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"{qubit_uid} — Photon number distributions with Poisson fits",
        fontsize=12,
        fontweight="bold",
    )

    return fig


def _plot_nbar_vs_amplitude(
    qubit_uid: str,
    amplitudes: np.ndarray,
    n_bar_fit: np.ndarray,
    n_bar_direct: np.ndarray,
) -> plt.Figure:
    """Plot mean photon number vs displacement amplitude.

    Shows both the Poisson-fitted n_bar and the directly computed
    n_bar from the distribution mean.

    Arguments:
        qubit_uid: Qubit UID for labels.
        amplitudes: Amplitude array.
        n_bar_fit: Fitted n_bar per amplitude.
        n_bar_direct: Directly computed n_bar per amplitude.

    Returns:
        fig: The matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)

    ax.plot(
        amplitudes,
        n_bar_fit,
        "o-",
        color="crimson",
        markersize=4,
        linewidth=1.5,
        label="Poisson fit $\\bar{n}$",
    )
    ax.plot(
        amplitudes,
        n_bar_direct,
        "s--",
        color="steelblue",
        markersize=4,
        linewidth=1.0,
        alpha=0.7,
        label="Direct mean $\\bar{n}$",
    )

    ax.set_xlabel("Memory drive amplitude")
    ax.set_ylabel("Mean photon number $\\bar{n}$")
    ax.set_title(f"{qubit_uid} — Mean photon number vs. displacement amplitude")
    ax.legend()
    ax.set_xlim(amplitudes[0], amplitudes[-1])
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)

    return fig
