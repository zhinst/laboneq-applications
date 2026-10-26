# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Analysis for the displacement calibration experiment.

For each displacement amplitude, the photon-number distribution is extracted from
the number-split transmon spectrum and fitted to a Poisson distribution. A line
fit of the amplitude vs. sqrt(n_bar) then gives `displacement_amp_per_unit_beta`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.simple import dsl
from scipy.stats import poisson

from laboneq_applications.analysis.fitting_helpers import fit_data_lmfit
from laboneq_applications.analysis.options import BasePlottingOptions
from laboneq_applications.conversion import (
    as_quantum_element_list,
    sweeps_per_quantum_element,
)

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.workflow_options
class DisplacementCalibrationAnalysisWorkflowOptions:
    """Options for the displacement calibration analysis workflow.

    Attributes:
        do_plotting:
            Whether to create plots.
            Default: True.
    """

    do_plotting: bool = workflow.option_field(
        True, description="Whether to create plots."
    )


@workflow.task_options
class DisplacementCalibrationFitDataOptions:
    """Options for the `fit_data` task of the displacement calibration analysis.

    Attributes:
        max_photon_number:
            Maximum photon number to consider for the distribution
            extraction and fitting. If None, it is the highest photon
            number whose peak lies within the swept frequency range; an
            error is raised if that gives more peaks than frequency points.
            Peaks without any swept frequency in their window are left
            out of the fit. Default: None.
        peak_window_width:
            Width of the frequency window (in Hz) around each expected
            peak position used for extracting peak heights. If None,
            it is set to 0.4 * |chi|.
    """

    max_photon_number: int | None = workflow.option_field(
        None,
        description="Maximum photon number to extract and fit. If None, it is "
        "the highest photon number whose peak lies within the swept frequencies.",
    )
    peak_window_width: float | None = workflow.option_field(
        None,
        description="Width of the frequency window (Hz) around each expected "
        "peak. If None, it is set to 0.4 * |chi|.",
    )


@workflow.task_options(base_class=BasePlottingOptions)
class PlotDisplacementMapOptions:
    """Options for the `plot_displacement_map` task.

    Attributes:
        plot_raw_data:
            If True, plot the raw complex data (real and imaginary).
            If False, plot the magnitude. Default: False.
        save_figures:
            Whether to save the figures.
            Default: `True`.
        close_figures:
            Whether to close the figures.
            Default: `True`.
    """

    plot_raw_data: bool = workflow.option_field(
        False, description="Whether to plot the real and imaginary parts."
    )


@workflow.task_options(base_class=BasePlottingOptions)
class PlotPhotonDistributionsOptions:
    """Options for the `plot_photon_distributions` task.

    Attributes:
        num_distributions_to_plot:
            Number of evenly-spaced amplitude slices to show as
            individual photon number distributions. Default: 5.
        save_figures:
            Whether to save the figures.
            Default: `True`.
        close_figures:
            Whether to close the figures.
            Default: `True`.
    """

    num_distributions_to_plot: int = workflow.option_field(
        5, description="Number of amplitude slices to plot."
    )


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    amplitudes: QubitSweepPoints,
    frequencies: QubitSweepPoints,
    options: DisplacementCalibrationAnalysisWorkflowOptions | None = None,
) -> None:
    """The displacement calibration analysis workflow.

    The workflow consists of the following steps:

    - [fit_data][]
    - [extract_qubit_parameters][]
    - [plot_displacement_map][]
    - [plot_photon_distributions][]
    - [plot_mean_photon_number][]

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        amplitudes:
            The displacement amplitudes that were swept for each qubit.
        frequencies:
            The transmon drive frequencies that were swept for each qubit.
        options:
            The options for building the workflow, passed as an instance of
            [DisplacementCalibrationAnalysisWorkflowOptions][].

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        analysis_result = analysis_workflow(
            result=result,
            qubits=q0,
            amplitudes=np.linspace(0.0, 1.0, 51),
            frequencies=np.linspace(4.9e9, 5.1e9, 101),
        ).run()
        ```
    """
    processed_data_dict = fit_data(qubits, result, amplitudes, frequencies)
    qubit_parameters = extract_qubit_parameters(qubits, processed_data_dict)
    with workflow.if_(options.do_plotting):
        plot_displacement_map(qubits, processed_data_dict)
        plot_photon_distributions(qubits, processed_data_dict)
        plot_mean_photon_number(qubits, processed_data_dict)
    workflow.return_(qubit_parameters)


@workflow.task
def fit_data(
    qubits: QuantumElements,
    result: RunExperimentResults,
    amplitudes: QubitSweepPoints,
    frequencies: QubitSweepPoints,
    options: DisplacementCalibrationFitDataOptions | None = None,
) -> dict[str, dict]:
    """Extract and fit the photon number distributions.

    Arguments:
        qubits:
            The qubits on which to run the analysis.
        result:
            The experiment results returned by the run_experiment task.
        amplitudes:
            The displacement amplitudes that were swept for each qubit.
        frequencies:
            The transmon drive frequencies that were swept for each qubit.
        options:
            The options for this task as an instance of
            [DisplacementCalibrationFitDataOptions][].

    Returns:
        dict with qubit UIDs as keys and the per-qubit analysis results
            (raw data, photon number distributions, Poisson fits, mean photon
            numbers and `displacement_amp_per_unit_beta`) as values.
    """
    opts = DisplacementCalibrationFitDataOptions() if options is None else options
    qubits = as_quantum_element_list(qubits)

    processed_data_dict = {}
    for q, q_amps_raw, q_freqs_raw in sweeps_per_quantum_element(
        qubits, amplitudes, frequencies
    ):
        q_amps = np.asarray(q_amps_raw)
        q_freqs = np.asarray(q_freqs_raw)

        handle = dsl.handles.result_handle(q.uid)
        data_raw = result.get_data(handle)
        data_2d = np.array(data_raw).reshape(len(q_amps), len(q_freqs))

        processed_data_dict[q.uid] = _analyze_single_qubit(
            q, data_2d, q_amps, q_freqs, opts
        )

    return processed_data_dict


@workflow.task
def extract_qubit_parameters(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict],
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Extract `displacement_amp_per_unit_beta` from the fit results.

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
                    q.uid: {"displacement_amp_per_unit_beta": value},
                },
                "old_parameter_values": {
                    q.uid: {"displacement_amp_per_unit_beta": value},
                },
            }
            ```

            If the fit did not give a finite value for a qubit, its
            new_parameter_values entry is left empty.
    """
    qubits = as_quantum_element_list(qubits)
    qubit_parameters = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
    }
    for q in qubits:
        qubit_parameters["old_parameter_values"][q.uid] = {
            "displacement_amp_per_unit_beta": (
                q.parameters.displacement_amp_per_unit_beta
            ),
        }
        value = processed_data_dict[q.uid]["displacement_amp_per_unit_beta"]
        if np.isfinite(value):
            qubit_parameters["new_parameter_values"][q.uid] = {
                "displacement_amp_per_unit_beta": value,
            }
    return qubit_parameters


@workflow.task
def plot_displacement_map(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict],
    options: PlotDisplacementMapOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot the number-split transmon spectrum vs. displacement amplitude.

    Arguments:
        qubits:
            The qubits on which to run the analysis.
        processed_data_dict:
            The processed data dictionary returned by fit_data.
        options:
            The options for this task as an instance of [PlotDisplacementMapOptions][].

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = PlotDisplacementMapOptions() if options is None else options
    qubits = as_quantum_element_list(qubits)
    figures = {}
    for q in qubits:
        qubit_result = processed_data_dict[q.uid]
        fig = _plot_2d_map(
            q.uid,
            qubit_result["raw_data"],
            qubit_result["amplitudes"],
            qubit_result["frequencies"],
            f_transmon_0=qubit_result["f_transmon_0"],
            chi=qubit_result["chi"],
            max_photon_number=int(qubit_result["photon_numbers"][-1]),
            plot_raw_data=opts.plot_raw_data,
        )
        _save_and_close(fig, f"displacement_cal_{q.uid}_2d_map", opts)
        figures[q.uid] = fig
    return figures


@workflow.task
def plot_photon_distributions(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict],
    options: PlotPhotonDistributionsOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot the extracted photon number distributions with their Poisson fits.

    Arguments:
        qubits:
            The qubits on which to run the analysis.
        processed_data_dict:
            The processed data dictionary returned by fit_data.
        options:
            The options for this task as an instance of
            [PlotPhotonDistributionsOptions][].

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = PlotPhotonDistributionsOptions() if options is None else options
    qubits = as_quantum_element_list(qubits)
    figures = {}
    for q in qubits:
        qubit_result = processed_data_dict[q.uid]
        fig = _plot_photon_distributions(
            q.uid,
            qubit_result["photon_numbers"],
            qubit_result["photon_distributions"],
            qubit_result["poisson_fits"],
            qubit_result["n_bar_fit"],
            qubit_result["amplitudes"],
            num_to_plot=opts.num_distributions_to_plot,
        )
        _save_and_close(fig, f"displacement_cal_{q.uid}_photon_distributions", opts)
        figures[q.uid] = fig
    return figures


@workflow.task
def plot_mean_photon_number(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict],
    options: BasePlottingOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot the mean photon number vs. displacement amplitude.

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
        fig = _plot_nbar_vs_amplitude(
            q.uid,
            qubit_result["amplitudes"],
            qubit_result["n_bar_fit"],
            qubit_result["n_bar_direct"],
        )
        _save_and_close(fig, f"displacement_cal_{q.uid}_nbar_vs_amplitude", opts)
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


def _analyze_single_qubit(
    q: QuantumElements,
    data_2d: np.ndarray,
    q_amplitudes: np.ndarray,
    q_frequencies: np.ndarray,
    opts: DisplacementCalibrationFitDataOptions,
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

    peak_window_width = (
        0.4 * abs(chi) if opts.peak_window_width is None else opts.peak_window_width
    )
    if opts.max_photon_number is None:
        max_n = _max_photon_number_in_sweep(
            q_frequencies, f_transmon_0, chi, peak_window_width
        )
        if max_n + 1 > len(q_frequencies):
            raise ValueError(
                f"Qubit {q.uid}: the sweep covers {max_n + 1} photon-number "
                f"peaks but has only {len(q_frequencies)} frequency points, so "
                f"the peaks cannot be resolved. Check that chi ({chi} Hz) is "
                f"given in Hz, or set max_photon_number explicitly."
            )
    else:
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
            peak_window_width=peak_window_width,
        )
        photon_distributions[i] = weights
        n_bar_i, fit_dist_i = _fit_coherent_state(photon_numbers, weights)
        n_bar_fit[i] = n_bar_i
        poisson_fits[i] = fit_dist_i
        total = np.nansum(weights)
        if total > 0:
            n_bar_direct[i] = np.nansum(photon_numbers * weights) / total

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
    }


# =====================================================================
# Fitting helpers
# =====================================================================


def _poisson_pmf(x: np.ndarray, n_bar: float) -> np.ndarray:
    """Poisson probability mass function.

    P(n) = n_bar^n * exp(-n_bar) / n!

    Arguments:
        x: photon number array.
        n_bar: mean photon number.

    Returns:
        Poisson probabilities.
    """
    return poisson.pmf(x, n_bar)


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
            Extracted peak heights for each photon number. NaN for
            photon numbers with no swept frequency in their window.
    """
    signal = np.abs(spectrum)

    # Baseline subtraction
    baseline = np.median(signal)
    signal_corrected = np.maximum(signal - baseline, 0.0)

    if peak_window_width is None:
        peak_window_width = 0.4 * abs(chi)

    photon_numbers = np.arange(max_photon_number + 1)
    peak_positions = f_transmon_0 + photon_numbers * chi

    weights = np.full(max_photon_number + 1, np.nan)
    for i, f_peak in enumerate(peak_positions):
        mask = np.abs(frequencies - f_peak) <= peak_window_width / 2
        if np.any(mask):
            weights[i] = np.max(signal_corrected[mask])

    return photon_numbers, weights


def _max_photon_number_in_sweep(
    frequencies: np.ndarray,
    f_transmon_0: float,
    chi: float,
    peak_window_width: float,
) -> int:
    """Return the highest photon number whose peak window overlaps the sweep.

    The peaks sit at f_transmon_0 + n * chi, so they move towards the
    upper end of the sweep for chi > 0 and towards the lower end for chi < 0.
    """
    if chi > 0:
        edge = np.max(frequencies) + peak_window_width / 2
    else:
        edge = np.min(frequencies) - peak_window_width / 2
    return max(int(np.floor((edge - f_transmon_0) / chi)), 0)


def _fit_coherent_state(
    photon_numbers: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Fit extracted photon number weights to a Poisson distribution.

    Arguments:
        photon_numbers:
            Array [0, 1, ..., N].
        weights:
            Extracted peak heights for each photon number. NaN entries
            (peaks outside the sweep) are left out of the fit.

    Returns:
        n_bar_fit:
            Fitted mean photon number.
        fit_distribution:
            Normalized Poisson distribution evaluated at the
            fitted n_bar.
    """
    measured = np.isfinite(weights)
    total = np.sum(weights[measured])
    if total <= 0:
        return 0.0, np.zeros_like(weights, dtype=float)

    probabilities = weights[measured] / total

    # Initial guess from direct mean
    n_bar_guess = float(np.sum(photon_numbers[measured] * probabilities))
    n_bar_guess = max(n_bar_guess, 0.01)

    try:
        fit_res = fit_data_lmfit(
            _poisson_pmf,
            photon_numbers[measured],
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
        total = np.nansum(weights)
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
