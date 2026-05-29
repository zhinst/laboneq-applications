# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""This module defines the analysis for a parametric CZ chevron (frequency vs duration).

In this analysis, we first interpret the raw data into qubit population using
principle-component analysis or rotation and projection on the measured calibration
states. We then fit each frequency row with a cosine to extract the oscillation
frequency as a function of flux pulse frequency, fit those data with a parabola,
and extract the optimal flux pulse frequency at the minimum oscillation frequency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import lmfit
import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.dsl.quantum import QPU, QuantumElement
from laboneq.dsl.quantum.qpu_topology import TopologyEdge

from laboneq_applications.analysis.calibration_traces_rotation import (
    calculate_qubit_population_2d,
)
from laboneq_applications.analysis.options import (
    PlotPopulationOptions,
    TuneUpAnalysisWorkflowOptions,
)
from laboneq_applications.analysis.plotting_helpers import (
    plot_data_2d,
    timestamped_title,
)
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_and_extract_edges_from_qubit_pairs,
)
from laboneq_applications.qpu_types.tunable_coupler import TunableCoupler
from laboneq_applications.tasks import extract_nodes_from_edges

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import (
        QuantumElements,
        QubitSweepPoints,
    )


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qpu: QPU,
    qubit_pairs: list[list[QuantumElement]],
    frequencies: QubitSweepPoints,
    durations: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The parametric CZ flux frequency vs duration analysis workflow.

    The workflow consists of the following steps:

    - [validate_and_extract_edges_from_qubit_pairs]()
    - [extract_nodes_from_edges]()
    - [calculate_qubit_population_2d]()
    - [fit_chevron_rows]()
    - [fit_parabola]()
    - [plot_population]()
    - [plot_fitted_chevron_frequency]()

    Arguments:
        result:
            The experiment results returned by the `run_experiment` task.
        qpu:
            The quantum processing unit.
        qubit_pairs:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        frequencies:
            Frequencies of the flux pulse on the coupler.
        durations:
            Duration (length) values swept for the flux pulse on the coupler.
        options:
            The options for building the workflow, passed as an instance of
                [TuneUpAnalysisWorkflowOptions].

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.
    """
    edges = validate_and_extract_edges_from_qubit_pairs(
        qpu, "cz", qubit_pairs, element_class=TunableCoupler
    )

    qubits_target = extract_nodes_from_edges(edges, "target")

    processed_data_dict_target = calculate_qubit_population_2d(
        qubits=qubits_target,
        result=result,
        sweep_points_1d=durations,
        sweep_points_2d=frequencies,
    )

    chevron_fit_results_target = fit_chevron_rows(
        qubits_target, processed_data_dict_target
    )

    parabola_results_target = fit_parabola(chevron_fit_results_target)

    qubit_parameters = extract_edge_parameters(edges, parabola_results_target)

    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_qubit_population_plotting):
            plot_population(qubits_target, processed_data_dict_target)
            plot_fitted_chevron_frequency(
                chevron_fit_results_target, parabola_results_target
            )

    workflow.return_(qubit_parameters)


@workflow.task
def extract_edge_parameters(
    edges: list[TopologyEdge],
    parabola_results: dict[str, dict | None],
) -> dict:
    """Extract the optimal coupler pulse parameters from parabola fit results.

    Arguments:
        edges:
            The topology edges of the calibrated qubit pairs.
        parabola_results:
            The dictionary returned by [fit_parabola](), keyed by target qubit UID.

    Returns:
        Dictionary with ``new_parameter_values`` and ``old_parameter_values``, each
        keyed by ``("cz", source_uid, target_uid)`` and containing dotted parameter
        paths ``"coupler_pulse.frequency"`` and ``"coupler_pulse.length"``
        compatible with ``qpu.update()``.
    """
    edge_parameters = {
        "old_parameter_values": {
            ("cz", e.source_node.uid, e.target_node.uid): {} for e in edges
        },
        "new_parameter_values": {
            ("cz", e.source_node.uid, e.target_node.uid): {} for e in edges
        },
    }

    for e in edges:
        edge_parameters["old_parameter_values"][
            ("cz", e.source_node.uid, e.target_node.uid)
        ] = {
            "coupler_pulse.frequency": e.parameters.coupler_pulse["frequency"],
            "coupler_pulse.length": e.parameters.coupler_pulse["length"],
        }

        par = parabola_results.get(e.target_node.uid)
        if par is not None:
            edge_parameters["new_parameter_values"][
                ("cz", e.source_node.uid, e.target_node.uid)
            ] = {
                "coupler_pulse.frequency": par["optimal_flux_frequency"],
                "coupler_pulse.length": 1.0 / par["min_osc_frequency"],
            }

    return edge_parameters


@workflow.task
def fit_chevron_rows(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
) -> dict[str, dict[str, ArrayLike]]:
    """Fit each flux-frequency row with a cosine to extract oscillation frequencies.

    For each flux pulse frequency, the qubit population vs duration trace is fitted
    with the model:
        P(t) = A * cos(2pi * nu * t + phi) + C

    The oscillation frequency nu is extracted for each flux frequency, forming the
    chevron frequency curve nu(f).

    Arguments:
        qubits:
            The qubits on which to run the task.
        processed_data_dict:
            The processed data dictionary returned by [calculate_qubit_population_2d]().
            Expected shape of population array: [N_frequencies, N_durations].

    Returns:
        Dictionary with qubit UIDs as keys. Each value contains:
            - ``flux_frequencies``: the swept flux pulse frequencies.
            - ``osc_frequencies``: fitted oscillation frequency for each flux frequency.
            - ``osc_frequencies_err``: fit uncertainty (stderr) for each oscillation
              frequency, NaN where the fit failed.
    """
    qubits = validate_and_convert_qubits_sweeps(qubits)
    fit_results = {}

    def _cosine(
        t: float,
        A: float,  # noqa: N803
        osc_freq: float,
        phase: float,
        offset: float,
    ) -> float:
        return A * np.cos(2 * np.pi * osc_freq * t + phase) + offset

    model = lmfit.Model(_cosine)

    for q in qubits:
        durations = np.asarray(processed_data_dict[q.uid]["sweep_points_1d"])
        flux_frequencies = np.asarray(processed_data_dict[q.uid]["sweep_points_2d"])
        data = np.asarray(processed_data_dict[q.uid]["population"])

        n_freq = len(flux_frequencies)
        osc_freqs = np.full(n_freq, np.nan)
        osc_freqs_err = np.full(n_freq, np.nan)

        dt = durations[1] - durations[0] if len(durations) > 1 else 1.0

        for i_f in range(n_freq):
            row = data[i_f]

            # FFT-based initial guess for oscillation frequency
            fft_vals = np.abs(np.fft.rfft(row - np.mean(row)))
            fft_freqs = np.fft.rfftfreq(len(row), d=dt)
            # Skip DC bin (index 0)
            peak_idx = int(np.argmax(fft_vals[1:])) + 1
            freq_guess = float(fft_freqs[peak_idx])

            amp_guess = float((np.max(row) - np.min(row)) / 2)
            offset_guess = float(np.mean(row))

            params = model.make_params(
                A={"value": amp_guess, "min": 0.0, "max": 1.0},
                osc_freq={"value": freq_guess, "min": 0.0},
                phase={"value": 0.0, "min": -np.pi, "max": np.pi},
                offset={"value": offset_guess, "min": 0.0, "max": 1.0},
            )

            try:
                fit = model.fit(row, params, t=durations)
                osc_freqs[i_f] = fit.params["osc_freq"].value
                stderr = fit.params["osc_freq"].stderr
                osc_freqs_err[i_f] = stderr if stderr is not None else np.nan
            except (ValueError, RuntimeError):
                pass

        fit_results[q.uid] = {
            "flux_frequencies": flux_frequencies,
            "osc_frequencies": osc_freqs,
            "osc_frequencies_err": osc_freqs_err,
        }

    return fit_results


_MIN_FIT_POINTS = 3


@workflow.task
def fit_parabola(
    chevron_fit_results: dict[str, dict[str, ArrayLike]],
) -> dict[str, dict | None]:
    """Fit oscillation frequency vs flux pulse frequency with a parabola.

    Fits the model:
        nu(f) = a * (f - f0)^2 + nu_min

    and extracts the optimal flux pulse frequency ``f0`` at which the oscillation
    frequency is minimum, and the minimum oscillation frequency ``nu_min``.

    Arguments:
        chevron_fit_results:
            The dictionary returned by [fit_chevron_rows]().

    Returns:
        Dictionary with qubit UIDs as keys. Each value contains:
            - ``optimal_flux_frequency``: the flux frequency at the parabola minimum.
            - ``optimal_flux_frequency_err``: fit uncertainty for the above.
            - ``min_osc_frequency``: the minimum oscillation frequency (nu_min).
            - ``min_osc_frequency_err``: fit uncertainty for the above.
            - ``fit_result``: the raw lmfit ModelResult.
            - ``flux_frequencies``: the valid (non-NaN) flux frequencies used for fit.
            - ``osc_frequencies``: the valid oscillation frequencies used for fit.
            Value is ``None`` if the fit could not be performed.
    """

    def _parabola(f: float, a: float, f0: float, nu_min: float) -> float:
        return a * (f - f0) ** 2 + nu_min

    model = lmfit.Model(_parabola)
    parabola_results = {}

    for q_uid, data in chevron_fit_results.items():
        flux_freqs = np.asarray(data["flux_frequencies"])
        osc_freqs = np.asarray(data["osc_frequencies"])

        # Keep only well-fitted rows
        valid = ~np.isnan(osc_freqs)
        flux_freqs_v = flux_freqs[valid]
        osc_freqs_v = osc_freqs[valid]

        if len(osc_freqs_v) < _MIN_FIT_POINTS:
            parabola_results[q_uid] = None
            continue

        # Initial guesses: minimum of the fitted oscillation frequencies
        min_idx = int(np.argmin(osc_freqs_v))
        f0_guess = float(flux_freqs_v[min_idx])
        nu_min_guess = float(osc_freqs_v[min_idx])

        # Curvature estimate: (max - min) / (half-range)^2
        f_half_range = (float(flux_freqs_v[-1]) - float(flux_freqs_v[0])) / 2
        nu_range = float(np.max(osc_freqs_v) - np.min(osc_freqs_v))
        a_guess = nu_range / f_half_range**2 if f_half_range > 0 else 1.0

        params = model.make_params(
            a={"value": a_guess, "min": 0.0},
            f0={"value": f0_guess},
            nu_min={"value": nu_min_guess, "min": 0.0},
        )

        try:
            fit = model.fit(osc_freqs_v, params, f=flux_freqs_v)
            f0_err = fit.params["f0"].stderr
            nu_min_err = fit.params["nu_min"].stderr
            parabola_results[q_uid] = {
                "fit_result": fit,
                "optimal_flux_frequency": fit.params["f0"].value,
                "optimal_flux_frequency_err": f0_err if f0_err is not None else np.nan,
                "min_osc_frequency": fit.params["nu_min"].value,
                "min_osc_frequency_err": nu_min_err
                if nu_min_err is not None
                else np.nan,
                "flux_frequencies": flux_freqs_v,
                "osc_frequencies": osc_freqs_v,
            }
        except (ValueError, RuntimeError):
            parabola_results[q_uid] = None

    return parabola_results


@workflow.task
def plot_population(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: PlotPopulationOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the 2D chevron (frequency vs duration) population plots.

    Arguments:
        qubits:
            The qubits on which to run the task. The UIDs of these qubits must exist in
            `processed_data_dict`.
        processed_data_dict:
            The processed data dictionary returned by [calculate_qubit_population_2d]().
        options:
            The options for processing the raw data.

    Returns:
        Dictionary with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = PlotPopulationOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)

    figures = {}
    for q in qubits:
        num_cal_traces = processed_data_dict[q.uid]["num_cal_traces"]

        sweep_points_1d = processed_data_dict[q.uid]["sweep_points_1d"]
        sweep_points_2d = processed_data_dict[q.uid]["sweep_points_2d"]
        data = processed_data_dict[q.uid][
            "population" if opts.do_rotation else "data_raw"
        ]

        fig, axs = plt.subplots()
        fig, axs = plot_data_2d(
            x_values=sweep_points_1d,
            y_values=sweep_points_2d,
            z_values=data,
            label_x_values="Flux pulse duration, $t$ (ns)",
            label_y_values="Frequency of gate, $f$ (MHz)",
            label_z_values="Principal Component (a.u)"
            if (num_cal_traces == 0 or opts.do_pca)
            else f"$|{opts.cal_states[-1]}\\rangle$-State Population",
            scaling_x_values=1e9,
            scaling_y_values=1e-6,
            figure=fig,
            plot_title=f" chevron {q.uid}",
            axis=axs,
            close_figures=opts.close_figures,
        )

        if opts.save_figures:
            workflow.save_artifact(f"parametric_cz_chevron_duration_{q.uid}", fig)

        figures[q.uid] = fig

    return figures


@workflow.task
def plot_fitted_chevron_frequency(
    chevron_fit_results: dict[str, dict[str, ArrayLike]],
    parabola_results: dict[str, dict | None],
    options: PlotPopulationOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot the fitted chevron oscillation frequency vs flux pulse frequency.

    For each qubit, plots the oscillation frequency extracted from cosine fits to each
    duration trace (data points), overlaid with the parabola fit. A vertical line marks
    the optimal flux frequency and a horizontal line marks the minimum oscillation
    frequency.

    Arguments:
        chevron_fit_results:
            The dictionary returned by [fit_chevron_rows]().
        parabola_results:
            The dictionary returned by [fit_parabola]().
        options:
            Plot options (save, close, etc.).

    Returns:
        Dictionary with qubit UIDs as keys and figures as values.
    """
    opts = PlotPopulationOptions() if options is None else options
    figures = {}

    for q_uid, chev_data in chevron_fit_results.items():
        flux_freqs = np.asarray(chev_data["flux_frequencies"])
        osc_freqs = np.asarray(chev_data["osc_frequencies"])
        osc_freqs_err = np.asarray(chev_data["osc_frequencies_err"])

        fig, ax = plt.subplots()

        # Plot measured oscillation frequencies with error bars
        valid = ~np.isnan(osc_freqs)
        ax.errorbar(
            flux_freqs[valid] * 1e-6,
            osc_freqs[valid] * 1e-6,
            yerr=osc_freqs_err[valid] * 1e-6,
            fmt="o",
            label="Fitted oscillation frequency",
            capsize=3,
        )

        par = parabola_results.get(q_uid)
        if par is not None:
            # Dense frequency axis for smooth parabola curve
            f_dense = np.linspace(
                float(flux_freqs[valid].min()), float(flux_freqs[valid].max()), 300
            )
            nu_fit = par["fit_result"].eval(f=f_dense)
            ax.plot(f_dense * 1e-6, nu_fit * 1e-6, "-", label="Parabola fit")

            f_opt = par["optimal_flux_frequency"]
            f_opt_err = par["optimal_flux_frequency_err"]
            nu_min = par["min_osc_frequency"]
            nu_min_err = par["min_osc_frequency_err"]

            f_opt_label = f"$f_{{opt}}$ = {f_opt * 1e-6:.3f} MHz"
            if not np.isnan(f_opt_err):
                f_opt_label += f" ± {f_opt_err * 1e-6:.3f} MHz"

            nu_min_label = f"$\\nu_{{min}}$ = {nu_min * 1e-6:.4f} MHz"
            if not np.isnan(nu_min_err):
                nu_min_label += f" ± {nu_min_err * 1e-6:.4f} MHz"

            ax.axvline(
                f_opt * 1e-6,
                color="red",
                linestyle="--",
                label=f_opt_label,
            )
            ax.axhline(
                nu_min * 1e-6,
                color="green",
                linestyle="--",
                label=nu_min_label,
            )

        ax.set_xlabel("Flux pulse frequency, $f$ (MHz)")
        ax.set_ylabel("Oscillation frequency, $\\nu$ (MHz)")
        ax.set_title(
            timestamped_title(f"Chevron frequency vs flux frequency - {q_uid}")
        )
        ax.legend(fontsize=8)

        if opts.save_figures:
            workflow.save_artifact(f"chevron_frequency_parabola_{q_uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q_uid] = fig

    return figures
