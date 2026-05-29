# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Analysis for a parametric CZ chevron (frequency vs amplitude).

In this analysis, we first interpret the raw data into qubit population using
principle-component analysis or rotation and projection on the measured calibration
states. We then fit each frequency row with a cosine in amplitude to extract the
oscillation rate as a function of flux pulse frequency, fit those data with a parabola,
and extract the optimal flux pulse frequency at the maximum oscillation rate. The
optimal amplitude is taken as one full oscillation period (full 2*pi Rabi rotation)
at the optimal frequency.
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
    amplitudes: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The parametric CZ flux frequency vs amplitude analysis workflow.

    The workflow consists of the following steps:

    - [validate_and_extract_edges_from_qubit_pairs]()
    - [extract_nodes_from_edges]()
    - [calculate_qubit_population_2d]()
    - [fit_chevron_rows]()
    - [fit_parabola]()
    - [extract_edge_parameters]()
    - [plot_population]()
    - [plot_fitted_chevron_rate]()

    Arguments:
        result:
            The experiment results returned by the `run_experiment` task.
        qpu:
            The quantum processing unit.
        qubit_pairs:
            The qubit pairs on which to run the analysis, passed as a list of UID lists.
            The UIDs of these qubits must exist in the result.
        frequencies:
            Frequencies of the flux pulse on the coupler.
        amplitudes:
            Amplitudes swept for the flux pulse on the coupler.
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
        sweep_points_1d=amplitudes,
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
            plot_fitted_chevron_rate(
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
        paths ``"coupler_pulse.frequency"`` and ``"coupler_pulse.amplitude"``
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
            "coupler_pulse.amplitude": e.parameters.coupler_pulse["amplitude"],
        }

        par = parabola_results.get(e.target_node.uid)
        if par is not None:
            edge_parameters["new_parameter_values"][
                ("cz", e.source_node.uid, e.target_node.uid)
            ] = {
                "coupler_pulse.frequency": par["optimal_flux_frequency"],
                "coupler_pulse.amplitude": 1.0 / par["max_osc_rate"],
            }

    return edge_parameters


@workflow.task
def fit_chevron_rows(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
) -> dict[str, dict[str, ArrayLike]]:
    """Fit each flux-frequency row with a cosine to extract oscillation rates.

    For each flux pulse frequency, the qubit population vs amplitude trace is fitted
    with the model:
        P(A) = A_p * cos(2π * k * A + φ) + C

    The oscillation rate k (cycles per unit amplitude) is extracted for each flux
    frequency, forming the chevron rate curve k(f). Near the |11>-|20> resonance the
    Rabi-like oscillation in amplitude is fastest (max k), so k(f) is approximately
    concave-down with a maximum at the resonance frequency.

    Arguments:
        qubits:
            The qubits on which to run the task.
        processed_data_dict:
            The processed data dictionary returned by [calculate_qubit_population_2d]().
            Expected shape of population array: [N_frequencies, N_amplitudes].

    Returns:
        Dictionary with qubit UIDs as keys. Each value contains:
            - ``flux_frequencies``: the swept flux pulse frequencies.
            - ``osc_rates``: fitted oscillation rate for each flux frequency.
            - ``osc_rates_err``: fit uncertainty (stderr) for each oscillation
              rate, NaN where the fit failed.
    """
    qubits = validate_and_convert_qubits_sweeps(qubits)
    fit_results = {}

    def _cosine(
        a: float,
        A: float,  # noqa: N803
        osc_rate: float,
        phase: float,
        offset: float,
    ) -> float:
        return A * np.cos(2 * np.pi * osc_rate * a + phase) + offset

    model = lmfit.Model(_cosine)

    for q in qubits:
        amplitudes = np.asarray(processed_data_dict[q.uid]["sweep_points_1d"])
        flux_frequencies = np.asarray(processed_data_dict[q.uid]["sweep_points_2d"])
        data = np.asarray(processed_data_dict[q.uid]["population"])

        n_freq = len(flux_frequencies)
        osc_rates = np.full(n_freq, np.nan)
        osc_rates_err = np.full(n_freq, np.nan)

        da = amplitudes[1] - amplitudes[0] if len(amplitudes) > 1 else 1.0

        for i_f in range(n_freq):
            row = data[i_f]

            fft_vals = np.abs(np.fft.rfft(row - np.mean(row)))
            fft_rates = np.fft.rfftfreq(len(row), d=da)
            peak_idx = int(np.argmax(fft_vals[1:])) + 1
            rate_guess = float(fft_rates[peak_idx])

            amp_guess = float((np.max(row) - np.min(row)) / 2)
            offset_guess = float(np.mean(row))

            params = model.make_params(
                A={"value": amp_guess, "min": 0.0, "max": 1.0},
                osc_rate={"value": rate_guess, "min": 0.0},
                phase={"value": 0.0, "min": -np.pi, "max": np.pi},
                offset={"value": offset_guess, "min": 0.0, "max": 1.0},
            )

            try:
                fit = model.fit(row, params, a=amplitudes)
                osc_rates[i_f] = fit.params["osc_rate"].value
                stderr = fit.params["osc_rate"].stderr
                osc_rates_err[i_f] = stderr if stderr is not None else np.nan
            except (ValueError, RuntimeError):
                pass

        fit_results[q.uid] = {
            "flux_frequencies": flux_frequencies,
            "osc_rates": osc_rates,
            "osc_rates_err": osc_rates_err,
        }

    return fit_results


_MIN_FIT_POINTS = 3


@workflow.task
def fit_parabola(
    chevron_fit_results: dict[str, dict[str, ArrayLike]],
) -> dict[str, dict | None]:
    """Fit oscillation rate vs flux pulse frequency with a (concave-down) parabola.

    Fits the model:
        k(f) = -a * (f - f0)² + k_max,   a > 0

    and extracts the optimal flux pulse frequency ``f0`` at which the oscillation
    rate is maximum, and the maximum oscillation rate ``k_max``.

    Arguments:
        chevron_fit_results:
            The dictionary returned by [fit_chevron_rows]().

    Returns:
        Dictionary with qubit UIDs as keys. Each value contains:
            - ``optimal_flux_frequency``: the flux frequency at the parabola maximum.
            - ``optimal_flux_frequency_err``: fit uncertainty for the above.
            - ``max_osc_rate``: the maximum oscillation rate (k_max).
            - ``max_osc_rate_err``: fit uncertainty for the above.
            - ``fit_result``: the raw lmfit ModelResult.
            - ``flux_frequencies``: the valid (non-NaN) flux frequencies used for fit.
            - ``osc_rates``: the valid oscillation rates used for fit.
            Value is ``None`` if the fit could not be performed.
    """

    def _parabola(f: float, a: float, f0: float, k_max: float) -> float:
        return -a * (f - f0) ** 2 + k_max

    model = lmfit.Model(_parabola)
    parabola_results = {}

    for q_uid, data in chevron_fit_results.items():
        flux_freqs = np.asarray(data["flux_frequencies"])
        osc_rates = np.asarray(data["osc_rates"])

        valid = ~np.isnan(osc_rates)
        flux_freqs_v = flux_freqs[valid]
        osc_rates_v = osc_rates[valid]

        if len(osc_rates_v) < _MIN_FIT_POINTS:
            parabola_results[q_uid] = None
            continue

        max_idx = int(np.argmax(osc_rates_v))
        f0_guess = float(flux_freqs_v[max_idx])
        k_max_guess = float(osc_rates_v[max_idx])

        f_half_range = (float(flux_freqs_v[-1]) - float(flux_freqs_v[0])) / 2
        k_range = float(np.max(osc_rates_v) - np.min(osc_rates_v))
        a_guess = k_range / f_half_range**2 if f_half_range > 0 else 1.0

        params = model.make_params(
            a={"value": a_guess, "min": 0.0},
            f0={"value": f0_guess},
            k_max={"value": k_max_guess, "min": 0.0},
        )

        try:
            fit = model.fit(osc_rates_v, params, f=flux_freqs_v)
            f0_err = fit.params["f0"].stderr
            k_max_err = fit.params["k_max"].stderr
            parabola_results[q_uid] = {
                "fit_result": fit,
                "optimal_flux_frequency": fit.params["f0"].value,
                "optimal_flux_frequency_err": f0_err if f0_err is not None else np.nan,
                "max_osc_rate": fit.params["k_max"].value,
                "max_osc_rate_err": k_max_err if k_max_err is not None else np.nan,
                "flux_frequencies": flux_freqs_v,
                "osc_rates": osc_rates_v,
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
    """Create the 2D chevron (frequency vs amplitude) population plots.

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
            label_x_values="Flux pulse amplitude (a.u.)",
            label_y_values="Frequency of gate, $f$ (MHz)",
            label_z_values="Principal Component (a.u)"
            if (num_cal_traces == 0 or opts.do_pca)
            else f"$|{opts.cal_states[-1]}\\rangle$-State Population",
            scaling_x_values=1,
            scaling_y_values=1e-6,
            figure=fig,
            plot_title=f" chevron {q.uid}",
            axis=axs,
            close_figures=opts.close_figures,
        )

        if opts.save_figures:
            workflow.save_artifact(f"parametric_cz_chevron_amplitude_{q.uid}", fig)

        figures[q.uid] = fig

    return figures


@workflow.task
def plot_fitted_chevron_rate(
    chevron_fit_results: dict[str, dict[str, ArrayLike]],
    parabola_results: dict[str, dict | None],
    options: PlotPopulationOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot the fitted oscillation rate vs flux pulse frequency.

    For each qubit, plots the oscillation rate extracted from cosine fits to each
    amplitude trace (data points), overlaid with the parabola fit. A vertical line
    marks the optimal flux frequency and a horizontal line marks the maximum
    oscillation rate.

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
        osc_rates = np.asarray(chev_data["osc_rates"])
        osc_rates_err = np.asarray(chev_data["osc_rates_err"])

        fig, ax = plt.subplots()

        valid = ~np.isnan(osc_rates)
        ax.errorbar(
            flux_freqs[valid] * 1e-6,
            osc_rates[valid],
            yerr=osc_rates_err[valid],
            fmt="o",
            label="Fitted oscillation rate",
            capsize=3,
        )

        par = parabola_results.get(q_uid)
        if par is not None:
            f_dense = np.linspace(
                float(flux_freqs[valid].min()), float(flux_freqs[valid].max()), 300
            )
            k_fit = par["fit_result"].eval(f=f_dense)
            ax.plot(f_dense * 1e-6, k_fit, "-", label="Parabola fit")

            f_opt = par["optimal_flux_frequency"]
            f_opt_err = par["optimal_flux_frequency_err"]
            k_max = par["max_osc_rate"]
            k_max_err = par["max_osc_rate_err"]

            f_opt_label = f"$f_{{opt}}$ = {f_opt * 1e-6:.3f} MHz"
            if not np.isnan(f_opt_err):
                f_opt_label += f" ± {f_opt_err * 1e-6:.3f} MHz"

            k_max_label = f"$k_{{max}}$ = {k_max:.4g} 1/a.u."
            if not np.isnan(k_max_err):
                k_max_label += f" ± {k_max_err:.2g} 1/a.u."

            ax.axvline(
                f_opt * 1e-6,
                color="red",
                linestyle="--",
                label=f_opt_label,
            )
            ax.axhline(
                k_max,
                color="green",
                linestyle="--",
                label=k_max_label,
            )

        ax.set_xlabel("Flux pulse frequency, $f$ (MHz)")
        ax.set_ylabel("Oscillation rate, $k$ (1/a.u.)")
        ax.set_title(timestamped_title(f"Chevron rate vs flux frequency - {q_uid}"))
        ax.legend(fontsize=8)

        if opts.save_figures:
            workflow.save_artifact(f"chevron_rate_parabola_{q_uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q_uid] = fig

    return figures
