# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Analysis for the parametric CZ frequency vs RZ phase experiment.

For each CZ gate frequency (outer sweep), the target qubit population is
measured as a function of RZ phase (inner sweep) and fitted with a cosine:

    P(φ) = A · cos(φ + φ₀) + C

The phase offset φ₀ is extracted for each frequency row. This is done
separately for the two measurement handles ``_0`` and ``_1`` (corresponding
to two different control-qubit state preparations). Finally, φ₀ vs. CZ
frequency is plotted for both cases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from laboneq_applications.analysis.plotting_helpers import timestamped_title
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_and_extract_edges_from_qubit_pairs,
)
from laboneq_applications.qpu_types.tunable_coupler import TunableCoupler
from laboneq_applications.tasks import extract_nodes_from_edges

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults

    from laboneq_applications.typing import (
        QuantumElements,
        QubitSweepPoints,
    )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qpu: QPU,
    qubit_pairs: list[list[QuantumElement]],
    frequencies: QubitSweepPoints,
    phases: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """Analysis workflow for the parametric CZ frequency vs RZ phase experiment.

    The workflow consists of the following steps:

    - [calculate_qubit_population_2d]() (for handles ``_0`` and ``_1``)
    - [fit_phase_oscillations]()
    - [fit_conditional_phase]()
    - [extract_edge_parameters]()
    - [plot_population_2d]()
    - [plot_phase_offset_vs_frequency]()

    Arguments:
        result:
            The experiment results returned by the `run_experiment` task.
        qpu:
            The quantum processing unit.
        qubit_pairs:
            The qubit pairs on which to run the analysis, passed as a list of UID
            lists. The UIDs of these qubits must exist in the result.
        frequencies:
            CZ gate frequencies swept in the outer loop (per qubit pair).
        phases:
            RZ gate phases swept in the inner loop (per qubit pair).
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

    processed_data_dict_0 = calculate_qubit_population_2d(
        qubits=qubits_target,
        result=result,
        sweep_points_1d=phases,
        sweep_points_2d=frequencies,
        result_handle="_0",
    )
    processed_data_dict_1 = calculate_qubit_population_2d(
        qubits=qubits_target,
        result=result,
        sweep_points_1d=phases,
        sweep_points_2d=frequencies,
        result_handle="_1",
    )

    fit_results = fit_phase_oscillations(
        qubits_target, processed_data_dict_0, processed_data_dict_1
    )

    poly_results = fit_conditional_phase(fit_results)

    qubit_parameters = extract_edge_parameters(edges, poly_results)

    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_qubit_population_plotting):
            plot_population_2d(
                qubits_target, processed_data_dict_0, processed_data_dict_1
            )
            plot_phase_offset_vs_frequency(poly_results)

    workflow.return_(qubit_parameters)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@workflow.task
def fit_phase_oscillations(
    qubits: QuantumElements,
    processed_data_dict_0: dict[str, dict],
    processed_data_dict_1: dict[str, dict],
    amplitude_threshold: float = 0.05,
) -> dict[str, dict]:
    """Fit qubit population vs. RZ phase with a cosine for each CZ frequency row.

    Uses a **linear parameterisation** to guarantee a unique, global solution:

        P(φ) = a·cos(φ) + b·sin(φ) + C

    The amplitude and phase offset are recovered analytically:

        A   = √(a² + b²)
        φ₀  = atan2(-b, a)

    This avoids the local-minima and initial-guess sensitivity of the
    nonlinear form ``A·cos(φ + φ₀) + C``.  Uncertainties are propagated
    from the least-squares covariance matrix via the delta method.

    The resulting phase offsets are unwrapped along the frequency axis so that
    the output is free of 2π jumps.

    Arguments:
        qubits:
            Target qubits.
        processed_data_dict_0:
            Population data for handle ``_0``, keyed by qubit UID.
            Each value has ``sweep_points_1d`` (phases), ``sweep_points_2d``
            (frequencies), and ``population`` [N_freq, N_phase].
        processed_data_dict_1:
            Same structure for handle ``_1``.
        amplitude_threshold:
            Minimum fitted amplitude A (in population units) below which the
            phase offset is considered unreliable and set to NaN.

    Returns:
        Dictionary keyed by qubit UID. Each value is a dict with keys ``"_0"``
        and ``"_1"``, each containing:

        - ``frequencies``: CZ frequency array [N_freq].
        - ``phase_offsets``: fitted φ₀ per frequency [N_freq] (rad), unwrapped.
        - ``phase_offsets_err``: uncertainty of φ₀ [N_freq] (rad), NaN on failure.
        - ``amplitudes``: fitted amplitude A [N_freq].
    """
    qubits = validate_and_convert_qubits_sweeps(qubits)

    results = {}
    for q in qubits:
        results[q.uid] = {}
        for label, data_dict in (
            ("_0", processed_data_dict_0),
            ("_1", processed_data_dict_1),
        ):
            data = data_dict[q.uid]
            phases = np.asarray(data["sweep_points_1d"])
            frequencies = np.asarray(data["sweep_points_2d"])
            population = np.asarray(data["population"])  # [N_freq, N_phase]

            # Design matrix: columns are [cos(φ), sin(φ), 1]
            X = np.column_stack(  # noqa: N806
                [np.cos(phases), np.sin(phases), np.ones(len(phases))]
            )
            n_params = X.shape[1]
            dof = len(phases) - n_params

            n_freq = len(frequencies)
            phase_offsets = np.full(n_freq, np.nan)
            phase_offsets_err = np.full(n_freq, np.nan)
            amplitudes = np.full(n_freq, np.nan)

            for i_f in range(n_freq):
                row = population[i_f]

                coeffs, _, _, _ = np.linalg.lstsq(X, row, rcond=None)
                a, b = coeffs[0], coeffs[1]
                A = float(np.sqrt(a**2 + b**2))  # noqa: N806
                amplitudes[i_f] = A

                if amplitude_threshold > A or dof < 1:
                    continue

                phi_0 = float(np.arctan2(-b, a))
                phase_offsets[i_f] = phi_0

                # Uncertainty via delta method from least-squares covariance
                rss = float(np.sum((row - X @ coeffs) ** 2))
                sigma2 = rss / dof
                try:
                    cov = sigma2 * np.linalg.inv(X.T @ X)
                    # var(phi0) delta-method: dphi0/da = b/A^2, dphi0/db = -a/A^2
                    da = b / A**2
                    db = -a / A**2
                    var_phi0 = (
                        da**2 * cov[0, 0] + db**2 * cov[1, 1] + 2 * da * db * cov[0, 1]
                    )
                    phase_offsets_err[i_f] = float(np.sqrt(max(var_phi0, 0.0)))
                except np.linalg.LinAlgError:
                    pass

            # Unwrap along the frequency axis to remove 2π jumps
            valid = ~np.isnan(phase_offsets)
            if valid.sum() > 1:
                phase_offsets[valid] = np.unwrap(phase_offsets[valid])

            results[q.uid][label] = {
                "frequencies": frequencies,
                "phase_offsets": phase_offsets,
                "phase_offsets_err": phase_offsets_err,
                "amplitudes": amplitudes,
            }

    return results


@workflow.task
def plot_population_2d(
    qubits: QuantumElements,
    processed_data_dict_0: dict[str, dict],
    processed_data_dict_1: dict[str, dict],
    options: PlotPopulationOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot target-qubit population as a 2-D colour map (phase x frequency).

    One figure per qubit per handle (two figures per qubit).

    Arguments:
        qubits:
            Target qubits.
        processed_data_dict_0:
            Population data for handle ``_0``, keyed by qubit UID.
        processed_data_dict_1:
            Population data for handle ``_1``, keyed by qubit UID.
        options:
            Plot options (save, close figures).

    Returns:
        Dictionary keyed by ``"{qubit_uid}_{handle}"`` with figures as values.
    """
    opts = PlotPopulationOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}

    for q in qubits:
        for label, data_dict in (
            ("_0", processed_data_dict_0),
            ("_1", processed_data_dict_1),
        ):
            data = data_dict[q.uid]
            phases = np.asarray(data["sweep_points_1d"])
            frequencies = np.asarray(data["sweep_points_2d"])
            num_cal_traces = data["num_cal_traces"]
            population = np.asarray(
                data["population"]
                if (num_cal_traces > 0 and not opts.do_pca)
                else data["data_raw"]
                if opts.do_rotation is False
                else data["population"]
            )

            fig, ax = plt.subplots()
            mesh = ax.pcolormesh(
                phases / np.pi,
                frequencies * 1e-6,
                population,
                shading="auto",
                cmap="RdBu_r",
            )
            fig.colorbar(
                mesh,
                ax=ax,
                label="Principal Component (a.u.)"
                if (num_cal_traces == 0 or opts.do_pca)
                else f"$|{opts.cal_states[-1]}\\rangle$-State Population",
            )
            ax.set_xlabel("RZ phase, $\\varphi$ ($\\pi$ rad)")
            ax.set_ylabel("CZ frequency, $f$ (MHz)")
            ax.set_title(timestamped_title(f"CZ freq vs RZ phase - {q.uid} {label}"))

            fig_key = f"{q.uid}{label}"
            if opts.save_figures:
                workflow.save_artifact(f"cz_freq_vs_phase_population_{fig_key}", fig)
            if opts.close_figures:
                plt.close(fig)

            figures[fig_key] = fig

    return figures


_MIN_POLY_FIT_POINTS = 4


@workflow.task
def fit_conditional_phase(
    fit_results: dict[str, dict],
) -> dict[str, dict]:
    """Fit a degree-3 polynomial to the conditional phase to find the CZ frequency.

    Computes the conditional phase delta_phi = phi0(_1) - phi0(_0) and fits it
    with a degree-3 polynomial to extract the CZ frequency where delta_phi = pi.

    Arguments:
        fit_results:
            Output of [fit_phase_oscillations](), keyed by qubit UID.

    Returns:
        Dictionary keyed by qubit UID. Each value contains:

        - ``frequencies``: valid CZ frequency array (non-NaN points) [N].
        - ``delta_phi``: conditional phase Δφ [N] (rad).
        - ``delta_phi_err``: propagated uncertainty [N] (rad).
        - ``poly_coeffs``: degree-3 polynomial coefficients (highest power first).
        - ``optimal_frequency``: frequency where polynomial crosses π (Hz), or NaN.
    """
    poly_results = {}

    for q_uid, handle_results in fit_results.items():
        data_0 = handle_results["_0"]
        data_1 = handle_results["_1"]

        frequencies = np.asarray(data_0["frequencies"])
        phi_0 = np.asarray(data_0["phase_offsets"])
        phi_1 = np.asarray(data_1["phase_offsets"])
        err_0 = np.asarray(data_0["phase_offsets_err"])
        err_1 = np.asarray(data_1["phase_offsets_err"])

        delta_phi = phi_1 - phi_0
        delta_phi_err = np.sqrt(err_0**2 + err_1**2)

        valid = ~np.isnan(delta_phi)
        freqs_v = frequencies[valid]
        delta_phi_v = delta_phi[valid]
        delta_phi_err_v = delta_phi_err[valid]

        optimal_frequency = np.nan
        poly_coeffs = None

        if len(freqs_v) >= _MIN_POLY_FIT_POINTS:
            poly_coeffs = np.polyfit(freqs_v, delta_phi_v, deg=3)

            # Find roots of poly(f) - π = 0 by shifting the constant term
            shifted = poly_coeffs.copy()
            shifted[-1] -= np.pi
            roots = np.roots(shifted)

            # Keep only real roots within the swept frequency range
            real_roots = roots[
                np.abs(roots.imag) < 1e-6 * np.abs(roots.real + 1e-30)
            ].real
            in_range = real_roots[
                (real_roots >= freqs_v.min()) & (real_roots <= freqs_v.max())
            ]

            if len(in_range) > 0:
                # If multiple crossings, pick the one closest to the centre
                centre = float((freqs_v.min() + freqs_v.max()) / 2)
                optimal_frequency = float(
                    in_range[np.argmin(np.abs(in_range - centre))]
                )

        poly_results[q_uid] = {
            "frequencies": freqs_v,
            "delta_phi": delta_phi_v,
            "delta_phi_err": delta_phi_err_v,
            "poly_coeffs": poly_coeffs,
            "optimal_frequency": optimal_frequency,
        }

    return poly_results


@workflow.task
def plot_phase_offset_vs_frequency(
    poly_results: dict[str, dict],
    options: PlotPopulationOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot the conditional phase Δφ vs. CZ frequency with polynomial fit overlay.

    Shows the data points with error bars, the degree-3 polynomial fit, a
    horizontal reference line at Δφ = π, and a vertical line at the extracted
    optimal frequency.

    Arguments:
        poly_results:
            Output of [fit_conditional_phase](), keyed by qubit UID.
        options:
            Plot options (save, close figures).

    Returns:
        Dictionary keyed by qubit UID with the corresponding figures.
    """
    opts = PlotPopulationOptions() if options is None else options
    figures = {}

    for q_uid, data in poly_results.items():
        freqs_v = np.asarray(data["frequencies"])
        delta_phi_v = np.asarray(data["delta_phi"])
        delta_phi_err_v = np.asarray(data["delta_phi_err"])
        poly_coeffs = data["poly_coeffs"]
        optimal_frequency = data["optimal_frequency"]

        fig, ax = plt.subplots()
        ax.errorbar(
            freqs_v * 1e-6,
            delta_phi_v / np.pi,
            yerr=delta_phi_err_v / np.pi,
            fmt="o",
            capsize=3,
            label="Data",
        )

        if poly_coeffs is not None:
            f_dense = np.linspace(freqs_v.min(), freqs_v.max(), 300)
            poly_vals = np.polyval(poly_coeffs, f_dense)
            ax.plot(f_dense * 1e-6, poly_vals / np.pi, "-", label="Degree-3 polynomial")

        ax.axhline(
            1.0, color="gray", linestyle="--", linewidth=0.8, label="$\\pi$ target"
        )

        if not np.isnan(optimal_frequency):
            ax.axvline(
                optimal_frequency * 1e-6,
                color="red",
                linestyle="--",
                label=f"$f_{{opt}}$ = {optimal_frequency * 1e-6:.3f} MHz",
            )

        ax.set_xlabel("CZ frequency, $f$ (MHz)")
        ax.set_ylabel("Conditional phase, $\\Delta\\varphi_0$ ($\\pi$ rad)")
        ax.set_title(timestamped_title(f"Conditional phase vs. CZ frequency - {q_uid}"))
        ax.legend()

        if opts.save_figures:
            workflow.save_artifact(f"cz_conditional_phase_vs_frequency_{q_uid}", fig)
        if opts.close_figures:
            plt.close(fig)

        figures[q_uid] = fig

    return figures


@workflow.task
def extract_edge_parameters(
    edges: list[TopologyEdge],
    poly_results: dict[str, dict],
) -> dict:
    """Extract the optimal CZ frequency (where conditional phase = π) per edge.

    Arguments:
        edges:
            The topology edges of the calibrated qubit pairs.
        poly_results:
            Output of [fit_conditional_phase](), keyed by target qubit UID.

    Returns:
        Dictionary with ``new_parameter_values`` and ``old_parameter_values``,
        each keyed by ``("cz", source_uid, target_uid)`` and containing the dotted
        parameter path ``"coupler_pulse.frequency"`` compatible with ``qpu.update()``.
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
        }

        par = poly_results.get(e.target_node.uid)
        if par is not None and not np.isnan(par["optimal_frequency"]):
            edge_parameters["new_parameter_values"][
                ("cz", e.source_node.uid, e.target_node.uid)
            ] = {
                "coupler_pulse.frequency": par["optimal_frequency"],
            }

    return edge_parameters
