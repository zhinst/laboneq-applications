# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Analysis for the single-qubit phase correction after a parametric CZ gate.

For each qubit pair, two Ramsey-like measurements are analysed:

  - Handle ``_0``: target qubit phase sweep (source in |0⟩)
  - Handle ``_1``: source qubit phase sweep (target in |0⟩)

The population vs. phase is fitted with a cosine using the linear
parameterisation:

    P(φ) = a·cos(φ) + b·sin(φ) + C  →  A = √(a²+b²),  φ₀ = atan2(-b, a)

The extracted phase offset φ₀ is the single-qubit phase correction that
must be applied after the CZ gate to restore each qubit to its intended state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.dsl.quantum import QPU, QuantumElement
from laboneq.dsl.quantum.qpu_topology import TopologyEdge

from laboneq_applications.analysis.calibration_traces_rotation import (
    calculate_qubit_population,
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
    phases: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """Analysis workflow for single-qubit phase correction after a parametric CZ gate.

    The workflow consists of the following steps:

    - [calculate_qubit_population]() for target qubits (handle ``_0``)
    - [calculate_qubit_population]() for source qubits (handle ``_1``)
    - [fit_phase_oscillations]()
    - [extract_edge_parameters]()
    - [plot_population]()

    Arguments:
        result:
            Experiment results from ``run_experiment``.
        qpu:
            The quantum processing unit.
        qubit_pairs:
            Qubit pairs used in the experiment.
        phases:
            RZ phase sweep points per qubit pair.
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
    qubits_source = extract_nodes_from_edges(edges, "source")

    # Target qubit: measured in section 0 (source in |0⟩)
    processed_data_target = calculate_qubit_population(
        qubits=qubits_target,
        result=result,
        sweep_points=phases,
    )
    # Source qubit: measured in section 1 (target in |0⟩)
    processed_data_source = calculate_qubit_population(
        qubits=qubits_source,
        result=result,
        sweep_points=phases,
    )

    fit_results = fit_phase_oscillations(
        qubits_target, qubits_source, processed_data_target, processed_data_source
    )

    qubit_parameters = extract_edge_parameters(edges, fit_results)

    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_qubit_population_plotting):
            plot_population(
                qubits_target,
                qubits_source,
                processed_data_target,
                processed_data_source,
                fit_results,
            )

    workflow.return_(qubit_parameters)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@workflow.task
def fit_phase_oscillations(
    qubits_target: QuantumElements,
    qubits_source: QuantumElements,
    processed_data_target: dict[str, dict],
    processed_data_source: dict[str, dict],
    amplitude_threshold: float = 0.05,
) -> dict[str, dict]:
    """Fit population vs. RZ phase with a cosine for each qubit.

    Uses a linear parameterisation for a unique, global solution:

        P(φ) = a·cos(φ) + b·sin(φ) + C

    Phase offset and amplitude are recovered as:

        A   = √(a² + b²)
        φ₀  = atan2(-b, a)

    Arguments:
        qubits_target:
            Target qubits (measured in section 0).
        qubits_source:
            Source qubits (measured in section 1).
        processed_data_target:
            Population data for target qubits, keyed by qubit UID.
            Each value has ``sweep_points`` (phases) and ``population``.
        processed_data_source:
            Population data for source qubits, keyed by qubit UID.
        amplitude_threshold:
            Minimum amplitude A below which the phase offset is set to NaN.

    Returns:
        Dictionary keyed by qubit UID (string). Each qubit gets its own entry
        containing its fit results and the UID of its paired qubit:

        - ``phase_offset``: fitted φ₀ (rad), or NaN.
        - ``phase_offset_err``: uncertainty of φ₀ (rad), or NaN.
        - ``amplitude``: fitted amplitude A.
        - ``phases``: phase sweep points.
        - ``population``: measured population.
        - ``paired_uid``: UID of the other qubit in the pair.
    """
    qubits_target = validate_and_convert_qubits_sweeps(qubits_target)
    qubits_source = validate_and_convert_qubits_sweeps(qubits_source)

    def _fit_single(phases: np.ndarray, population: np.ndarray) -> dict:
        """Fit one population vs phase trace, return phase offset and amplitude."""
        X = np.column_stack(  # noqa: N806
            [np.cos(phases), np.sin(phases), np.ones(len(phases))]
        )
        dof = len(phases) - 3

        coeffs, _, _, _ = np.linalg.lstsq(X, population, rcond=None)
        a, b = coeffs[0], coeffs[1]
        A = float(np.sqrt(a**2 + b**2))  # noqa: N806

        if amplitude_threshold > A or dof < 1:
            return {"phase_offset": np.nan, "phase_offset_err": np.nan, "amplitude": A}

        phi_0 = float(np.arctan2(-b, a))

        # Uncertainty via delta method from least-squares covariance
        rss = float(np.sum((population - X @ coeffs) ** 2))
        sigma2 = rss / dof
        try:
            cov = sigma2 * np.linalg.inv(X.T @ X)
            da = b / A**2
            db = -a / A**2
            var_phi0 = da**2 * cov[0, 0] + db**2 * cov[1, 1] + 2 * da * db * cov[0, 1]
            phi_0_err = float(np.sqrt(max(var_phi0, 0.0)))
        except np.linalg.LinAlgError:
            phi_0_err = np.nan

        return {"phase_offset": phi_0, "phase_offset_err": phi_0_err, "amplitude": A}

    fit_results = {}
    for q_t, q_s in zip(qubits_target, qubits_source, strict=False):
        phases_t = np.asarray(processed_data_target[q_t.uid]["sweep_points"])
        pop_t = np.asarray(processed_data_target[q_t.uid]["population"])

        phases_s = np.asarray(processed_data_source[q_s.uid]["sweep_points"])
        pop_s = np.asarray(processed_data_source[q_s.uid]["population"])

        fit_results[q_t.uid] = {
            **_fit_single(phases_t, pop_t),
            "phases": phases_t,
            "population": pop_t,
            "paired_uid": q_s.uid,
        }
        fit_results[q_s.uid] = {
            **_fit_single(phases_s, pop_s),
            "phases": phases_s,
            "population": pop_s,
            "paired_uid": q_t.uid,
        }

    return fit_results


@workflow.task
def extract_edge_parameters(
    edges: list[TopologyEdge],
    fit_results: dict[str, dict],
) -> dict:
    """Extract single-qubit phase corrections for each edge.

    Arguments:
        edges:
            The topology edges of the calibrated qubit pairs.
        fit_results:
            Output of [fit_phase_oscillations](), keyed by qubit UID.

    Returns:
        Dictionary with ``new_parameter_values`` and ``old_parameter_values``,
        each keyed by ``("cz", source_uid, target_uid)`` and containing the
        parameter paths ``"target_angle"`` (phase correction for the target qubit)
        and ``"control_angle"`` (phase correction for the source qubit), compatible
        with ``qpu.update()``.
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
            "target_angle": e.parameters.target_angle,
            "control_angle": e.parameters.control_angle,
        }

        par_target = fit_results.get(e.target_node.uid)
        par_source = fit_results.get(e.source_node.uid)

        phi_target = par_target["phase_offset"] if par_target is not None else np.nan
        phi_source = par_source["phase_offset"] if par_source is not None else np.nan

        if not np.isnan(phi_target) or not np.isnan(phi_source):
            edge_parameters["new_parameter_values"][
                ("cz", e.source_node.uid, e.target_node.uid)
            ] = {
                "target_angle": phi_target,
                "control_angle": phi_source,
            }

    return edge_parameters


@workflow.task
def plot_population(
    qubits_target: QuantumElements,
    qubits_source: QuantumElements,
    processed_data_target: dict[str, dict],
    processed_data_source: dict[str, dict],
    fit_results: dict[tuple, dict],
    options: PlotPopulationOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot population vs. RZ phase with fit overlay for each qubit.

    One figure per qubit pair, with two subplots (target and source).

    Arguments:
        qubits_target:
            Target qubits.
        qubits_source:
            Source qubits.
        processed_data_target:
            Population data for target qubits.
        processed_data_source:
            Population data for source qubits.
        fit_results:
            Output of [fit_phase_oscillations]().
        options:
            Plot options (save, close figures).

    Returns:
        Dictionary keyed by ``"source_uid-target_uid"`` with figures as values.
    """
    opts = PlotPopulationOptions() if options is None else options
    qubits_target = validate_and_convert_qubits_sweeps(qubits_target)
    qubits_source = validate_and_convert_qubits_sweeps(qubits_source)
    figures = {}

    for q_t, q_s in zip(qubits_target, qubits_source, strict=False):
        par_t = fit_results.get(q_t.uid)
        par_s = fit_results.get(q_s.uid)
        if par_t is None and par_s is None:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(timestamped_title(f"SQ phase correction - {q_s.uid} -> {q_t.uid}"))

        for ax, qubit_label, par, role in (
            (axes[0], q_t.uid, par_t, "target"),
            (axes[1], q_s.uid, par_s, "source"),
        ):
            if par is None:
                ax.set_title(f"{qubit_label} ({role}) - no data")
                continue

            phases = par["phases"]
            population = par["population"]

            ax.scatter(phases / np.pi, population, s=20, label="Data", zorder=3)

            if not np.isnan(par["phase_offset"]):
                phi_dense = np.linspace(phases[0], phases[-1], 300)
                A = par["amplitude"]  # noqa: N806
                phi_0 = par["phase_offset"]
                C = float(np.mean(population))  # noqa: N806
                pop_fit = A * np.cos(phi_dense + phi_0) + C
                ax.plot(phi_dense / np.pi, pop_fit, "-", label="Fit")

                phi_0_label = f"$\\varphi_0$ = {phi_0 / np.pi:.3f}$\\pi$"
                if not np.isnan(par["phase_offset_err"]):
                    phi_0_label += f" ± {par['phase_offset_err'] / np.pi:.3f}$\\pi$"
                ax.axvline(
                    -phi_0 / np.pi,
                    color="red",
                    linestyle="--",
                    linewidth=0.9,
                    label=phi_0_label,
                )

            ax.set_xlabel("RZ phase, $\\varphi$ ($\\pi$ rad)")
            ax.set_ylabel("Population (a.u.)")
            ax.set_title(f"{qubit_label} ({role})")
            ax.legend(fontsize=8)

        fig.tight_layout()

        fig_key = f"{q_s.uid}-{q_t.uid}"
        if opts.save_figures:
            workflow.save_artifact(f"cz_sq_phase_correction_{fig_key}", fig)
        if opts.close_figures:
            plt.close(fig)

        figures[fig_key] = fig

    return figures
