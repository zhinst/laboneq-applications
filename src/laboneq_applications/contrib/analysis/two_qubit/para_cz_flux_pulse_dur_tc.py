# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""This module defines the analysis for a parametric CZ pulse-duration calibration.

In this analysis, the raw data is first interpreted into qubit population using
principal-component analysis or rotation and projection on the measured calibration
states. The target-qubit population vs flux pulse duration is fitted with a cosine
to extract the |11><->|20> oscillation frequency nu, and the optimal coupler-pulse
length is taken as 1/nu (one full Rabi period, i.e. a 2pi gate).
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
    calculate_qubit_population,
)
from laboneq_applications.analysis.options import (
    PlotPopulationOptions,
    TuneUpAnalysisWorkflowOptions,
)
from laboneq_applications.analysis.plotting_helpers import (
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
    durations: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The parametric CZ pulse-duration analysis workflow.

    The workflow consists of the following steps:

    - [validate_and_extract_edges_from_qubit_pairs]()
    - [extract_nodes_from_edges]()
    - [calculate_qubit_population]()
    - [fit_oscillation]()
    - [extract_edge_parameters]()
    - [plot_population]()

    Arguments:
        result:
            The experiment results returned by the `run_experiment` task.
        qpu:
            The quantum processing unit.
        qubit_pairs:
            The qubit pairs on which to run the analysis. The UIDs of these qubits
            must exist in the result.
        durations:
            Duration (length) values swept for the flux pulse on the coupler,
            per qubit pair.
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

    processed_data_dict_target = calculate_qubit_population(
        qubits=qubits_target,
        result=result,
        sweep_points=durations,
    )

    fit_results = fit_oscillation(qubits_target, processed_data_dict_target)

    qubit_parameters = extract_edge_parameters(edges, fit_results)

    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_qubit_population_plotting):
            plot_population(qubits_target, processed_data_dict_target, fit_results)

    workflow.return_(qubit_parameters)


@workflow.task
def extract_edge_parameters(
    edges: list[TopologyEdge],
    fit_results: dict[str, dict | None],
) -> dict:
    """Extract the optimal coupler pulse length from oscillation fit results.

    Arguments:
        edges:
            The topology edges of the calibrated qubit pairs.
        fit_results:
            The dictionary returned by [fit_oscillation](), keyed by target qubit UID.

    Returns:
        Dictionary with ``new_parameter_values`` and ``old_parameter_values``, each
        keyed by ``("cz", source_uid, target_uid)`` and containing the dotted parameter
        path ``"coupler_pulse.length"`` compatible with ``qpu.update()``.
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
            "coupler_pulse.length": e.parameters.coupler_pulse["length"],
        }

        par = fit_results.get(e.target_node.uid)
        if par is not None:
            edge_parameters["new_parameter_values"][
                ("cz", e.source_node.uid, e.target_node.uid)
            ] = {
                "coupler_pulse.length": par["opt_length"],
            }

    return edge_parameters


@workflow.task
def fit_oscillation(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
) -> dict[str, dict[str, ArrayLike]]:
    """Fit population vs. flux-duration trace with a cosine to extract oscillation freq.

    The qubit population vs duration trace is fitted
    with the model:
        P(t) = A * cos(2pi * nu * t + phi) + C

    The oscillation frequency nu is extracted.

    Arguments:
        qubits:
            The qubits on which to run the task.
        processed_data_dict:
            The processed data dictionary returned by [calculate_qubit_population]().

    Returns:
        Dictionary with qubit UIDs as keys. Each value is ``None`` if the fit
        failed, otherwise a dict containing:
            - ``fit_result``: the raw lmfit ModelResult.
            - ``opt_length``: optimal pulse length (1/nu) for the CZ pulse.
            - ``osc_frequencies``: fitted oscillation frequency nu.
            - ``osc_frequencies_err``: fit uncertainty (stderr), NaN on failure.
            - ``population``: the measured population.
            - ``durations``: the swept flux pulse durations.
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
        durations = np.asarray(processed_data_dict[q.uid]["sweep_points"])
        data = np.asarray(processed_data_dict[q.uid]["population"])

        dt = durations[1] - durations[0] if len(durations) > 1 else 1.0

        # FFT-based initial guess for oscillation frequency
        fft_vals = np.abs(np.fft.rfft(data - np.mean(data)))
        fft_freqs = np.fft.rfftfreq(len(data), d=dt)
        # Skip DC bin (index 0)
        peak_idx = int(np.argmax(fft_vals[1:])) + 1
        freq_guess = float(fft_freqs[peak_idx])

        amp_guess = float((np.max(data) - np.min(data)) / 2)
        offset_guess = float(np.mean(data))

        params = model.make_params(
            A={"value": amp_guess, "min": 0.0, "max": 1.0},
            osc_freq={"value": freq_guess, "min": 0.0},
            phase={"value": 0.0, "min": -np.pi, "max": np.pi},
            offset={"value": offset_guess, "min": 0.0, "max": 1.0},
        )

        try:
            fit = model.fit(data, params, t=durations)
            osc_freqs = fit.params["osc_freq"].value
            opt_length = 1.0 / osc_freqs
            stderr = fit.params["osc_freq"].stderr
            osc_freqs_err = stderr if stderr is not None else np.nan

            fit_results[q.uid] = {
                "fit_result": fit,
                "opt_length": opt_length,
                "osc_frequencies": osc_freqs,
                "osc_frequencies_err": osc_freqs_err,
                "population": data,
                "durations": durations,
            }

        except (ValueError, RuntimeError):
            fit_results[q.uid] = None

    return fit_results


@workflow.task
def plot_population(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, dict | None],
    options: PlotPopulationOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the population plots.

    Arguments:
        qubits:
            The qubits on which to run the task. The UIDs of these qubits must exist in
            `processed_data_dict`.
        processed_data_dict:
            The processed data dictionary returned by [calculate_qubit_population]().
        fit_results:
            The dictionary returned by [fit_oscillation](), keyed by qubit UID.
        options:
            The options for processing the raw data.

    Returns:
        Dictionary with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = PlotPopulationOptions() if options is None else options
    figures = {}

    for q_uid, fit_data in fit_results.items():
        if fit_data:
            durations = np.asarray(fit_data["durations"])
            population = np.asarray(fit_data["population"])

            fig, ax = plt.subplots()

            # Plot measured oscillation frequencies with error bars
            ax.scatter(
                durations * 1e9,
                population,
            )

            par = fit_results.get(q_uid)
            # Dense frequency axis for smooth parabola curve
            dur_dense = np.linspace(durations[0], durations[-1], 301)
            pop_fit = par["fit_result"].eval(t=dur_dense)
            ax.plot(dur_dense * 1e9, pop_fit, "-", label="Fit")

            opt_length = fit_data["opt_length"]

            opt_length_label = f"$l_{{opt}}$ = {opt_length * 1e9:.3f} ns"

            ax.axvline(
                opt_length * 1e9,
                color="red",
                linestyle="--",
                label=opt_length_label,
            )

        ax.set_xlabel("Flux pulse length, (ns)")
        ax.set_ylabel("Population, (au)")
        ax.set_title(timestamped_title(f"CZ duration vs. qubit population - {q_uid}"))
        ax.legend(fontsize=8)

        if opts.save_figures:
            workflow.save_artifact(f"chevron_frequency_parabola_{q_uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q_uid] = fig

    return figures
