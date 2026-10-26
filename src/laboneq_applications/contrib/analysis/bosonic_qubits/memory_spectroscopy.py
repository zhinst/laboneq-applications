# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Analysis for the memory spectroscopy experiment.

When the spectroscopy tone is resonant with the memory cavity, the photons it
creates shift the transmon frequency, so the subsequent transmon pi pulse no
longer inverts the transmon. The resonance therefore appears as a dip in the
magnitude of the readout signal. A Lorentzian is fitted to the magnitude, and
its position gives `memory_resonance_frequency`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.simple import dsl

from laboneq_applications.analysis.fitting_helpers import lorentzian_fit
from laboneq_applications.analysis.options import BasePlottingOptions
from laboneq_applications.conversion import (
    as_quantum_element_list,
    sweeps_per_quantum_element,
)

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


# ═════════════════════════════════════════════════════════════
# Options
# ═════════════════════════════════════════════════════════════


@workflow.workflow_options
class MemorySpectroscopyAnalysisWorkflowOptions:
    """Options for the memory spectroscopy analysis workflow.

    Attributes:
        do_plotting:
            Whether to create plots.
            Default: True.
    """

    do_plotting: bool = workflow.option_field(
        True, description="Whether to create plots."
    )


@workflow.task_options
class MemorySpectroscopyFitDataOptions:
    """Options for the `fit_data` task of the memory spectroscopy analysis.

    Attributes:
        spectral_feature:
            Whether the resonance is fitted as a "dip" or a "peak" in the
            signal magnitude, or "auto" to work it out from the data.
            Default: "dip".
        num_fit_points:
            Number of points for the fit curve. Default: 501.
    """

    spectral_feature: Literal["peak", "dip", "auto"] = workflow.option_field(
        "dip", description="Whether to fit the resonance as a dip or a peak."
    )
    num_fit_points: int = workflow.option_field(
        501, description="Number of points for the fit curve."
    )


# ═════════════════════════════════════════════════════════════
# Workflow
# ═════════════════════════════════════════════════════════════


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    frequencies: QubitSweepPoints,
    options: MemorySpectroscopyAnalysisWorkflowOptions | None = None,
) -> None:
    """The memory spectroscopy analysis workflow.

    The workflow consists of the following steps:

    - [fit_data][]
    - [extract_qubit_parameters][]
    - [plot_memory_spectroscopy][]

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        frequencies:
            The memory drive frequencies that were swept for each qubit.
        options:
            The options for building the workflow, passed as an instance of
            [MemorySpectroscopyAnalysisWorkflowOptions][].

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        analysis_result = analysis_workflow(
            result=result,
            qubits=q0,
            frequencies=np.linspace(5.9e9, 6.1e9, 1001),
        ).run()
        ```
    """
    processed_data_dict = fit_data(qubits, result, frequencies)
    qubit_parameters = extract_qubit_parameters(qubits, processed_data_dict)
    with workflow.if_(options.do_plotting):
        plot_memory_spectroscopy(qubits, processed_data_dict)
    workflow.return_(qubit_parameters)


# ═════════════════════════════════════════════════════════════
# Analysis tasks
# ═════════════════════════════════════════════════════════════


@workflow.task
def fit_data(
    qubits: QuantumElements,
    result: RunExperimentResults,
    frequencies: QubitSweepPoints,
    options: MemorySpectroscopyFitDataOptions | None = None,
) -> dict[str, dict]:
    """Fit a Lorentzian to the magnitude of the memory spectroscopy signal.

    Arguments:
        qubits:
            The qubits on which to run the analysis.
        result:
            The experiment results returned by the run_experiment task.
        frequencies:
            The memory drive frequencies that were swept for each qubit.
        options:
            The options for this task as an instance of
            [MemorySpectroscopyFitDataOptions][].

    Returns:
        Dictionary keyed by qubit UID. Each entry contains:

            - ``"frequencies"``: The frequency sweep array.
            - ``"raw_data"``: The complex readout signal.
            - ``"magnitude"``: The magnitude of the readout signal.
            - ``"memory_resonance_frequency"``: The fitted resonance position (NaN if the fit failed).
            - ``"linewidth"``: The fitted Lorentzian width (NaN if the fit failed).
            - ``"fit_params"``: The fitted Lorentzian parameters (empty if the fit failed).
            - ``"fit_success"``: Whether the fit succeeded.
            - ``"fit_x"``, ``"fit_y"``: Dense fit curve arrays for plotting.
    """
    opts = MemorySpectroscopyFitDataOptions() if options is None else options
    qubits = as_quantum_element_list(qubits)

    processed_data_dict = {}
    for q, q_freqs_raw in sweeps_per_quantum_element(qubits, frequencies):
        q_freqs = np.asarray(q_freqs_raw)
        data = np.asarray(result.get_data(dsl.handles.result_handle(q.uid)))
        magnitude = np.abs(data)
        fit_x = np.linspace(q_freqs[0], q_freqs[-1], opts.num_fit_points)

        try:
            fit_res = lorentzian_fit(
                q_freqs, magnitude, spectral_feature=opts.spectral_feature
            )
            fit_params = dict(fit_res.best_values)
            success = bool(fit_res.success)
            position = float(fit_params["position"])
            linewidth = float(fit_params["width"])
            fit_y = fit_res.model.func(fit_x, **fit_params)
        except (RuntimeError, ValueError, TypeError) as err:
            workflow.comment(f"Fit failed for {q.uid}: {err}.")
            fit_params = {}
            success = False
            position = np.nan
            linewidth = np.nan
            fit_y = np.full_like(fit_x, np.nan)

        processed_data_dict[q.uid] = {
            "frequencies": q_freqs,
            "raw_data": data,
            "magnitude": magnitude,
            "memory_resonance_frequency": position,
            "linewidth": linewidth,
            "fit_params": fit_params,
            "fit_success": success,
            "fit_x": fit_x,
            "fit_y": fit_y,
        }

    return processed_data_dict


@workflow.task
def extract_qubit_parameters(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict],
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Extract `memory_resonance_frequency` from the fit results.

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
                    q.uid: {"memory_resonance_frequency": value},
                },
                "old_parameter_values": {
                    q.uid: {"memory_resonance_frequency": value},
                },
            }
            ```

            If the fit failed for a qubit, or the fitted resonance lies outside
            the swept frequency range, its new_parameter_values entry is left empty.
    """
    qubits = as_quantum_element_list(qubits)
    qubit_parameters = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
    }
    for q in qubits:
        qubit_parameters["old_parameter_values"][q.uid] = {
            "memory_resonance_frequency": q.parameters.memory_resonance_frequency,
        }
        res = processed_data_dict[q.uid]
        position = res["memory_resonance_frequency"]
        freqs = res["frequencies"]
        in_range = np.min(freqs) <= position <= np.max(freqs)
        if res["fit_success"] and np.isfinite(position) and in_range:
            qubit_parameters["new_parameter_values"][q.uid] = {
                "memory_resonance_frequency": position,
            }
    return qubit_parameters


@workflow.task
def plot_memory_spectroscopy(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict],
    options: BasePlottingOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot the memory spectroscopy signal magnitude and the Lorentzian fit.

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
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.plot(
            res["frequencies"] / 1e9,
            res["magnitude"],
            "o",
            markersize=3,
            alpha=0.6,
            label="Data",
        )
        if res["fit_success"]:
            ax.plot(res["fit_x"] / 1e9, res["fit_y"], "-", linewidth=1.5, label="Fit")
        position = res["memory_resonance_frequency"]
        if np.isfinite(position):
            ax.axvline(
                position / 1e9,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"f_memory = {position / 1e9:.6f} GHz",
            )
        ax.set_xlabel("Memory drive frequency (GHz)")
        ax.set_ylabel("|S| (a.u.)")
        ax.set_title(f"{q.uid} — Memory spectroscopy", fontweight="bold")
        ax.legend(fontsize=8)
        _save_and_close(fig, f"memory_spectroscopy_{q.uid}", opts)
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
