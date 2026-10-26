# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the analysis of the qubit spectroscopy versus flux data.

This analysis plots the raw data and the signal magnitude and phase as 2D maps of
the qubit frequency and the flux-pulse amplitude. It also traces the frequency of
the spectroscopy dip in every flux slice and returns the frequency shift between
the smallest and the largest flux-pulse amplitude.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.simple import Results
from laboneq.workflow.tasks.run_experiment import RunExperimentResults

from laboneq_applications.analysis.options import BasePlottingOptions
from laboneq_applications.analysis.plotting_helpers import (
    _get_raw_data_collection,
    plot_raw_complex_data_2d,
    plot_signal_magnitude_and_phase_2d,
    timestamped_title,
)
from laboneq_applications.analysis.spectroscopy_two_dimensional_plotting import (
    Spectroscopy2DPlottingWorkflowOptions,
)
from laboneq_applications.conversion import as_quantum_element_list

if TYPE_CHECKING:
    import matplotlib as mpl
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.workflow_options(base_class=Spectroscopy2DPlottingWorkflowOptions)
class QubitSpectroscopyQFluxAnalysisWorkflowOptions:
    """Option class for the qubit spectroscopy versus flux analysis workflow.

    Attributes:
        do_plotting_each_slice:
            Whether to plot the frequency of the dip in every flux slice.
            Default: True.
    """

    do_plotting_each_slice: bool = workflow.option_field(
        True,
        description=(
            "Whether to plot the minimum value (dip) for each slice to make "
            "a parabola around the optimal value."
        ),
    )


@workflow.workflow(name="analysis_workflow")
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    sweep_points_1d: QubitSweepPoints,
    sweep_points_2d: QubitSweepPoints,
    label_sweep_points_1d: str,
    label_sweep_points_2d: str,
    scaling_sweep_points_2d: float = 1.0,
    options: QubitSpectroscopyQFluxAnalysisWorkflowOptions | None = None,
) -> None:
    """The analysis Workflow for plotting spectroscopy 2D data.

    The workflow consists of the following steps:

    - [plot_raw_complex_data_2d][]
    - [plot_signal_magnitude_and_phase_2d][]
    - [trace_spectroscopy_dips][]
    - [plot_spectroscopy_dips][]
    - [extract_frequency_shift][]

    Arguments:
        result:
            The experiment results returned by the `run_experiment` task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        sweep_points_1d:
            The sweep points corresponding to the innermost sweep.
            If `qubits` is a single qubit, `sweep_points_1d` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        sweep_points_2d:
            The sweep points corresponding to the outermost sweep.
            If `qubits` is a single qubit, `sweep_points_2d` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        label_sweep_points_1d:
            The label that will appear on the axis of the 1D sweep points.
            Passed to the `label_sweep_points_1d` parameter of
            `plot_raw_complex_data_2d`.
        label_sweep_points_2d:
            The label that will appear on the axis of the 2D sweep points.
            Passed to the `label_sweep_points_2d` parameter of
            `plot_raw_complex_data_2d`.
        scaling_sweep_points_2d:
            The scaling factor of the 2D sweep points.
            Passed to the `scaling_sweep_points_2d` parameter of
            `plot_raw_complex_data_2d`.
            Default: 1.0.
        options:
            The options for building the workflow, passed as an instance of
            [QubitSpectroscopyQFluxAnalysisWorkflowOptions][]. See the docstring of this
            class for more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        result = analysis_workflow(
            result=result,
            qubits=[q0, q1],
            sweep_points_1d=[np.linspace(6.0, 6.3, 301), np.linspace(5.8, 6.1, 301)],
            sweep_points_2d=[np.linspace(0, 1, 11), np.linspace(0, 1, 11)],
            label_sweep_points_1d="Qubit Frequency (Hz)",
            label_sweep_points_2d="Flux-Pulse Amplitude (a.u.)",
            options=analysis_workflow.options(),
        ).run()
        ```
    """
    with workflow.if_(options.do_raw_data_plotting):
        plot_raw_complex_data_2d(
            qubits=qubits,
            result=result,
            sweep_points_1d=sweep_points_1d,
            sweep_points_2d=sweep_points_2d,
            label_sweep_points_1d=label_sweep_points_1d,
            label_sweep_points_2d=label_sweep_points_2d,
            scaling_sweep_points_1d=1e-9,
            scaling_sweep_points_2d=scaling_sweep_points_2d,
        )
    with workflow.if_(options.do_plotting_magnitude_phase):
        plot_signal_magnitude_and_phase_2d(
            qubits=qubits,
            result=result,
            sweep_points_1d=sweep_points_1d,
            sweep_points_2d=sweep_points_2d,
            label_sweep_points_1d=label_sweep_points_1d,
            label_sweep_points_2d=label_sweep_points_2d,
            scaling_sweep_points_1d=1e-9,
            scaling_sweep_points_2d=scaling_sweep_points_2d,
        )
    spectroscopy_dips = trace_spectroscopy_dips(qubits=qubits, result=result)
    with workflow.if_(options.do_plotting_each_slice):
        plot_spectroscopy_dips(
            qubits=qubits,
            spectroscopy_dips=spectroscopy_dips,
            label_sweep_points_2d=label_sweep_points_2d,
        )
    frequency_shift = extract_frequency_shift(qubits, spectroscopy_dips)

    workflow.return_(frequency_shift)


@workflow.task
def trace_spectroscopy_dips(
    qubits: QuantumElements,
    result: RunExperimentResults | Results,
) -> dict[str, dict[str, ArrayLike]]:
    """Find the frequency of the spectroscopy dip in every flux slice.

    Returns the flux-pulse amplitudes and the frequency at which the signal
    magnitude is smallest for each of them, for every qubit.
    """
    qubits = as_quantum_element_list(qubits)

    raw_result = result
    if isinstance(result, Results):
        raw_result = RunExperimentResults(data=result.acquired_results)

    spectroscopy_dips = {}
    for q in qubits:
        raw_data_collection = _get_raw_data_collection(raw_result, q)
        if len(raw_data_collection) > 1:
            raise ValueError(
                f"The result for qubit {q.uid} contains more than one acquired "
                "result. This analysis expects a single acquire handle per qubit."
            )
        _, acquired_results = raw_data_collection[0]

        # the data has the shape len(flux_amplitude) x len(frequency)
        frequencies = np.asarray(acquired_results.axis[1])
        minimum_indices = np.argmin(np.abs(acquired_results.data), axis=1)
        spectroscopy_dips[q.uid] = {
            "flux_amplitudes": np.asarray(acquired_results.axis[0]),
            "dip_frequencies": frequencies[minimum_indices],
        }
    return spectroscopy_dips


@workflow.task
def plot_spectroscopy_dips(
    qubits: QuantumElements,
    spectroscopy_dips: dict[str, dict[str, ArrayLike]],
    label_sweep_points_2d: str,
    options: BasePlottingOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot the frequency of the spectroscopy dip against the flux-pulse amplitude."""
    opts = BasePlottingOptions() if options is None else options
    qubits = as_quantum_element_list(qubits)

    figures = {}
    for q in qubits:
        fig, ax = plt.subplots()
        ax.plot(
            spectroscopy_dips[q.uid]["flux_amplitudes"],
            spectroscopy_dips[q.uid]["dip_frequencies"],
        )
        ax.set_xlabel(label_sweep_points_2d)
        ax.set_ylabel("Min. Qubit Frequency (Hz)")
        ax.set_title(timestamped_title(f"Qubit Freq vs Flux, {q.uid}"))

        if opts.save_figures:
            workflow.save_artifact(f"Qubit_frequency_vs_flux_{q.uid}", fig)
        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig
    return figures


@workflow.task
def extract_frequency_shift(
    qubits: QuantumElements,
    spectroscopy_dips: dict[str, dict[str, ArrayLike]],
) -> dict[str, float]:
    """Return the frequency shift between the smallest and largest flux amplitude."""
    qubits = as_quantum_element_list(qubits)

    frequency_shift = {}
    for q in qubits:
        flux_amplitudes = spectroscopy_dips[q.uid]["flux_amplitudes"]
        dip_frequencies = spectroscopy_dips[q.uid]["dip_frequencies"]
        smallest_amplitude_idx = np.argmin(flux_amplitudes)
        largest_amplitude_idx = np.argmax(flux_amplitudes)
        frequency_shift[q.uid] = np.abs(
            dip_frequencies[largest_amplitude_idx]
            - dip_frequencies[smallest_amplitude_idx]
        )
    return frequency_shift
