# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the analysis for a resonator photon number time resolution
experiment.

The experiment is defined in laboneq_applications.experiments.

In this analysis, we determine the time resolved resonator photon number during
readout. The photon number is determined by tracking the qubits frequency during
readout.
"""  # noqa: D205

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq.simple import dsl
from laboneq.workflow import (
    if_,
    return_,
    save_artifact,
    task,
    workflow,
)

from laboneq_applications.analysis import qubit_spectroscopy
from laboneq_applications.analysis.options import (
    BasePlottingOptions,
    TuneUpAnalysisWorkflowOptions,
)
from laboneq_applications.analysis.plotting_helpers import (
    plot_data_2d,
    timestamped_title,
)
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_result,
)

if TYPE_CHECKING:
    import lmfit
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    times: QubitSweepPoints,
    frequencies: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The photon number time resolution analysis workflow.

    The workflow consists of the following steps:
    - [calculate_signal_magnitudes_and_phases]()
    - [fit_data]()
    - [plot_resonator_photon_number]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit
            or a list of qubits. The UIDs of these qubits must exist in the result.
        times:
            The array of times that were swept over the measurement pulse in the
            experiment. If `qubits` is a single qubit, `times` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers
            or arrays.
        frequencies:
            The array of frequencies that were swept over in the experiment. If
            `qubits` is a single qubit, `frequencies` must be a list of numbers
            or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        options:
            The options for building the workflow, passed as an instance of
            [TuneUpAnalysisWorkflowOptions]. See the docstring of this class for
            more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        result = analysis_workflow(
            results=results
            qubits=[q0, q1],
            times=[
                np.linspace(0, 3e-6, 21),
                np.linspace(0, 3e-6, 21),
            ],
            frequencies=[
                np.linspace(6.0e9, 6.3e9, 201),
                np.linspace(5.8e9, 6.1e9, 201),
            ],
            options=analysis_workflow.options(),
        ).run()
        ```
    """
    processed_data_dict = calculate_signal_magnitudes_and_phases(
        qubits, result, frequencies, times
    )
    fit_results = fit_data(qubits, processed_data_dict)
    with if_(options.do_plotting):
        plot_resonator_photon_number(qubits, processed_data_dict, fit_results)
    return_(fit_results)


@task
def calculate_signal_magnitudes_and_phases(
    qubits: QuantumElements,
    result: RunExperimentResults,
    frequencies: ArrayLike,
    times: ArrayLike,
) -> dict[str, dict[str, ArrayLike]]:
    """Calculates the magnitude and phase of the spectroscopy signal in result.

    Arguments:
        result:
            The experiment results returned by the `run_experiment` task.
        qubits:
            The qubits on which to run the task. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        frequencies:
            The array of frequencies that were swept over in the experiment.
        times:
            The array of times that were swept over in the experiment.

    Returns:
        Dictionary with the qubit UIDs as keys and a processed data dict as
        values, containing the following data:
            `sweep_points`
            `data_raw`
            `magnitude`
            `phase`
    """
    # qubits, times and frequencies must have the same length
    qubits_validated, frequencies = validate_and_convert_qubits_sweeps(
        qubits, frequencies
    )
    qubits_validated, times = validate_and_convert_qubits_sweeps(qubits, times)
    validate_result(result)

    proc_data_dict = {}
    for q, freqs, time in zip(qubits_validated, frequencies, times):
        raw_data = result[dsl.handles.result_handle(q.uid)].data
        proc_data_dict[q.uid] = {
            "sweep_points_freq": freqs,
            "sweep_points_time": time,
            "data_raw": raw_data,
            "magnitudes": np.abs(raw_data),
            "phases": np.angle(raw_data),
        }

    return proc_data_dict


@task
def fit_data(
    qubits: QuantumElements, processed_data_dict: dict
) -> dict[str, dict[str, ArrayLike]]:
    """Perform a qubit spectroscopy fit for every time.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit
            or a list of qubits. The UIDs of these qubits must exist in the
            `processed_data_dict`.
        processed_data_dict: The processed data dictionary returned by
            `calculate_signal_magnitude_and_phase`.

    Returns:
        Dictionary with qubit UIDs as keys and the fit results for each qubit as keys.
    """
    qubits = validate_and_convert_qubits_sweeps(qubits)

    fit_results = {}
    for q in qubits:
        fit_results[q.uid] = {}
        fit_by_time = []
        for i in range(len(processed_data_dict[q.uid]["sweep_points_time"])):
            magnitudes = processed_data_dict[q.uid]["magnitudes"][i]
            sweep_points = processed_data_dict[q.uid]["sweep_points_freq"]

            proc_data_dict = {
                q.uid: {"sweep_points": sweep_points, "magnitude": magnitudes}
            }

            fit_result = qubit_spectroscopy.fit_data([q], proc_data_dict)[q.uid]
            fit_by_time.append(fit_result)

        fit_results[q.uid]["qubit_frequency_at_time"] = fit_by_time

    return fit_results


@task(save=False)
def plot_resonator_photon_number(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, lmfit.model.ModelResult] | None,
    options: BasePlottingOptions | None = None,
) -> dict[str, dict[str, mpl.figure.Figure]]:
    """Create the time-resolved photon number plot and the 2D qubit spectroscopy plot.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit
            or a list of qubits. The UIDs of these qubits must exist in the
            `processed_data_dict` and `qubit_parameters`.
        processed_data_dict:
            The processed data dictionary returned by
            `calculate_signal_magnitude_and_phase`.
        fit_results: The fit-results dictionary returned by `fit_data`.
        qubit_parameters: The qubit-parameters dictionary returned by
            `extract_qubit_parameters`.
        options:
            The options for this task as an instance of
            [PlotQubitSpectroscopyOptions]. See the docstring of this class for
            more details.

    Returns:
        Dictionary with qubit UIDs as keys and the dictionary of "1D" and "2D" figures
        as values.
    """
    opts = BasePlottingOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}

    for q in qubits:
        figures[q.uid] = {}

        times = processed_data_dict[q.uid]["sweep_points_time"]
        qubit_frequency_at_time = np.array(
            [
                result.values["position"]
                for result in fit_results[q.uid]["qubit_frequency_at_time"]
            ]
        )

        fig_1d, ax1 = plt.subplots()

        chi = q.parameters.ge_chi_shift

        if chi:
            ax1.set_title(
                timestamped_title(f"Resonator Photon Number Ground State {q.uid}")
            )
            ax1.plot(
                times * 1e6,
                (qubit_frequency_at_time - qubit_frequency_at_time[0]) / (-2 * chi),
                "o",
                zorder=2,
                label="Data",
            )
            ax1.set_ylabel("Photon Number")
        else:
            ax1.set_title(
                timestamped_title(f"Qubit Spectroscopy During Readout {q.uid}")
            )
            ax1.plot(
                times * 1e6,
                qubit_frequency_at_time * 1e-9,
                "o",
                zorder=2,
                label="Data",
            )
            ax1.set_ylabel("Qubit Frequency (GHz)")
        ax1.set_xlabel("Time ($\\mu$s)")
        ax1.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), handlelength=1.5, frameon=False
        )
        figures[q.uid]["1D"] = fig_1d

        time = processed_data_dict[q.uid]["sweep_points_time"]
        frequency = processed_data_dict[q.uid]["sweep_points_freq"]
        # plot magnitude
        fig_2d, ax2 = plot_data_2d(
            x_values=time,
            y_values=frequency,
            z_values=processed_data_dict[q.uid]["magnitudes"].T,
            label_x_values="Time ($\\mu$s)",
            label_y_values="Frequency (GHz)",
            label_z_values="Spectroscopy Signal\nMagnitude (a.u.)",
            scaling_x_values=1e6,
            scaling_y_values=1e-9,
            close_figures=False,
        )

        ax2.set_title(timestamped_title(f"Qubit Spectroscopy During Readout {q.uid}"))

        if opts.save_figures:
            save_artifact(f"Resonator_Photons_Time_Resolved_{q.uid}", fig_1d)
            save_artifact(f"Resonator_Photons_Time_Resolved_{q.uid}_2D", fig_2d)

        figures[q.uid]["2D"] = fig_2d

        if opts.close_figures:
            plt.close(fig_1d)
            plt.close(fig_2d)

    return figures
