# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the analysis for an AC Stark shift vs power experiment.

The experiment is defined in laboneq_applications.contrib.experiments.

In this analysis, we process the magnitude of a spectroscopy signal to perform qubit
spectroscopy at various readout CW signal amplitudes. Each qubit spectroscopy dataset
is individually fitted to determine the qubit frequency. The resulting qubit frequencies
are then plotted against the square of the corresponding readout CW signal amplitudes.
This data is subsequently fitted with a linear model. The y-intercept of this linear fit
provides a more precise estimation of the qubit frequency compared to using a single
qubit spectroscopy measurement.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from laboneq import workflow
from laboneq.simple import dsl

from laboneq_applications.analysis import qubit_spectroscopy
from laboneq_applications.analysis.fitting_helpers import linear_fit
from laboneq_applications.analysis.options import BasePlottingOptions, DoFittingOption
from laboneq_applications.analysis.plotting_helpers import (
    plot_data_2d,
    timestamped_title,
)
from laboneq_applications.analysis.qubit_spectroscopy import (
    FitDataQubitSpecOptions,
    QubitSpectroscopyAnalysisWorkflowOptions,
)
from laboneq_applications.conversion import (
    as_quantum_element_list,
    sweeps_per_quantum_element,
)
from laboneq_applications.core import validation

if TYPE_CHECKING:
    import lmfit
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.task_options
class PlotAcStarkOption(DoFittingOption, BasePlottingOptions):
    """Options for the `plot_ac_stark_shift_vs_power` task of the qubit spec. analysis.

    Attributes:
        do_fitting:
            Whether to perform the fit.
            Default: `True`.
        save_figures:
            Whether to save the figures.
            Default: `True`.
        close_figures:
            Whether to close the figures.
            Default: `True`.
    """


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    frequencies: QubitSweepPoints,
    amplitudes: QubitSweepPoints,
    options: QubitSpectroscopyAnalysisWorkflowOptions | None = None,
) -> None:
    """The AC Stark shift vs power analysis Workflow.

    The workflow consists of the following steps:

    - [calculate_signal_magnitudes_and_phases][]
    - [fit_data][]
    - [extract_qubit_parameters][]
    - [plot_ac_stark_shift_vs_power][]

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        frequencies:
            The array of frequencies that were swept over in the experiment. If
            `qubits` is a single qubit, `frequencies` must be a list of numbers or
            an array. Otherwise, it must be a list of lists of numbers or arrays.
        amplitudes:
            The array of amplitudes that were swept over in the experiment. If
            `qubits` is a single qubit, `amplitudes` must be a list of numbers or
            an array. Otherwise, it must be a list of lists of numbers or arrays.
        options:
            The options for building the workflow, passed as an instance of
                [QubitSpectroscopyAnalysisWorkflowOptions][]. See the docstring of
                [QubitSpectroscopyAnalysisWorkflowOptions][] for more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        result = analysis_workflow(
            result=result,
            qubits=q0,
            frequencies=np.linspace(4.0e9, 4.6e9, 101),
            amplitudes=np.linspace(0.2, 0.6, 21),
            options=analysis_workflow.options(),
        ).run()
        ```
    """
    processed_data_dict = calculate_signal_magnitudes_and_phases(
        qubits, result, frequencies, amplitudes
    )
    fit_result = fit_data(qubits, processed_data_dict)
    qubit_parameter_and_fit_results = extract_qubit_parameters(qubits, fit_result)
    with workflow.if_(options.do_plotting):
        plot_ac_stark_shift_vs_power(qubits, processed_data_dict, fit_result)
    workflow.return_(qubit_parameter_and_fit_results)


@workflow.task
def calculate_signal_magnitudes_and_phases(
    qubits: QuantumElements,
    result: RunExperimentResults,
    frequencies: QubitSweepPoints,
    amplitudes: QubitSweepPoints,
) -> dict[str, dict[str, ArrayLike]]:
    """Calculates the magnitude and phase of the spectroscopy signal in result.

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the task. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        frequencies:
            The array of frequencies that were swept over in the experiment. If
            `qubits` is a single qubit, `frequencies` must be a list of numbers or
            an array. Otherwise, it must be a list of lists of numbers or arrays.
        amplitudes:
            The array of amplitudes that were swept over in the experiment. If
            `qubits` is a single qubit, `amplitudes` must be a list of numbers or
            an array. Otherwise, it must be a list of lists of numbers or arrays.

    Returns:
        dictionary with the qubit uids as keys and a processed data dict as
            values, containing the following data:

            - `sweep_points_freq`
            - `sweep_points_amp`
            - `data_raw`
            - `magnitudes`
            - `phases`
    """
    validation.validate_result(result)

    proc_data_dict = {}
    for q, freqs, amps in sweeps_per_quantum_element(qubits, frequencies, amplitudes):
        raw_data = result[dsl.handles.result_handle(q.uid)].data
        proc_data_dict[q.uid] = {
            "sweep_points_freq": freqs,
            "sweep_points_amp": amps,
            "data_raw": raw_data,
            "magnitudes": np.abs(raw_data),
            "phases": np.angle(raw_data),
        }

    return proc_data_dict


@workflow.task
def fit_data(
    qubits: QuantumElements,
    processed_data_dict: dict,
    options: FitDataQubitSpecOptions | None = None,
) -> dict[str, dict[str, lmfit.model.ModelResult | list[lmfit.model.ModelResult]]]:
    """Perform a qubit spectroscopy fit for every amplitude and a linear fit.

    The qubit frequencies obtained from the qubit spectroscopy fits are fitted with a
    linear model against the square of the amplitudes. Amplitudes for which the qubit
    spectroscopy fit fails are excluded from the linear fit.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit
            or a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
        processed_data_dict: the processed data dictionary returned by
            calculate_signal_magnitudes_and_phases
        options:
            The options class for this task as an instance of [FitDataQubitSpecOptions][].
            See the docstring of this class for accepted options.

    Returns:
        A dict with qubit UIDs as keys and the fit results for each qubit as values.
            The fit results for each qubit are a dict with the following keys:

            - `qubit_frequency_at_amp`: list with the qubit spectroscopy fit result
              for every amplitude, or None if the fit failed for that amplitude.
            - `linear_fit`: the result of the linear fit. Only present if the fit
              succeeded.

            If the do_fitting option is False, an empty dictionary is returned.
    """
    opts = FitDataQubitSpecOptions() if options is None else options
    qubits = as_quantum_element_list(qubits)

    fit_results = {}
    if not opts.do_fitting:
        return fit_results

    for q in qubits:
        fit_results[q.uid] = {}
        fit_at_amp = []
        for i in range(len(processed_data_dict[q.uid]["sweep_points_amp"])):
            magnitudes = processed_data_dict[q.uid]["magnitudes"][i]
            sweep_points = processed_data_dict[q.uid]["sweep_points_freq"]

            proc_data_dict = {
                q.uid: {"sweep_points": sweep_points, "magnitude": magnitudes}
            }

            fit_result = qubit_spectroscopy.fit_data(
                [q], proc_data_dict, options=opts
            ).get(q.uid)
            fit_at_amp.append(fit_result)

        fit_results[q.uid]["qubit_frequency_at_amp"] = fit_at_amp

        amps, qubit_frequency_at_amp = _successful_qubit_frequencies(
            processed_data_dict[q.uid]["sweep_points_amp"], fit_at_amp
        )
        if len(amps) < 2:  # noqa: PLR2004
            workflow.log(
                logging.ERROR,
                "Linear fit skipped for %s: fewer than two successful qubit "
                "spectroscopy fits.",
                q.uid,
            )
            continue

        try:
            fit_res = linear_fit(
                amps**2,
                qubit_frequency_at_amp * 1e-9,
            )
            fit_results[q.uid]["linear_fit"] = fit_res
        except ValueError as err:
            workflow.log(logging.ERROR, "Fit failed for %s: %s.", q.uid, err)

    return fit_results


def _successful_qubit_frequencies(
    amplitudes: ArrayLike,
    fit_at_amp: list[lmfit.model.ModelResult | None],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the amplitudes and fitted qubit frequencies of the successful fits.

    Arguments:
        amplitudes: the amplitudes that were swept over in the experiment.
        fit_at_amp: the qubit spectroscopy fit result for every amplitude, or None
            if the fit failed for that amplitude.

    Returns:
        the amplitudes and qubit frequencies for which the qubit spectroscopy fit
            succeeded.
    """
    successful = [i for i, fit_res in enumerate(fit_at_amp) if fit_res is not None]
    amps = np.asarray(amplitudes)[successful]
    qubit_frequencies = np.array([fit_at_amp[i].values["position"] for i in successful])
    return amps, qubit_frequencies


@workflow.task
def extract_qubit_parameters(
    qubits: QuantumElements,
    fit_results: dict[str, lmfit.model.ModelResult] | None,
    options: DoFittingOption | None = None,
) -> dict[
    str, dict[str, dict[str, unc.core.Variable | lmfit.model.ModelResult | None]]
]:
    """Extract the qubit parameters from the fit results.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
        fit_results: the fit-results dictionary returned by fit_data
        options:
            The options for this task as an instance of [DoFittingOption][].
            See the docstring of this class for more details.

    Returns:
        dict with extracted qubit parameters and the previous values for those qubit
            parameters. The dictionary has the following form:

            ```python
            {
                "new_parameter_values": {
                    q.uid: {
                        qb_param_name: qb_param_value
                    },
                },
                "old_parameter_values": {
                    q.uid: {
                        qb_param_name: qb_param_value
                    },
                },
                "fit_results": {
                    q.uid: {
                        linear_fit: fit_res,
                        qubit_frequency_at_amp: fit_at_amp
                    },
            }
            ```

            If the do_fitting option is False, the new_parameter_values are not
            extracted and the function only returns the old_parameter_values. If the
            linear fit is not found in fit_results for a qubit, the new_parameter_values
            entry for that qubit is left empty.
    """
    opts = DoFittingOption() if options is None else options
    qubits = as_quantum_element_list(qubits)
    qubit_parameters_and_results = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
        "fit_results": {q.uid: {} for q in qubits},
    }

    for q in qubits:
        # Store the qubit frequency value
        qubit_parameters_and_results["old_parameter_values"][q.uid] = {
            "resonance_frequency_ge": q.parameters.resonance_frequency_ge,
        }
        qubit_fit_results = fit_results.get(q.uid, {})
        qubit_parameters_and_results["fit_results"][q.uid] = qubit_fit_results

        if opts.do_fitting and "linear_fit" in qubit_fit_results:
            # Extract and store the qubit frequency value. The linear fit is
            # performed in GHz.
            fit_res = qubit_fit_results["linear_fit"]
            qb_freq = unc.ufloat(
                fit_res.params["intercept"].value * 1e9,
                fit_res.params["intercept"].stderr * 1e9,
            )

            qubit_parameters_and_results["new_parameter_values"][q.uid] = {
                "resonance_frequency_ge": qb_freq
            }

    return qubit_parameters_and_results


@workflow.task(save=False)
def plot_ac_stark_shift_vs_power(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, lmfit.model.ModelResult] | None,
    options: PlotAcStarkOption | None = None,
) -> dict[str, dict[str, mpl.figure.Figure]]:
    """Create the 2D qubit spectroscopy plot with frequency against amplitude squared.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit
            or a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict and qubit_parameters.
        processed_data_dict: the processed data dictionary returned by
            calculate_signal_magnitudes_and_phases.
        fit_results: the fit-results dictionary returned by fit_data
        options:
            The options for this task as an instance of
            [PlotAcStarkOption][]. See the docstring of this class for
            more details.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = PlotAcStarkOption() if options is None else options
    qubits = as_quantum_element_list(qubits)
    figures = {}
    for q in qubits:
        figures[q.uid] = {}
        qubit_fit_results = fit_results.get(q.uid, {})

        amps, qubit_frequency_at_amp = _successful_qubit_frequencies(
            processed_data_dict[q.uid]["sweep_points_amp"],
            qubit_fit_results.get("qubit_frequency_at_amp", []),
        )

        fig_1d, ax1 = plt.subplots()
        ax1.set_title(
            timestamped_title(f"AC Stark Shift vs. Offresonant Drive Amplitude {q.uid}")
        )
        ax1.plot(amps**2, qubit_frequency_at_amp * 1e-9, "o", zorder=2, label="Data")
        ax1.set_xlabel("Offresonant Drive Amplitude Squared")
        ax1.set_ylabel("Qubit Frequency (GHz)")

        if opts.do_fitting and "linear_fit" in qubit_fit_results:
            fit_res = qubit_fit_results["linear_fit"]
            swpts_fit = fit_res.userkws["x"]

            # Plot fit of the magnitude
            swpts_fine = np.linspace(swpts_fit[0], swpts_fit[-1], 501)
            ax1.plot(
                swpts_fine,
                fit_res.model.func(swpts_fine, **fit_res.best_values),
                "r-",
                zorder=1,
                label="Fit",
            )
            textstr = (
                f"Qubit ge frequency: "
                f"{np.round(fit_res.params['intercept'].value, 4)} GHz $\\pm$ "
                f"{np.round(fit_res.params['intercept'].stderr * 1e3, 2)} MHz"
            )
            ax1.text(0, -0.15, textstr, ha="left", va="top", transform=ax1.transAxes)

        ax1.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), handlelength=1.5, frameon=False
        )
        figures[q.uid]["1D"] = fig_1d

        amplitude = processed_data_dict[q.uid]["sweep_points_amp"]
        frequency = processed_data_dict[q.uid]["sweep_points_freq"]
        # plot magnitude
        fig_2d, ax2 = plot_data_2d(
            x_values=amplitude,
            y_values=frequency,
            z_values=processed_data_dict[q.uid]["magnitudes"].T,
            label_x_values="Readout Amplitude",
            label_y_values="Frequency (GHz)",
            label_z_values="Spectroscopy Signal\nMagnitude (a.u.)",
            scaling_x_values=1,
            scaling_y_values=1e-9,
            close_figures=False,
        )

        ax2.set_title(
            timestamped_title(f"AC Stark Shift vs. Offresonant Drive Amplitude {q.uid}")
        )

        if opts.save_figures:
            workflow.save_artifact(f"AC_Stark_vs_Power_{q.uid}", fig_1d)
            workflow.save_artifact(f"AC_Stark_vs_Power_{q.uid}_2D", fig_2d)

        if opts.close_figures:
            plt.close(fig_1d)
            plt.close(fig_2d)

        figures[q.uid]["2D"] = fig_2d

    return figures
