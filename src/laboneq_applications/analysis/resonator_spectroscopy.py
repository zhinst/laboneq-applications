# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the analysis for a resonator-spectroscopy experiment.

The experiment is defined in laboneq_applications.experiments.

In this analysis, we first interpret the raw data into the signal magnitude and phase.
Then we either extract the frequency corresponding to the min or max of the magnitude
data, or we fit a Lorentzian model to the signal magnitude and extract frequency
corresponding to the peak of the Lorentzian from the fit. Finally, we plot the data and
the fit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from laboneq import workflow
from laboneq.simple import dsl

from laboneq_applications.analysis.fitting_helpers import lorentzian_fit
from laboneq_applications.analysis.options import BasePlottingOptions
from laboneq_applications.analysis.plotting_helpers import (
    plot_raw_complex_data_1d,
    timestamped_title,
)
from laboneq_applications.conversion import (
    as_quantum_element_list,
    sweeps_per_quantum_element,
)
from laboneq_applications.core import validation
from laboneq_applications.tasks.parameter_updating import (
    _handle_deprecated_qubit_argument,
)

if TYPE_CHECKING:
    import lmfit
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.workflow_options
class ResonatorSpectroscopyAnalysisWorkflowOptions:
    """Option class for spectroscopy analysis workflows.

    Attributes:
        do_plotting:
            Whether to create plots.
            Default: 'True'.
        do_raw_data_plotting:
            Whether to plot the raw data.
            Default: True.
        do_plotting_magnitude_phase:
            Whether to plot the magnitude and phase.
            Default: True.
        do_plotting_real_imaginary:
            Whether to plot the real and imaginary data.
            Default: True.
    """

    do_plotting: bool = workflow.option_field(
        True, description="Whether to create plots."
    )
    do_raw_data_plotting: bool = workflow.option_field(
        True, description="Whether to plot the raw data."
    )
    do_plotting_magnitude_phase: bool = workflow.option_field(
        True, description="Whether to plot the magnitude and phase."
    )
    do_plotting_real_imaginary: bool = workflow.option_field(
        True, description="Whether to plot the real and imaginary data."
    )


@workflow.task_options
class FitDataResSpecOptions:
    """Options for the `fit_data` task of the resonator spectroscopy analysis.

    Attributes:
        fit_lorentzian:
            Whether to fit a Lorentzian model to the data.
            Default: `False`.
        fit_parameters_hints:
            Parameters hints accepted by lmfit
            Default: None.
    """

    fit_lorentzian: bool = workflow.option_field(
        False, description="Whether to fit a Lorentzian model to the data."
    )
    fit_parameters_hints: dict[str, dict[str, float | bool | str]] | None = (
        workflow.option_field(None, description="Parameters hints accepted by lmfit")
    )


@workflow.task_options
class ExtractQubitParametersResSpecOptions:
    """Options for the `extract_qubit_parameters` task of the resonator spec. analysis.

    Attributes:
        find_peaks:
            Whether to search for peaks (True) or dips (False) in the spectrum.
            Default: `False`.
    """

    find_peaks: bool = workflow.option_field(
        False,
        description="Whether to search for peaks (True) or dips (False) "
        "in the spectrum.",
    )


@workflow.task_options(base_class=BasePlottingOptions)
class PlotMagnitudePhaseOptions:
    """Options for the `plot_magnitude_phase` task of the resonator spec. analysis.

    Attributes:
        fit_lorentzian:
            Whether to fit a Lorentzian model to the data.
            Default: `False`.
        find_peaks:
            Whether to search for peaks (True) or dips (False) in the spectrum.
            Default: `False`.
        save_figures:
            Whether to save the figures.
            Default: `True`.
        close_figures:
            Whether to close the figures.
            Default: `True`.

    """

    fit_lorentzian: bool = workflow.option_field(
        False, description="Whether to fit a Lorentzian model to the data."
    )
    find_peaks: bool = workflow.option_field(
        False,
        description="Whether to search for peaks (True) or dips (False) "
        "in the spectrum.",
    )


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements | None = None,
    frequencies: QubitSweepPoints | None = None,
    options: ResonatorSpectroscopyAnalysisWorkflowOptions | None = None,
    qubit: QuantumElements | None = None,
) -> None:
    """The Resonator Spectroscopy analysis Workflow.

    The workflow consists of the following steps:

    - [calculate_signal_magnitude_and_phase][]
    - [fit_data][]
    - [extract_qubit_parameters][]
    - [plot_raw_complex_data_1d][]
    - [plot_magnitude_phase][]
    - [plot_real_imaginary][]

    !!! version-changed "Deprecated in version 26.10.0"
        The `qubit` argument is deprecated. Please use `qubits` instead.

    !!! version-changed "Changed in version 26.10.0"
        The `qubit` argument was renamed to `qubits` and now also accepts
        multiple qubits, passed as a list.

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single
            qubit or a list of qubits. The UIDs of these qubits must exist in
            the result.
        frequencies:
            The array of frequencies that were swept over in the experiment for
            each qubit. If `qubits` is a single qubit, `frequencies` must be a
            list of numbers or an array. Otherwise, it must be a list of lists
            of numbers or arrays.
        qubit:
            Deprecated. The single qubit on which to run the analysis. Use
            `qubits` instead.
        options:
            The options for building the workflow, passed as an instance of
                [ResonatorSpectroscopyAnalysisWorkflowOptions][]. See the docstring of
                [ResonatorSpectroscopyAnalysisWorkflowOptions][] for more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        analysis_result = analysis_workflow(
            result=result,
            qubits=[q0, q1],
            frequencies=[
                np.linspace(7.0, 7.1, 101),
                np.linspace(7.2, 7.3, 101),
            ],
        ).run()
        ```
    """
    qubits = _handle_deprecated_qubit_argument(qubits, qubit)
    processed_data_dict = calculate_signal_magnitude_and_phase(
        qubits, result, frequencies
    )
    fit_results = fit_data(qubits, processed_data_dict)
    qubit_parameters = extract_qubit_parameters(
        qubits, processed_data_dict, fit_results
    )
    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_raw_data_plotting):
            plot_raw_complex_data_1d(
                qubits,
                result,
                frequencies,
                xlabel="Readout Frequency, $f_{\\mathrm{RO}}$ (GHz)",
                xscaling=1e-9,
            )
        with workflow.if_(options.do_plotting_magnitude_phase):
            plot_magnitude_phase(
                qubits, processed_data_dict, fit_results, qubit_parameters
            )
        with workflow.if_(options.do_plotting_real_imaginary):
            plot_real_imaginary(qubits, result)
    workflow.return_(qubit_parameters)


@workflow.task
def calculate_signal_magnitude_and_phase(
    qubits: QuantumElements,
    result: RunExperimentResults,
    frequencies: QubitSweepPoints,
) -> dict[str, dict[str, ArrayLike]]:
    """Calculates the magnitude and phase of the spectroscopy signal in result.

    !!! version-changed "Changed in version 26.10.0"
        The `qubit` argument was renamed to `qubits` and now also accepts
        multiple qubits. The returned dictionary has the qubit UIDs as keys.

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single
            qubit or a list of qubits. The UIDs of these qubits must exist in
            the result.
        frequencies:
            The array of frequencies that were swept over in the experiment for
            each qubit.

    Returns:
        A dictionary with the qubit UIDs as keys and a processed data dict as values,
            containing the following keys:

            - `sweep_points`
            - `data_raw`
            - `magnitude`
            - `phase`
    """
    validation.validate_result(result)
    proc_data_dict = {}
    for q, q_frequencies in sweeps_per_quantum_element(qubits, frequencies):
        raw_data = result[dsl.handles.result_handle(q.uid)].data
        proc_data_dict[q.uid] = {
            "sweep_points": q_frequencies,
            "data_raw": raw_data,
            "magnitude": np.abs(raw_data),
            "phase": np.angle(raw_data),
        }
    return proc_data_dict


@workflow.task
def fit_data(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: FitDataResSpecOptions | None = None,
) -> dict[str, lmfit.model.ModelResult]:
    """Perform a fit of a Lorentzian model to the data if fit_lorentzian == True.

    !!! version-changed "Changed in version 26.10.0"
        The `qubits` argument was added and the task now fits the data of
        multiple qubits. The returned dictionary has the qubit UIDs as keys.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single
            qubit or a list of qubits. The UIDs of these qubits must exist in
            the processed_data_dict.
        processed_data_dict: the processed data dictionary returned by
            calculate_signal_magnitude_and_phase
        options:
            The options for this task as an instance of [FitDataResSpecOptions][].
            See the docstring of this class for more details.

    Returns:
        dict with qubit UIDs as keys and the fit results for each qubit as values.
    """
    opts = FitDataResSpecOptions() if options is None else options
    qubits = as_quantum_element_list(qubits)
    fit_results = {}
    if not opts.fit_lorentzian:
        return fit_results

    for q in qubits:
        swpts_fit = processed_data_dict[q.uid]["sweep_points"]
        data_to_fit = processed_data_dict[q.uid]["magnitude"]
        try:
            fit_res = lorentzian_fit(
                swpts_fit,
                data_to_fit,
                param_hints=opts.fit_parameters_hints,
            )
            fit_results[q.uid] = fit_res
        except ValueError as err:
            workflow.log(logging.ERROR, "Fit failed for %s: %s", q.uid, err)

    return fit_results


@workflow.task
def extract_qubit_parameters(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, lmfit.model.ModelResult],
    options: ExtractQubitParametersResSpecOptions | None = None,
) -> dict[str, dict[str, dict[str, int | float | unc.core.Variable | None]]]:
    """Extract the qubit parameters from the fit results.

    !!! version-changed "Changed in version 26.10.0"
        The `qubit` argument was renamed to `qubits` and now also accepts
        multiple qubits.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single
            qubit or a list of qubits. The UIDs of these qubits must exist in
            the processed_data_dict.
        processed_data_dict: the processed data dictionary returned by
            calculate_signal_magnitude_and_phase.
        fit_results: the fit-results dictionary returned by fit_data
        options:
            The options for this task as an instance of
            [ExtractQubitParametersResSpecOptions][].
            See the docstring of this class for more details.

    Returns:
        dict with extracted qubit parameters and the previous values for those qubit
            parameters. The dictionary has the following form:

            ```python
            {
                "new_parameter_values": {
                    q.uid: {qb_param_name: qb_param_value},
                },
                "old_parameter_values": {
                    q.uid: {qb_param_name: qb_param_value},
                },
            }
            ```
    """
    opts = ExtractQubitParametersResSpecOptions() if options is None else options
    qubits = as_quantum_element_list(qubits)

    qubit_parameters = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
    }

    for q in qubits:
        # Store the readout resonator frequency value
        qubit_parameters["old_parameter_values"][q.uid] = {
            "readout_resonator_frequency": q.parameters.readout_resonator_frequency,
        }

        # Extract and store the readout resonator frequency value
        if q.uid in fit_results:
            fit_res = fit_results[q.uid]
            rr_freq = unc.ufloat(
                fit_res.params["position"].value,
                fit_res.params["position"].stderr,
            )
        else:
            # find frequency at min or max of the signal magnitude
            take_extremum = np.argmax if opts.find_peaks else np.argmin
            freqs = processed_data_dict[q.uid]["sweep_points"]
            signal_magnitude = processed_data_dict[q.uid]["magnitude"]
            rr_freq = unc.core.Variable(
                freqs[take_extremum(signal_magnitude)],
                0,
            )

        qubit_parameters["new_parameter_values"][q.uid] = {
            "readout_resonator_frequency": rr_freq
        }

    return qubit_parameters


@workflow.task
def plot_magnitude_phase(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, lmfit.model.ModelResult],
    qubit_parameters: dict[
        str, dict[str, dict[str, int | float | unc.core.Variable | None]]
    ],
    options: PlotMagnitudePhaseOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot the magnitude and phase of the spectroscopy signal.

    !!! version-changed "Changed in version 26.10.0"
        The `qubit` argument was renamed to `qubits` and now also accepts
        multiple qubits. The task creates one figure per qubit and returns them
        in a dictionary with the qubit UIDs as keys.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single
            qubit or a list of qubits. The UIDs of these qubits must exist in
            the processed_data_dict and qubit_parameters.
        processed_data_dict: the processed data dictionary returned by
            calculate_signal_magnitude_and_phase.
        fit_results: the fit-results dictionary returned by fit_data
        qubit_parameters: the qubit-parameters dictionary returned by
            extract_qubit_parameters
        options:
            The options for this task as an instance of [PlotMagnitudePhaseOptions][].
            See the docstring of this class for more details.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.

            If there are no new_parameter_values for a qubit, then the fit result and
            the textbox with the extracted readout resonator frequency are not plotted.
    """
    opts = PlotMagnitudePhaseOptions() if options is None else options
    qubits = as_quantum_element_list(qubits)
    figures = {}

    for q in qubits:
        sweep_points = processed_data_dict[q.uid]["sweep_points"]
        magnitude = processed_data_dict[q.uid]["magnitude"]
        phase = processed_data_dict[q.uid]["phase"]
        fit_result = fit_results.get(q.uid)

        fig, axs = plt.subplots(nrows=2, sharex=True)
        axs[0].set_title(timestamped_title(f"Magnitude-Phase {q.uid}"))
        axs[0].plot(
            sweep_points / 1e9,
            magnitude,
            "-",
            zorder=2,
            label="data",
        )
        axs[0].set_ylabel("Transmission Signal\nMagnitude, $|S_{21}|$ (a.u.)")
        axs[1].plot(sweep_points / 1e9, phase, "-", zorder=2, label="data")
        axs[1].set_ylabel("Transmission Signal\nPhase of $S_{21}$ (a.u.)")
        axs[1].set_xlabel("Readout Frequency, $f_{\\mathrm{RO}}$ (GHz)")
        fig.align_ylabels()
        fig.subplots_adjust(hspace=0.1)

        if opts.fit_lorentzian and fit_result is not None:
            # Plot fit of the magnitude
            swpts_fine = np.linspace(sweep_points[0], sweep_points[-1], 501)
            axs[0].plot(
                swpts_fine / 1e9,
                fit_result.model.func(swpts_fine, **fit_result.best_values),
                "r-",
                zorder=1,
                label="fit",
            )

        if len(qubit_parameters["new_parameter_values"][q.uid]) > 0:
            rr_freq = qubit_parameters["new_parameter_values"][q.uid][
                "readout_resonator_frequency"
            ]

            # Point at the extracted readout resonator frequency
            if opts.fit_lorentzian and fit_result is not None:
                # Point on the magnitude plot from the fit result
                axs[0].plot(
                    rr_freq.nominal_value / 1e9,
                    fit_result.model.func(
                        rr_freq.nominal_value,
                        **fit_result.best_values,
                    ),
                    "or",
                    zorder=3,
                    markersize=plt.rcParams["lines.markersize"] + 1,
                )
                # Legend
                axs[0].legend(
                    loc="center left",
                    bbox_to_anchor=(1, 0),
                    handlelength=1.5,
                    frameon=False,
                )
            else:
                take_extremum = np.argmax if opts.find_peaks else np.argmin
                # Point on the magnitude plot at the rr freq
                axs[0].plot(
                    rr_freq.nominal_value / 1e9,
                    magnitude[take_extremum(magnitude)],
                    "or",
                    zorder=3,
                    markersize=plt.rcParams["lines.markersize"] + 1,
                )

            # Line on the phase plot corresponding to the rr freq
            ylims = axs[1].get_ylim()
            axs[1].vlines(
                rr_freq.nominal_value / 1e9,
                *ylims,
                linestyles="--",
                colors="r",
                zorder=0,
            )
            axs[1].set_ylim(ylims)

            # Textbox
            old_rr_freq = qubit_parameters["old_parameter_values"][q.uid][
                "readout_resonator_frequency"
            ]
            textstr = (
                f"Readout-resonator frequency: "
                f"{rr_freq.nominal_value / 1e9:.4f} GHz $\\pm$ "
                f"{rr_freq.std_dev / 1e6:.4f} MHz"
            )
            textstr += f"\nPrevious value: {old_rr_freq / 1e9:.4f} GHz"
            axs[1].text(
                0, -0.35, textstr, ha="left", va="top", transform=axs[1].transAxes
            )

        if opts.save_figures:
            workflow.save_artifact(f"Magnitude_Phase_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures


@workflow.task
def plot_real_imaginary(
    qubits: QuantumElements,
    result: RunExperimentResults,
    options: BasePlottingOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot the real and imaginary parts of the spectroscopy signal.

    !!! version-changed "Changed in version 26.10.0"
        The `qubit` argument was renamed to `qubits` and now also accepts
        multiple qubits. The task creates one figure per qubit and returns them
        in a dictionary with the qubit UIDs as keys.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single
            qubit or a list of qubits. The UIDs of these qubits must exist in
            the result.
        result:
            The experiment results returned by the run_experiment task.
        options:
            The options for this task as an instance of [BasePlottingOptions][].
            See the docstring for this class for more information.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = BasePlottingOptions() if options is None else options
    validation.validate_result(result)
    qubits = as_quantum_element_list(qubits)
    figures = {}

    for q in qubits:
        raw_data = result[dsl.handles.result_handle(q.uid)].data

        fig, ax = plt.subplots()
        ax.set_title(timestamped_title(f"Real-Imaginary {q.uid}"))
        ax.set_xlabel("Real Transmission Signal, Re($S_{21}$) (a.u.)")
        ax.set_ylabel("Imaginary Transmission Signal, Im($S_{21}$) (a.u.)")
        ax.plot(
            np.real(raw_data),
            np.imag(raw_data),
            "o",
            zorder=2,
            label="data",
        )

        if opts.save_figures:
            workflow.save_artifact(f"Real_Imaginary_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures
