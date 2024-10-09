"""This module defines the analysis for a qubit-spectroscopy experiment.

The experiment is defined in laboneq_applications.experiments.

In this analysis, we first interpret the raw data into the signal magnitude and phase.
Then we fit a Lorentzian model to the signal magnitude and extract the frequency
corresponding to the peak of the Lorentzian from the fit. This is the new qubit
frequency. Finally, we plot the data and the fit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc

from laboneq_applications import workflow
from laboneq_applications.analysis.fitting_helpers import lorentzian_fit
from laboneq_applications.analysis.plotting_helpers import plot_raw_complex_data_1d
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_result,
)
from laboneq_applications.experiments.options import (
    QubitSpectroscopyAnalysisOptions,
    QubitSpectroscopyAnalysisWorkflowOptions,
)

if TYPE_CHECKING:
    import lmfit
    import matplotlib as mpl
    from numpy.typing import ArrayLike

    from laboneq_applications.tasks.run_experiment import RunExperimentResults
    from laboneq_applications.typing import Qubits, QubitSweepPoints

options = QubitSpectroscopyAnalysisWorkflowOptions


@workflow.workflow(name="qubit_spectroscopy_analysis")
def analysis_workflow(
    result: RunExperimentResults,
    qubits: Qubits,
    frequencies: QubitSweepPoints,
    options: QubitSpectroscopyAnalysisWorkflowOptions | None = None,
) -> None:
    """The Qubit Spectroscopy analysis Workflow.

    The workflow consists of the following steps:

    - [calculate_signal_magnitude_and_phase]()
    - [fit_data]()
    - [extract_qubit_parameters]()
    - [plot_raw_complex_data_1d]()
    - [plot_qubit_spectroscopy]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        frequencies:
            The array of frequencies that were swept over in the experiment. If `qubits`
            is a single qubit, `frequencies` must be a list of numbers or an array.
            Otherwise, it must be a list of lists of numbers or arrays.
        options:
            The options for building the workflow, passed as an instance of
                [QubitSpectroscopyAnalysisWorkflowOptions].
            In addition to options from [WorkflowOptions], the following
            custom options are supported: do_plotting, do_raw_data_plotting,
            do_plotting_qubit_spectroscopy and the options of
            the [QubitSpectroscopyAnalysisOptions] class. See the docstring of
            [QubitSpectroscopyAnalysisOptions] for more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        result = analysis_workflow(
            results=results
            qubits=[q0, q1],
            frequencies=[
                np.linspace(6.0, 6.3, 301),
                np.linspace(5.8, 6.1, 301),
            ],
            options=analysis_workflow.options(),
        ).run()
        ```
    """
    processed_data_dict = calculate_signal_magnitude_and_phase(
        qubits, result, frequencies
    )
    fit_result = fit_data(qubits, processed_data_dict)
    qubit_parameters = extract_qubit_parameters(qubits, fit_result)
    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_raw_data_plotting):
            plot_raw_complex_data_1d(
                qubits,
                result,
                frequencies,
                xlabel="Qubit Frequency, $f_{\\mathrm{qb}}$ (GHz)",
                xscaling=1e-9,
            )
        with workflow.if_(options.do_plotting_qubit_spectroscopy):
            plot_qubit_spectroscopy(
                qubits, processed_data_dict, fit_result, qubit_parameters
            )
    workflow.return_(qubit_parameters)


@workflow.task
def calculate_signal_magnitude_and_phase(
    qubits: Qubits,
    result: RunExperimentResults,
    frequencies: ArrayLike,
) -> dict[str, dict[str, ArrayLike]]:
    """Calculates the magnitude and phase of the spectroscopy signal in result.

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the task. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        frequencies:
            The array of frequencies that were swept over in the experiment.

    Returns:
        dictionary with the qubit uids as keys and a processed data dict as values,
        containing the following data:
            sweep_points
            data_raw
            magnitude
            phase
    """
    qubits, frequencies = validate_and_convert_qubits_sweeps(qubits, frequencies)
    validate_result(result)
    proc_data_dict = {}
    for q, freqs in zip(qubits, frequencies):
        raw_data = result.result[q.uid].data
        proc_data_dict[q.uid] = {
            "sweep_points": freqs,
            "data_raw": raw_data,
            "magnitude": np.abs(raw_data),
            "phase": np.angle(raw_data),
        }

    return proc_data_dict


@workflow.task
def fit_data(
    qubits: Qubits,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: QubitSpectroscopyAnalysisOptions | None = None,
) -> dict[str, lmfit.model.ModelResult] | None:
    """Perform a fit of a Lorentzian model to the data.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
        processed_data_dict: the processed data dictionary returned by
            calculate_signal_magnitude_and_phase
        options:
            The options for processing the raw data.
            See [QubitSpectroscopyAnalysisOptions] for accepted options.

    Returns:
        dict with qubit UIDs as keys and the fit results for each qubit as keys.
    """
    opts = QubitSpectroscopyAnalysisOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    fit_results = {}
    if not opts.do_fitting:
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
            workflow.log(logging.ERROR, "Fit failed for %s: %s.", q.uid, err)

    return fit_results


@workflow.task
def extract_qubit_parameters(
    qubits: Qubits,
    fit_results: dict[str, lmfit.model.ModelResult] | None,
    options: QubitSpectroscopyAnalysisOptions | None = None,
) -> dict[str, dict[str, dict[str, int | float | unc.core.Variable | None]]]:
    """Extract the qubit parameters from the fit results.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
        fit_results: the fit-results dictionary returned by fit_data
        options:
            The options for extracting the qubit parameters.
            See [QubitSpectroscopyAnalysisOptions] for accepted options.

    Returns:
        dict with extracted qubit parameters and the previous values for those qubit
        parameters. The dictionary has the following form:
        ```python
        {
            "new_parameter_values": {
                q.uid: {
                    qb_param_name: qb_param_value
                },
            }
            "old_parameter_values": {
                q.uid: {
                    qb_param_name: qb_param_value
                },
            }
        }
        ```
        If the do_fitting option is False, the new_parameter_values are not extracted
        and the function only returns the old_parameter_values.
        If a qubit uid is not found in fit_results, the new_parameter_values entry for
        that qubit is left empty.
    """
    opts = QubitSpectroscopyAnalysisOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    qubit_parameters = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
    }

    for q in qubits:
        # Store the qubit frequency value
        qubit_parameters["old_parameter_values"][q.uid] = {
            "resonance_frequency_ge": q.parameters.resonance_frequency_ge,
        }

        if opts.do_fitting and q.uid in fit_results:
            # Extract and store the qubit frequency value
            fit_res = fit_results[q.uid]
            qb_freq = unc.ufloat(
                fit_res.params["position"].value,
                fit_res.params["position"].stderr,
            )

            qubit_parameters["new_parameter_values"][q.uid] = {
                "resonance_frequency_ge": qb_freq
            }

    return qubit_parameters


@workflow.task
def plot_qubit_spectroscopy(
    qubits: Qubits,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, lmfit.model.ModelResult] | None,
    qubit_parameters: dict[
        str, dict[str, dict[str, int | float | unc.core.Variable | None]]
    ],
    options: QubitSpectroscopyAnalysisOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the qubit-spectroscopy plots.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict and qubit_parameters.
        processed_data_dict: the processed data dictionary returned by
            calculate_signal_magnitude_and_phase.
        fit_results: the fit-results dictionary returned by fit_data
        qubit_parameters: the qubit-parameters dictionary returned by
            extract_qubit_parameters
        options:
            The options for extracting the qubit parameters.
            See [QubitSpectroscopyAnalysisOptions] for accepted options.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = QubitSpectroscopyAnalysisOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}

    for q in qubits:
        sweep_points = processed_data_dict[q.uid]["sweep_points"]
        magnitude = processed_data_dict[q.uid]["magnitude"]

        fig, ax = plt.subplots()
        ax.set_title(f"Qubit Spectroscopy {q.uid}")  # add timestamp here
        ax.plot(
            sweep_points / 1e9,
            magnitude,
            "o",
            zorder=2,
            label="data",
        )
        ax.set_ylabel("Transmission Signal\nMagnitude, $|S_{21}|$ (a.u.)")
        ax.set_xlabel("Qubit Frequency, $f_{\\mathrm{QB}}$ (GHz)")

        if opts.do_fitting and q.uid in fit_results:
            fit_res = fit_results[q.uid]

            # Plot fit of the magnitude
            swpts_fine = np.linspace(sweep_points[0], sweep_points[-1], 501)
            ax.plot(
                swpts_fine / 1e9,
                fit_res.model.func(swpts_fine, **fit_res.best_values),
                "r-",
                zorder=1,
                label="fit",
            )

            if len(qubit_parameters["new_parameter_values"][q.uid]) > 0:
                qb_freq = qubit_parameters["new_parameter_values"][q.uid][
                    "resonance_frequency_ge"
                ]

                # Point at the qubit frequency
                ax.plot(
                    qb_freq.nominal_value / 1e9,
                    fit_res.model.func(
                        qb_freq.nominal_value,
                        **fit_res.best_values,
                    ),
                    "or",
                    zorder=3,
                    markersize=plt.rcParams["lines.markersize"] + 1,
                )
                # Textbox
                old_qb_freq = qubit_parameters["old_parameter_values"][q.uid][
                    "resonance_frequency_ge"
                ]
                textstr = (
                    f"Qubit ge frequency: "
                    f"{qb_freq.nominal_value / 1e9:.4f} GHz $\\pm$ "
                    f"{qb_freq.std_dev / 1e6:.4f} MHz"
                )
                textstr += f"\nPrevious value: {old_qb_freq / 1e9:.4f} GHz"
                ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)
        # Legend
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            handlelength=1.5,
            frameon=False,
        )

        if opts.save_figures:
            workflow.save_artifact(f"Qubit_Spectroscopy_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures
