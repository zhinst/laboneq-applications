"""This module defines the analysis for an drage beta parameter experiment.

The experiment is defined in laboneq_applications.experiments.

In this analysis, we first interpret the raw data into qubit populations using
principle-component analysis or rotation and projection on the measured calibration
states. Then we fit the measured qubit population as a function of the beta parameter
and determine the optimal beta parameter. Finally, we plot the data and the fit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc

from laboneq_applications.analysis.cal_trace_rotation import calculate_population_1d
from laboneq_applications.analysis.fitting_helpers import (
    fit_data_lmfit,
    linear,
)
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_result,
)
from laboneq_applications.experiments.options import (
    TuneupAnalysisOptions,
    TuneUpAnalysisWorkflowOptions,
)
from laboneq_applications.workflow import (
    comment,
    if_,
    save_artifact,
    task,
    workflow,
)

if TYPE_CHECKING:
    import lmfit
    import matplotlib as mpl
    from numpy.typing import ArrayLike

    from laboneq_applications.tasks.run_experiment import RunExperimentResults
    from laboneq_applications.typing import Qubits, QubitSweepPoints


@workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: Qubits,
    beta_values: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The Drag Beta Parameter analysis Workflow.

    The workflow consists of the following steps:

    - [calculate_qubit_population]()
    - [fit_data]()
    - [extract_qubit_parameters]()
    - [plot_population]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        beta_values:
            The beta_values that were swept over in the amplitude-Rabi experiment for
            each qubit. If `qubits` is a single qubit, `beta_values` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        options:
            The options for building the workflow, passed as an instance of
                [TuneUpAnalysisWorkflowOptions].
            In addition to options from [WorkflowOptions], the following
            custom options are supported: do_fitting, do_plotting, and the options of
            the [TuneupAnalysisOptions] class. See the docstring of
            [TuneUpAnalysisWorkflowOptions] for more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        options = TuneUpAnalysisWorkflowOptions()
        result = analysis_workflow(
            results=results
            qubits=[q0, q1],
            beta_values=[
                np.linspace(-0.05, 0.05, 11),
                np.linspace(-0.05, 0.05, 11),
            ],
            options=options,
        ).run()
        ```
    """
    processed_data_dict = calculate_qubit_population_from_ids(
        qubits, result, beta_values, ["y180", "my180"]
    )
    fit_results = fit_data(qubits, processed_data_dict)
    qubit_parameters = extract_qubit_parameters(
        qubits, processed_data_dict, fit_results
    )
    with if_(options.do_plotting):
        with if_(options.do_qubit_population_plotting):
            plot_population(
                qubits,
                processed_data_dict,
                fit_results,
                qubit_parameters,
                ["y180", "my180"],
            )


@task
def calculate_qubit_population_from_ids(
    qubits: Qubits,
    result: RunExperimentResults,
    beta_values: QubitSweepPoints,
    pulse_ids: list[str],
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, dict[str, ArrayLike]]:
    """Processes the raw data from the experiment result.

     The data is processed in the following way:

     - If calibration traces were used in the experiment, the raw data is rotated based
     on the calibration traces.
     See [cal_trace_rotation.py/rotate_data_to_cal_trace_results] for more details.
     - If no calibration traces were used in the experiment, or do_pca = True is passed
     in options, principal-component analysis is performed on the data.
     See [cal_trace_rotation.py/principal_component_analysis] for more details.

    Arguments:
        qubits:
            The qubits on which the experiments was run. May be either
            a single qubit or a list of qubits.
        result: the result of the experiment, returned by the run_experiment task.
        beta_values:
            The beta_values that were swept over in the experiment for
            each qubit. If `qubits` is a single qubit, `beta_values` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        pulse_ids:
            IDs to identify different measurement results.
        options:
            The options for processing the raw data.
            See [TuneupAnalysisOptions], [TuneupExperimentOptions] and
            [BaseExperimentOptions] for accepted options.
            Overwrites the options from [TuneupAnalysisOptions],
            [TuneupExperimentOptions] and [BaseExperimentOptions].

    Returns:
        dict with qubit UIDs as keys. The dictionary of processed data for each qubit
        further has "y180" and "my180" as keys.
        See [cal_trace_rotation.py/calculate_population_1d] for what this
        dictionary looks like.

    Raises:
        TypeError if result is not an instance of RunExperimentResults.
    """
    validate_result(result)
    opts = TuneupAnalysisOptions() if options is None else options
    qubits, beta_values = validate_and_convert_qubits_sweeps(qubits, beta_values)
    processed_data_dict = {}
    for q, amps in zip(qubits, beta_values):
        processed_data_dict[q.uid] = {}
        if opts.use_cal_traces:
            calibration_traces = [
                result.cal_trace[q.uid][cs].data for cs in opts.cal_states
            ]
            do_pca = opts.do_pca
        else:
            calibration_traces = []
            do_pca = True
        for pulse_id in pulse_ids:
            raw_data = result.result[f"{q.uid}_{pulse_id}"].data
            data_dict = calculate_population_1d(
                raw_data,
                amps,
                calibration_traces,
                do_pca=do_pca,
            )
            processed_data_dict[q.uid][pulse_id] = data_dict
    return processed_data_dict


@task
def fit_data(
    qubits: Qubits,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, lmfit.model.ModelResult]:
    """Perform a fit of a linear model to the data.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
        processed_data_dict: the processed data dictionary returned by process_raw_data
        options:
            The options for processing the raw data.
            See [TuneupAnalysisOptions], [TuneupExperimentOptions] and
            [BaseExperimentOptions] for accepted options.

    Returns:
        dict with qubit UIDs as keys, "y180"/"my180" as subkeys and the fit results
        for each qubit as values.
    """
    opts = TuneupAnalysisOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    fit_results = {}
    if not opts.do_fitting:
        return fit_results

    for q in qubits:
        fit_results[q.uid] = {}
        for pulse_id in processed_data_dict[q.uid]:
            swpts_fit = processed_data_dict[q.uid][pulse_id]["sweep_points"]
            data_to_fit = processed_data_dict[q.uid][pulse_id][
                "population" if opts.do_rotation else "data_raw"
            ]
            if not opts.do_rotation:
                data_to_fit = np.array(data_to_fit, dtype=np.int32)
            if not opts.fit_parameters_hints:
                gradient = (data_to_fit[-1] - data_to_fit[0]) / (
                    swpts_fit[-1] - swpts_fit[0]
                )
                opts.fit_parameters_hints = {
                    "gradient": {"value": gradient},
                    "intercept": {"value": data_to_fit[-1] - gradient * swpts_fit[-1]},
                }
            try:
                fit_res = fit_data_lmfit(
                    model=linear,
                    x=swpts_fit,
                    y=data_to_fit,
                    param_hints=opts.fit_parameters_hints,
                )
                fit_results[q.uid][pulse_id] = fit_res
            except ValueError as err:
                comment(f"Fit failed for {q.uid}: {err}.")

    return fit_results


@task
def extract_qubit_parameters(
    qubits: Qubits,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, lmfit.model.ModelResult],
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, dict[str, dict[str, int | float | unc.core.Variable | None]]]:
    """Extract the qubit parameters from the fit results.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
        processed_data_dict: the processed data dictionary returned by process_raw_data
        fit_results: the fit-results dictionary returned by fit_data
        options:
            The options for extracting the qubit parameters.
            See [TuneupAnalysisOptions], [TuneupExperimentOptions] and
            [BaseExperimentOptions] for accepted options.

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
    opts = TuneupAnalysisOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    qubit_parameters = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
    }

    for q in qubits:
        # Store the old beta values
        old_beta = (
            q.parameters.ef_drive_pulse["beta"]
            if "f" in opts.transition
            else q.parameters.ge_drive_pulse["beta"]
        )
        qubit_parameters["old_parameter_values"][q.uid] = {
            f"{opts.transition}_beta": old_beta,
        }

        if opts.do_fitting and q.uid in fit_results:
            # Extract and store the new pi and pi-half pulse amplitude values
            gradient = {}
            intercept = {}
            for i, pulse_id in enumerate(processed_data_dict[q.uid]):
                gradient[i] = unc.ufloat(
                    fit_results[q.uid][pulse_id].params["gradient"].value,
                    fit_results[q.uid][pulse_id].params["gradient"].stderr,
                )
                intercept[i] = unc.ufloat(
                    fit_results[q.uid][pulse_id].params["intercept"].value,
                    fit_results[q.uid][pulse_id].params["intercept"].stderr,
                )
            intercept_diff_mean = intercept[0] - intercept[1]
            slope_diff_mean = gradient[1] - gradient[0]
            new_beta = intercept_diff_mean / slope_diff_mean
            qubit_parameters["new_parameter_values"][q.uid] = {
                f"{opts.transition}_beta": new_beta,
            }
    return qubit_parameters


@task
def plot_population(
    qubits: Qubits,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, lmfit.model.ModelResult] | None,
    qubit_parameters: dict[
        str,
        dict[str, dict[str, int | float | unc.core.Variable | None]],
    ]
    | None,
    pulse_ids: list[str],
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the Drag Beta Parameter plots.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in
            processed_data_dict and qubit_parameters.
        processed_data_dict: the processed data dictionary returned by process_raw_data
        fit_results: the fit-results dictionary returned by fit_data
        qubit_parameters: the qubit-parameters dictionary returned by
            extract_qubit_parameters
        pulse_ids:
            ids to identify different measurement results.
        options:
            The options for processing the raw data.
            See [TuneupAnalysisOptions], [TuneupExperimentOptions] and
            [BaseExperimentOptions] for accepted options.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.

        If a qubit uid is not found in fit_results, the fit and the textbox with the
        extracted qubit parameters are not plotted.
    """
    opts = TuneupAnalysisOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}
    for q in qubits:
        fig, ax = plt.subplots()
        ax.set_title(f"DRAG Beta {q.uid}")  # add timestamp here
        ax.set_xlabel("Beta")
        num_cal_traces = processed_data_dict[q.uid][pulse_ids[0]]["num_cal_traces"]
        ax.set_ylabel(
            "Principal Component (a.u)"
            if (num_cal_traces == 0 or opts.do_pca)
            else f"$|{opts.cal_states[-1]}\\rangle$-State Population",
        )
        for pulse_id in pulse_ids:
            sweep_points = processed_data_dict[q.uid][pulse_id]["sweep_points"]
            data = processed_data_dict[q.uid][pulse_id][
                "population" if opts.do_rotation else "data_raw"
            ]
            ax.plot(sweep_points, data, "o", zorder=2, label=f"data_{pulse_id}")

        if processed_data_dict[q.uid][pulse_ids[0]]["num_cal_traces"] > 0:
            # plot lines at the calibration traces
            xlims = ax.get_xlim()
            ax.hlines(
                processed_data_dict[q.uid][pulse_ids[0]]["population_cal_traces"],
                *xlims,
                linestyles="--",
                colors="gray",
                zorder=0,
                label="calib.\ntraces",
            )
            ax.set_xlim(xlims)

        if opts.do_fitting and q.uid in fit_results:
            for pulse_id in pulse_ids:
                fit_res_qb = fit_results[q.uid][pulse_id]
                # plot fit
                sweep_points = processed_data_dict[q.uid][pulse_id]["sweep_points"]
                swpts_fine = np.linspace(sweep_points[0], sweep_points[-1], 501)
                ax.plot(
                    swpts_fine,
                    fit_res_qb.model.func(swpts_fine, **fit_res_qb.best_values),
                    "r-",
                    zorder=1,
                    label=f"fit_{pulse_id}",
                )

            if len(qubit_parameters["new_parameter_values"][q.uid]) > 0:
                new_beta = qubit_parameters["new_parameter_values"][q.uid][
                    f"{opts.transition}_beta"
                ]
                # point at pi-pulse amplitude
                ax.plot(
                    new_beta.nominal_value,
                    fit_res_qb.model.func(
                        new_beta.nominal_value,
                        **fit_res_qb.best_values,
                    ),
                    "sk",
                    zorder=3,
                    markersize=plt.rcParams["lines.markersize"] + 1,
                )
                # textbox
                old_beta = qubit_parameters["old_parameter_values"][q.uid][
                    f"{opts.transition}_beta"
                ]
                textstr = (
                    "$\\beta$: "
                    f"{new_beta.nominal_value:.4f} $\\pm$ "
                    f"{new_beta.std_dev:.4f}"
                )
                textstr += "\nOld $\\beta$: " + f"{old_beta:.4f}"
                ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            handlelength=1.5,
            frameon=False,
        )

        if opts.save_figures:
            save_artifact(f"Rabi_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures
