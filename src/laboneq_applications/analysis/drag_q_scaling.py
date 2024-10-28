"""This module defines the analysis for a DRAG calibration experiment.

The experiment is defined in laboneq_applications.experiments. See the docstring of
this module for more details about the experiment and its parameters.

In this analysis, we first interpret the raw data into qubit populations using
principle-component analysis or rotation and projection on the measured calibration
states. Then we fit the measured qubit population as a function of the beta parameter
and determine the optimal beta parameter. Finally, we plot the data and the fit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc

from laboneq_applications import dsl, workflow
from laboneq_applications.analysis.calibration_traces_rotation import (
    calculate_population_1d,
)
from laboneq_applications.analysis.fitting_helpers import (
    fit_data_lmfit,
    linear,
)
from laboneq_applications.analysis.plotting_helpers import (
    plot_raw_complex_data_1d,
    timestamped_title,
)
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_result,
)
from laboneq_applications.experiments.options import (
    TuneupAnalysisOptions,
    TuneUpAnalysisWorkflowOptions,
)

if TYPE_CHECKING:
    import lmfit
    import matplotlib as mpl
    from numpy.typing import ArrayLike

    from laboneq_applications.tasks.run_experiment import RunExperimentResults
    from laboneq_applications.typing import Qubits, QubitSweepPoints


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: Qubits,
    q_scalings: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The analysis Workflow for the DRAG quadrature-scaling calibration.

    The workflow consists of the following steps:

    - [calculate_qubit_population_for_pulse_ids]()
    - [fit_data]()
    - [extract_qubit_parameters]()
    - [plot_raw_complex_data_1d]()
    - [plot_population]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        q_scalings:
            The quadrature scaling factors that were swept over in the experiment for
            each qubit. If `qubits` is a single qubit, `q_scalings` must be a list of
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
        options = analysis_workflow.options()
        options.close_figures(False)
        result = analysis_workflow(
            results=results
            qubits=[q0, q1],
            q_scalings=[
                np.linspace(-0.05, 0.05, 11),
                np.linspace(-0.05, 0.05, 11),
            ],
            options=options,
        ).run()
        ```
    """
    processed_data_dict = calculate_qubit_population_for_pulse_ids(
        qubits, result, q_scalings
    )
    fit_results = fit_data(qubits, processed_data_dict)
    qubit_parameters = extract_qubit_parameters(qubits, fit_results)
    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_raw_data_plotting):
            plot_raw_complex_data_1d(
                qubits,
                result,
                q_scalings,
                xlabel="DRAG Quadrature Scaling Factor, $\\beta$",
            )
        with workflow.if_(options.do_qubit_population_plotting):
            plot_population(
                qubits,
                processed_data_dict,
                fit_results,
                qubit_parameters,
            )
    workflow.return_(qubit_parameters)


@workflow.task
def calculate_qubit_population_for_pulse_ids(
    qubits: Qubits,
    result: RunExperimentResults,
    q_scalings: QubitSweepPoints,
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, dict[str, dict[str, ArrayLike]]]:
    """Processes the raw data from the experiment result.

     The data is processed in the following way:

     - If calibration traces were used in the experiment, the raw data is rotated based
     on the calibration traces.
     See [calibration_traces_rotation.py/rotate_data_to_cal_trace_results] for more
     details.
     - If no calibration traces were used in the experiment, or do_pca = True is passed
     in options, principal-component analysis is performed on the data.
     See [calibration_traces_rotation.py/principal_component_analysis] for more details.

    Arguments:
        qubits:
            The qubits on which the experiments was run. May be either
            a single qubit or a list of qubits.
        result: the result of the experiment, returned by the run_experiment task.
        q_scalings:
            The quadrature scaling factors that were swept over in the experiment for
            each qubit. If `qubits` is a single qubit, `q_scalings` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        options:
            The options for processing the raw data.
            See [TuneupAnalysisOptions], [TuneupExperimentOptions] and
            [BaseExperimentOptions] for accepted options.
            Overwrites the options from [TuneupAnalysisOptions],
            [TuneupExperimentOptions] and [BaseExperimentOptions].

    Returns:
        dict with qubit UIDs as keys. The dictionary of processed data for each qubit
        further has "y180" and "my180" as keys.
        See [calibration_traces_rotation.py/calculate_population_1d] for what this
        dictionary looks like.

    Raises:
        TypeError:
            If result is not an instance of RunExperimentResults.
    """
    validate_result(result)
    opts = TuneupAnalysisOptions() if options is None else options
    qubits, q_scalings = validate_and_convert_qubits_sweeps(qubits, q_scalings)
    processed_data_dict = {}
    for q, qscales in zip(qubits, q_scalings):
        processed_data_dict[q.uid] = {}
        if opts.use_cal_traces:
            calibration_traces = [
                result[dsl.handles.calibration_trace_handle(q.uid, cs)].data
                for cs in opts.cal_states
            ]
            do_pca = opts.do_pca
        else:
            calibration_traces = []
            do_pca = True
        for pulse_id in result[dsl.handles.result_handle(q.uid)]:
            raw_data = result[dsl.handles.result_handle(q.uid, suffix=pulse_id)].data
            data_dict = calculate_population_1d(
                raw_data,
                qscales,
                calibration_traces,
                do_pca=do_pca,
            )
            processed_data_dict[q.uid][pulse_id] = data_dict
    return processed_data_dict


@workflow.task
def fit_data(
    qubits: Qubits,
    processed_data_dict: dict[str, dict[str, dict[str, ArrayLike]]],
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, dict[str, lmfit.model.ModelResult]]:
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
            data_to_fit = processed_data_dict[q.uid][pulse_id]["population"]
            if pulse_id == "xx":
                param_hints = {
                    "gradient": {"value": 0, "vary": False},
                    "intercept": {"value": np.mean(data_to_fit)},
                }
            else:
                gradient = (data_to_fit[-1] - data_to_fit[0]) / (
                    swpts_fit[-1] - swpts_fit[0]
                )
                param_hints = {
                    "gradient": {"value": gradient},
                    "intercept": {"value": data_to_fit[-1] - gradient * swpts_fit[-1]},
                }
            param_hints_user = opts.fit_parameters_hints
            if param_hints_user is None:
                param_hints_user = {}
            param_hints.update(param_hints_user)
            try:
                fit_res = fit_data_lmfit(
                    model=linear,
                    x=swpts_fit,
                    y=data_to_fit,
                    param_hints=param_hints,
                )
                fit_results[q.uid][pulse_id] = fit_res
            except ValueError as err:
                workflow.log(logging.ERROR, "Fit failed for %s: %s.", q.uid, err)

    return fit_results


@workflow.task
def extract_qubit_parameters(
    qubits: Qubits,
    fit_results: dict[str, dict[str, lmfit.model.ModelResult]],
    options: TuneupAnalysisOptions | None = None,
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
        # Store the old quadrature scaling factor values
        old_beta = (
            q.parameters.ef_drive_pulse["beta"]
            if "f" in opts.transition
            else q.parameters.ge_drive_pulse["beta"]
        )
        qubit_parameters["old_parameter_values"][q.uid] = {
            f"{opts.transition}_drive_pulse.beta": old_beta,
        }

        if opts.do_fitting and q.uid in fit_results:
            # Extract and store the new pi and pi-half pulse amplitude values
            gradient = {}
            intercept = {}
            for i, pulse_id in enumerate(["xy", "xmy"]):
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
            if slope_diff_mean.nominal_value != 0:
                new_beta = intercept_diff_mean / slope_diff_mean
                qubit_parameters["new_parameter_values"][q.uid] = {
                    f"{opts.transition}_drive_pulse.beta": new_beta
                }
            else:
                workflow.log(
                    logging.ERROR,
                    "Could not extract the DRAG quadrature scaling for %s "
                    "because the slope was zero.",
                    q.uid,
                )
    return qubit_parameters


@workflow.task
def plot_population(
    qubits: Qubits,
    processed_data_dict: dict[str, dict[str, dict[str, ArrayLike]]],
    fit_results: dict[str, dict[str, lmfit.model.ModelResult]] | None,
    qubit_parameters: dict[
        str,
        dict[str, dict[str, int | float | unc.core.Variable | None]],
    ]
    | None,
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the DRAG quadrature-scaling calibration plots.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in
            processed_data_dict and qubit_parameters.
        processed_data_dict: the processed data dictionary returned by process_raw_data
        fit_results: the fit-results dictionary returned by fit_data
        qubit_parameters: the qubit-parameters dictionary returned by
            extract_qubit_parameters
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
        pulse_ids = list(processed_data_dict[q.uid])
        fig, ax = plt.subplots()
        ax.set_title(timestamped_title(f"DRAG Q-Scaling {q.uid}"))
        ax.set_xlabel("Quadrature Scaling Factor, $\\beta$")
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
            # plot data
            [line] = ax.plot(sweep_points, data, "o", zorder=2, label=pulse_id)
            if opts.do_fitting and q.uid in fit_results:
                fit_res_qb = fit_results[q.uid][pulse_id]
                # plot fit
                sweep_points = processed_data_dict[q.uid][pulse_id]["sweep_points"]
                swpts_fine = np.linspace(sweep_points[0], sweep_points[-1], 501)
                ax.plot(
                    swpts_fine,
                    fit_res_qb.model.func(swpts_fine, **fit_res_qb.best_values),
                    c=line.get_color(),
                    zorder=1,
                    label="fit",
                )

        # the block plotting the lines at the calibration traces needs to come after
        # the xlims have been determined by plotting the data because we want these
        # lines to extend across the entire width of the axis
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

        if len(qubit_parameters["new_parameter_values"][q.uid]) > 0:
            new_beta = qubit_parameters["new_parameter_values"][q.uid][
                f"{opts.transition}_drive_pulse.beta"
            ]
            # point at the optimal quadrature scaling factor
            fit_res_qb = fit_results[q.uid][pulse_ids[0]]
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
                f"{opts.transition}_drive_pulse.beta"
            ]
            textstr = (
                "$\\beta$: "
                f"{new_beta.nominal_value:.4f} $\\pm$ "
                f"{new_beta.std_dev:.4f}"
            )
            textstr += "\nPrevious value: " + f"{old_beta:.4f}"
            ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            handlelength=1.5,
            frameon=False,
        )

        if opts.save_figures:
            workflow.save_artifact(f"Drag_q_scaling_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures
