# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the analysis for an lifetime_measurement experiment.

The experiment is defined in laboneq_applications.experiments.

In this analysis, we first interpret the raw data into qubit population using
principle-component analysis or rotation and projection on the measured calibration
states. Then we fit an exponential-decay model to the qubit population and extract the
qubit energy relaxation time lifetime_measurement from the fit. Finally, we plot the
data and the fit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from laboneq import workflow

from laboneq_applications.analysis.calibration_traces_rotation import (
    calculate_qubit_population,
)
from laboneq_applications.analysis.fitting_helpers import exponential_decay_fit
from laboneq_applications.analysis.options import (
    ExtractQubitParametersTransitionOptions,
    FitDataOptions,
    PlotPopulationOptions,
    TuneUpAnalysisWorkflowOptions,
)
from laboneq_applications.analysis.plotting_helpers import (
    plot_raw_complex_data_1d,
    timestamped_title,
)
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps

if TYPE_CHECKING:
    import lmfit
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.task_options(base_class=FitDataOptions)
class FitDataT1Options:
    """Options for the `fit_data` task of the Ramsey analysis.

    See [FitDataOptions] for additional accepted options.

    Attributes:
        do_pca:
            Whether to perform principal component analysis on the raw data independent
            of whether there were calibration traces in the experiment.
            Default: `False`.
        use_cal_traces:
            Whether to include calibration traces in the experiment.
            Default: `True`.
    """

    do_pca: bool = workflow.option_field(
        False,
        description="Whether to perform principal component analysis on the raw data"
        " independent of whether there were calibration traces in the experiment.",
    )
    use_cal_traces: bool = workflow.option_field(
        True, description="Whether to include calibration traces in the experiment."
    )


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    delays: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The lifetime_measurement analysis Workflow.

    The workflow consists of the following steps:

    - [calculate_qubit_population]()
    - [fit_data]()
    - [extract_qubit_parameters]()
    - [plot_raw_complex_data_1d]()
    - [plot_population]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task of the
            lifetime_measurement experiment workflow.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        delays:
            The delays that were swept over in the lifetime_measurement experiment for
            each qubit. If `qubits` is a single qubit, `delays` must be an array of
            numbers. Otherwise, it must be a list of arrays of numbers.
        options:
            The options for building the workflow, passed as an instance of
            [TuneUpAnalysisWorkflowOptions]. See the docstring of this class for
            more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        options = TuneUpAnalysisWorkflowOptions()
        result = analysis_workflow(
            results=results
            qubits=[q0, q1],
            delays=[
                np.linspace(0, 10e-6, 11),
                np.linspace(0, 10e-6, 11),
            ],
            options=options,
        ).run()
        ```
    """
    processed_data_dict = calculate_qubit_population(qubits, result, delays)
    fit_results = fit_data(qubits, processed_data_dict)
    qubit_parameters = extract_qubit_parameters(qubits, fit_results)
    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_raw_data_plotting):
            plot_raw_complex_data_1d(
                qubits,
                result,
                delays,
                xlabel="Pulse Delay, $\\tau$ ($\\mu$s)",
                xscaling=1e6,
            )
        with workflow.if_(options.do_qubit_population_plotting):
            plot_population(qubits, processed_data_dict, fit_results, qubit_parameters)
    workflow.return_(qubit_parameters)


@workflow.task
def fit_data(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: FitDataT1Options | None = None,
) -> dict[str, lmfit.model.ModelResult]:
    """Perform a fit of an exponential-decay model to the qubit state population.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in processed_data_dict
        processed_data_dict: the processed data dictionary returned by process_raw_data
        options:
            The options class for this task as an instance of [FitDataT1Options]. See
            the docstring of this class for accepted options.

    Returns:
        dict with qubit UIDs as keys and the fit results for each qubit as keys.
    """
    opts = FitDataT1Options() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    fit_results = {}
    if not opts.do_fitting:
        return fit_results

    for q in qubits:
        swpts_fit = processed_data_dict[q.uid]["sweep_points"]
        data_to_fit = processed_data_dict[q.uid][
            "population" if opts.do_rotation else "data_raw"
        ]
        param_hints = {
            "offset": {"value": 0, "vary": opts.do_pca or not opts.use_cal_traces},
        }
        param_hints_user = opts.fit_parameters_hints
        if param_hints_user is None:
            param_hints_user = {}
        param_hints.update(param_hints_user)
        try:
            fit_res = exponential_decay_fit(
                swpts_fit,
                data_to_fit,
                param_hints=param_hints,
            )
            fit_results[q.uid] = fit_res
        except ValueError as err:
            workflow.log(logging.ERROR, "Fit failed for %s: %s.", q.uid, err)

    return fit_results


@workflow.task
def extract_qubit_parameters(
    qubits: QuantumElements,
    fit_results: dict[str, lmfit.model.ModelResult],
    options: ExtractQubitParametersTransitionOptions | None = None,
) -> dict[str, dict[str, dict[str, int | float | unc.core.Variable | None]]]:
    """Extract the qubit parameters from the fit results.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits.
        fit_results: the fit-results dictionary returned by fit_data
        options:
            The options for extracting the qubit parameters.
            See [ExtractQubitParametersTransitionOptions] for accepted options.

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
    opts = ExtractQubitParametersTransitionOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    qubit_parameters = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
    }

    for q in qubits:
        # Store the old lifetime_measurement value
        old_t1 = q.parameters.ef_T1 if "f" in opts.transition else q.parameters.ge_T1
        qubit_parameters["old_parameter_values"][q.uid] = {
            f"{opts.transition}_T1": old_t1,
        }

        if opts.do_fitting and q.uid in fit_results:
            # Extract and store the lifetime_measurement value
            fit_res = fit_results[q.uid]
            dec_rt = unc.ufloat(
                fit_res.params["decay_rate"].value, fit_res.params["decay_rate"].stderr
            )
            lifetime_measurement = 1 / dec_rt

            qubit_parameters["new_parameter_values"][q.uid] = {
                f"{opts.transition}_T1": lifetime_measurement,
            }

    return qubit_parameters


@workflow.task
def plot_population(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, lmfit.model.ModelResult] | None,
    qubit_parameters: dict[
        str,
        dict[str, dict[str, int | float | unc.core.Variable | None]],
    ]
    | None,
    options: PlotPopulationOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the lifetime_measurement plots.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in
            processed_data_dict.
        processed_data_dict: the processed data dictionary returned by process_raw_data
        fit_results: the fit-results dictionary returned by fit_data
        qubit_parameters: the qubit-parameters dictionary returned by
            extract_qubit_parameters
        options:
            The options class for this task as an instance of [PlotPopulationOptions].
            See the docstring of this class for accepted options.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.

        If a qubit uid is not found in fit_results, the fit and the textbox with the
        extracted qubit parameters are not plotted.
    """
    opts = PlotPopulationOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}
    for q in qubits:
        sweep_points = processed_data_dict[q.uid]["sweep_points"]
        data = processed_data_dict[q.uid][
            "population" if opts.do_rotation else "data_raw"
        ]
        num_cal_traces = processed_data_dict[q.uid]["num_cal_traces"]

        fig, ax = plt.subplots()
        ax.set_title(timestamped_title(f"Lifetime Measurement {q.uid}"))
        ax.set_xlabel("Pulse Delay, $\\tau$ ($\\mu$s)")
        ax.set_ylabel(
            "Principal Component (a.u)"
            if (num_cal_traces == 0 or opts.do_pca)
            else f"$|{opts.cal_states[-1]}\\rangle$-State Population",
        )
        ax.plot(sweep_points * 1e6, data, "o", zorder=2, label="data")
        if processed_data_dict[q.uid]["num_cal_traces"] > 0:
            # plot lines at the calibration traces
            xlims = ax.get_xlim()
            ax.hlines(
                processed_data_dict[q.uid]["population_cal_traces"],
                *xlims,
                linestyles="--",
                colors="gray",
                zorder=0,
                label="calib.\ntraces",
            )
            ax.set_xlim(xlims)

        if opts.do_fitting and q.uid in fit_results:
            fit_res_qb = fit_results[q.uid]

            # Plot fit
            swpts_fine = np.linspace(sweep_points[0], sweep_points[-1], 501)
            ax.plot(
                swpts_fine * 1e6,
                fit_res_qb.model.func(swpts_fine, **fit_res_qb.best_values),
                "r-",
                zorder=1,
                label="fit",
            )

            # Add textbox
            if (
                qubit_parameters is not None
                and len(qubit_parameters["new_parameter_values"][q.uid]) > 0
            ):
                old_t1 = qubit_parameters["old_parameter_values"][q.uid][
                    f"{opts.transition}_T1"
                ]
                new_t1 = qubit_parameters["new_parameter_values"][q.uid][
                    f"{opts.transition}_T1"
                ]
                textstr = (
                    "$T_1$: "
                    f"{new_t1.nominal_value*1e6:.4f} $\\mu$s $\\pm$ "
                    f"{new_t1.std_dev*1e6:.4f} $\\mu$s"
                )
                textstr += "\nPrevious value: " + f"{old_t1*1e6:.4f} $\\mu$s"
                ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)

        # Add legend
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            handlelength=1.5,
            frameon=False,
        )

        if opts.save_figures:
            workflow.save_artifact(f"T1_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures
