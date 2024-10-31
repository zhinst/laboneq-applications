"""This module defines the analysis for extracting T2 from a Hahn Echo experiment.

The experiment is defined in laboneq_applications.experiments. See the docstring of
this file for more details on how the experiment is created.

In this analysis, we first interpret the raw data into qubit populations using
principal component analysis or rotation and projection on the measured calibration
states. Then we fit an exponential decay model to the qubit population and extract the
qubit dephasing time T2 from the fit. Finally, we plot the data and the fit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from laboneq import workflow

from laboneq_applications.analysis import plotting_helpers as plt_hlp
from laboneq_applications.analysis.calibration_traces_rotation import (
    calculate_qubit_population,
)
from laboneq_applications.analysis.fitting_helpers import exponential_decay_fit
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
    from laboneq.simple import Results
    from numpy.typing import ArrayLike

    from laboneq_applications.tasks.run_experiment import RunExperimentResults
    from laboneq_applications.typing import Qubits, QubitSweepPoints


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: Qubits,
    delays: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The Hahn echo analysis workflow.

    The workflow consists of the following steps:

    - [calculate_qubit_population]()
    - [fit_data]()
    - [extract_qubit_parameters]()
    - [plot_raw_complex_data_1d]()
    - [plot_population]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task of the
            Hahn echo experiment workflow.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        delays:
            The delays to sweep over for each qubit. The delays between the two x90
            pulses and the refocusing pulse are `delays / 2`; see the schematic of
            the pulse sequence in the file defining the experiment. Note that `delays`
            must be identical for qubits that use the same measure port.
        options:
            The options for building the workflow as an instance of
            [TuneUpAnalysisWorkflowOptions]. See the docstrings of this class for more
            details.

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
            plot_raw_complex_data_1d(qubits, result, delays)
        with workflow.if_(options.do_qubit_population_plotting):
            plot_population(qubits, processed_data_dict, fit_results, qubit_parameters)
    workflow.return_(qubit_parameters)


@workflow.task
def fit_data(
    qubits: Qubits,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, lmfit.model.ModelResult]:
    """Perform a fit of an exponential-decay model to the qubit e-state population.

    Arguments:
        qubits:
            The qubits on which to run the task. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in processed_data_dict
        processed_data_dict: the processed data dictionary containing the qubit
            population to be fitted and the sweep points of the experiment.
        options:
            The options for building the workflow as an instance of
            [TuneupAnalysisOptions]. See the docstrings of this class for more details.

    Returns:
        dict with qubit UIDs as keys and the fit results for each qubit as values.
    """
    opts = TuneupAnalysisOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    fit_results = {}
    if not opts.do_fitting:
        return fit_results

    for q in qubits:
        echo_pulse_length = (
            q.parameters.ef_drive_length
            if "f" in opts.transition
            else q.parameters.ge_drive_length
        )
        swpts_fit = processed_data_dict[q.uid]["sweep_points"] + echo_pulse_length
        data_to_fit = processed_data_dict[q.uid]["population"]

        param_hints = {
            "amplitude": {"value": 0.5},
            "offset": {"value": 0.5, "vary": opts.do_pca or not opts.use_cal_traces},
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
    qubits: Qubits,
    fit_results: dict[str, lmfit.model.ModelResult],
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, dict[str, dict[str, int | float | unc.core.Variable | None]]]:
    """Extract the new T2 values for each qubit from the fit results.

    Arguments:
        qubits:
            The qubits on which to run the task. May be either a single qubit or
            a list of qubits.
        fit_results: the fit-results dictionary returned by the task `fit_data`.
        options:
            The options for building the workflow as an instance of
            [TuneupAnalysisOptions]. See the docstrings of this class for more details.

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
        # Store the old T1 value
        old_t2 = q.parameters.ef_T2 if "f" in opts.transition else q.parameters.ge_T2
        qubit_parameters["old_parameter_values"][q.uid] = {
            f"{opts.transition}_T2": old_t2,
        }

        if opts.do_fitting and q.uid in fit_results:
            # Extract and store the T1 value
            fit_res = fit_results[q.uid]
            dec_rt = unc.ufloat(
                fit_res.params["decay_rate"].value, fit_res.params["decay_rate"].stderr
            )
            t2 = 1 / dec_rt

            qubit_parameters["new_parameter_values"][q.uid] = {
                f"{opts.transition}_T2": t2,
            }

    return qubit_parameters


@workflow.task
def plot_raw_complex_data_1d(
    qubits: Qubits,
    result: RunExperimentResults | tuple[RunExperimentResults, Results],
    delays: QubitSweepPoints,
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, dict[str, ArrayLike]]:
    """Creates the raw-data plots for the Hahn Echo experiment.

    Calls plot_raw_complex_data_1d. This task is needed because we want to plot the
    raw data as a function of the total time separation between the two x90 pulses in
    the experiment. Therefore, we need to first add the length of the refocusing pulse
    to the delays before calling plot_raw_complex_data_1d.

    Args:
        result:
            The experiment results returned by the run_experiment task of the
            Hahn echo experiment workflow.
        qubits:
            The qubits on which to run the task. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        delays:
            The delays to sweep over for each qubit. The delays between the two x90
            pulses and the refocusing pulse are `delays / 2`; see the schematic of
            the pulse sequence in the file defining the experiment. Note that `delays`
            must be identical for qubits that use the same measure port.
        options:
            The options for building the workflow as an instance of
            [TuneupAnalysisOptions]. See the docstrings of this class for more details.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = TuneupAnalysisOptions() if options is None else options
    validate_result(result)
    qubits, delays = validate_and_convert_qubits_sweeps(qubits, delays)

    figures = {}
    for q, qubit_delays in zip(qubits, delays):
        echo_pulse_length = (
            q.parameters.ef_drive_length
            if "f" in opts.transition
            else q.parameters.ge_drive_length
        )
        qubit_fig = plt_hlp.plot_raw_complex_data_1d(
            q,
            result,
            np.array(qubit_delays) + echo_pulse_length,
            xlabel="x90-Pulse Separation, $\\tau$ ($\\mu$s)",
            xscaling=1e6,
            options=opts,
        )
        figures.update(qubit_fig)

    return figures


@workflow.task
def plot_population(
    qubits: Qubits,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, lmfit.model.ModelResult] | None,
    qubit_parameters: dict[
        str,
        dict[str, dict[str, int | float | unc.core.Variable | None]],
    ]
    | None,
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the Hahn echo plots.

    Arguments:
        qubits:
            The qubits on which to run the task. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in
            processed_data_dict, fit_results and qubit_parameters.
        processed_data_dict: the processed data dictionary returned by process_raw_data
        fit_results: the fit-results dictionary returned by fit_data
        qubit_parameters: the qubit-parameters dictionary returned by
            extract_qubit_parameters
        options:
            The options for processing the raw data.
            See [TuneupAnalysisOptions], [TuneupExperimentOptions] and
            [BaseExperimentOptions] for accepted options.
            Overwrites the options from [TuneupAnalysisOptions],
            [TuneupExperimentOptions] and [BaseExperimentOptions].

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.

        If a qubit uid is not found in fit_results, the fit and the textbox with the
        extracted qubit parameters are not plotted.
    """
    opts = TuneupAnalysisOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}
    for q in qubits:
        echo_pulse_length = (
            q.parameters.ef_drive_length
            if "f" in opts.transition
            else q.parameters.ge_drive_length
        )
        sweep_points = processed_data_dict[q.uid]["sweep_points"] + echo_pulse_length
        data = processed_data_dict[q.uid][
            "population" if opts.do_rotation else "data_raw"
        ]
        num_cal_traces = processed_data_dict[q.uid]["num_cal_traces"]

        fig, ax = plt.subplots()
        ax.set_title(plt_hlp.timestamped_title(f"Echo {q.uid}"))
        ax.set_xlabel("x90-Pulse Separation, $\\tau$ ($\\mu$s)")
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
            if len(qubit_parameters["new_parameter_values"][q.uid]) > 0:
                old_t2 = qubit_parameters["old_parameter_values"][q.uid][
                    f"{opts.transition}_T2"
                ]
                new_t2 = qubit_parameters["new_parameter_values"][q.uid][
                    f"{opts.transition}_T2"
                ]
                textstr = (
                    "$T_2$: "
                    f"{new_t2.nominal_value*1e6:.4f} $\\mu$s $\\pm$ "
                    f"{new_t2.std_dev*1e6:.4f} $\\mu$s"
                )
                textstr += "\nPrevious value: " + f"{old_t2*1e6:.4f} $\\mu$s"
                ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)

        # Add legend
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            handlelength=1.5,
            frameon=False,
        )

        if opts.save_figures:
            workflow.save_artifact(f"T2_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures
