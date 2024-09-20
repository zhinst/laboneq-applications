"""This module defines the analysis for an amplitude-rabi experiment.

The experiment is defined in laboneq_applications.experiments.

In this analysis, we first interpret the raw data into qubit populations using
principle-component analysis or rotation and projection on the measured calibration
states. Then we fit a cosine model to the qubit population and extract the pi and
pi-half pulse amplitudes from the fit. Finally, we plot the data and the fit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc

from laboneq_applications.analysis.cal_trace_rotation import calculate_qubit_population
from laboneq_applications.analysis.fitting_helpers import (
    cosine_oscillatory_fit,
    get_pi_pi2_xvalues_on_cos,
)
from laboneq_applications.analysis.plotting_helpers import plot_raw_complex_data_1d
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
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

options = TuneUpAnalysisWorkflowOptions


@workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: Qubits,
    amplitudes: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The Amplitude Rabi analysis Workflow.

    The workflow consists of the following steps:

    - [calculate_qubit_population]()
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
        amplitudes:
            The amplitudes that were swept over in the amplitude-Rabi experiment for
            each qubit. If `qubits` is a single qubit, `amplitudes` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        options:
            The options for building the workflow, passed as an instance of
                TuneUpAnalysisWorkflowOptions.
            In addition to options from [WorkflowOptions], the following
            custom options are supported:
                - process_raw_data: The options for creating the experiment as an
                    instance of TuneupAnalysisOptions.
                - fit_data: The options for performing a fit, passed as an
                    instance of TuneupAnalysisOptions.
                - extract_qubit_parameters: The options for extracting qubit parameters
                    from the fit, passed as an instance of TuneupAnalysisOptions.
                - plot_data: The options for plotting, passed as an instance of
                    TuneupAnalysisOptions.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        options = TuneUpAnalysisWorkflowOptions()
        result = analysis_workflow(
            results=results
            qubits=[q0, q1],
            amplitudes=[
                np.linspace(0, 1, 11),
                np.linspace(0, 0.75, 11),
            ],
            options=options,
        ).run()
        ```
    """
    processed_data_dict = calculate_qubit_population(qubits, result, amplitudes)
    fit_results = fit_data(qubits, processed_data_dict)
    qubit_parameters = extract_qubit_parameters(
        qubits, processed_data_dict, fit_results
    )
    with if_(options.do_plotting):
        with if_(options.do_raw_data_plotting):
            plot_raw_complex_data_1d(
                qubits, result, amplitudes, xlabel="Amplitude Scaling"
            )
        with if_(options.do_qubit_population_plotting):
            plot_population(qubits, processed_data_dict, fit_results, qubit_parameters)


@task
def fit_data(
    qubits: Qubits,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, lmfit.model.ModelResult]:
    """Perform a fit of a cosine model to the data.

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
            Overwrites the options from [TuneupAnalysisOptions],
            [TuneupExperimentOptions] and [BaseExperimentOptions].

    Returns:
        dict with qubit UIDs as keys and the fit results for each qubit as keys.
    """
    opts = TuneupAnalysisOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    fit_results = {}
    if not opts.do_fitting:
        return fit_results

    for q in qubits:
        swpts_fit = processed_data_dict[q.uid]["sweep_points"]
        data_to_fit = processed_data_dict[q.uid][
            "population" if opts.do_rotation else "data_raw"
        ]
        if not opts.do_rotation:
            data_to_fit = np.array(data_to_fit, dtype=np.int32)
        try:
            fit_res = cosine_oscillatory_fit(
                swpts_fit,
                data_to_fit,
                param_hints=opts.fit_parameters_hints,
            )
            fit_results[q.uid] = fit_res
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
            processed_data_dict and fit_results.
        processed_data_dict: the processed data dictionary returned by process_raw_data
        fit_results: the fit-results dictionary returned by fit_data
        options:
            The options for extracting the qubit parameters.
            See [TuneupAnalysisOptions], [TuneupExperimentOptions] and
            [BaseExperimentOptions] for accepted options.
            Overwrites the options from [TuneupAnalysisOptions],
            [TuneupExperimentOptions] and [BaseExperimentOptions].

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
    Raises:
        ValueError:
            If fit_results are empty (have length 0).
    """
    opts = TuneupAnalysisOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    qubit_parameters = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
    }
    if not opts.do_fitting:
        return qubit_parameters

    for q in qubits:
        if q.uid not in fit_results:
            continue

        # Store the old pi and pi-half pulse amplitude values
        old_pi_amp = (
            q.parameters.ef_drive_amplitude_pi
            if "f" in opts.transition
            else q.parameters.ge_drive_amplitude_pi
        )
        old_pi2_amp = (
            q.parameters.ef_drive_amplitude_pi2
            if "f" in opts.transition
            else q.parameters.ge_drive_amplitude_pi2
        )
        qubit_parameters["old_parameter_values"][q.uid] = {
            f"{opts.transition}_drive_amplitude_pi": old_pi_amp,
            f"{opts.transition}_drive_amplitude_pi2": old_pi2_amp,
        }

        # Extract and store the new pi and pi-half pulse amplitude values
        fit_res = fit_results[q.uid]
        swpts_fit = processed_data_dict[q.uid]["sweep_points"]
        freq_fit = unc.ufloat(
            fit_res.params["frequency"].value,
            fit_res.params["frequency"].stderr,
        )
        phase_fit = unc.ufloat(
            fit_res.params["phase"].value,
            fit_res.params["phase"].stderr,
        )
        (
            pi_amps_top,
            pi_amps_bottom,
            pi2_amps_rise,
            pi2_amps_fall,
        ) = get_pi_pi2_xvalues_on_cos(swpts_fit, freq_fit, phase_fit)

        # if pca is done, it can happen that the pi-pulse amplitude
        # is in pi_amps_bottom and the pi/2-pulse amplitude in pi2_amps_fall
        pi_amps = np.sort(np.concatenate([pi_amps_top, pi_amps_bottom]))
        pi2_amps = np.sort(np.concatenate([pi2_amps_rise, pi2_amps_fall]))
        try:
            pi2_amp = pi2_amps[0]
            pi_amp = pi_amps[pi_amps > pi2_amp][0]
        except IndexError:
            comment(f"Could not extract pi- and pi/2-pulse amplitudes for {q.uid}.")
            continue

        qubit_parameters["new_parameter_values"][q.uid] = {
            f"{opts.transition}_drive_amplitude_pi": pi_amp,
            f"{opts.transition}_drive_amplitude_pi2": pi2_amp,
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
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the amplitude-Rabi plots.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
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
    """
    opts = TuneupAnalysisOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}
    for q in qubits:
        base_name = f"Amplitude_Rabi_{q.uid}"
        sweep_points = processed_data_dict[q.uid]["sweep_points"]
        data = processed_data_dict[q.uid][
            "population" if opts.do_rotation else "data_raw"
        ]
        num_cal_traces = processed_data_dict[q.uid]["num_cal_traces"]

        fig, ax = plt.subplots()
        ax.set_title(base_name)  # add timestamp here
        ax.set_xlabel("Amplitude Scaling")
        ax.set_ylabel(
            "Principal Component (a.u)"
            if (num_cal_traces == 0 or opts.do_pca)
            else f"$|{opts.cal_states[-1]}\\rangle$-State Population",
        )
        ax.plot(sweep_points, data, "o", zorder=2, label="data")
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

        if opts.do_fitting:
            if q.uid not in fit_results:
                continue
            fit_res_qb = fit_results[q.uid]

            # plot fit
            swpts_fine = np.linspace(sweep_points[0], sweep_points[-1], 501)
            ax.plot(
                swpts_fine,
                fit_res_qb.model.func(swpts_fine, **fit_res_qb.best_values),
                "r-",
                zorder=1,
                label="fit",
            )

            if len(qubit_parameters["new_parameter_values"][q.uid]) == 0:
                continue
            new_pi_amp = qubit_parameters["new_parameter_values"][q.uid][
                f"{options.transition}_drive_amplitude_pi"
            ]
            new_pi2_amp = qubit_parameters["new_parameter_values"][q.uid][
                f"{options.transition}_drive_amplitude_pi2"
            ]
            # point at pi-pulse amplitude
            ax.plot(
                new_pi_amp.nominal_value,
                fit_res_qb.model.func(
                    new_pi_amp.nominal_value,
                    **fit_res_qb.best_values,
                ),
                "sk",
                zorder=3,
                markersize=plt.rcParams["lines.markersize"] + 1,
            )
            # point at pi/2-pulse amplitude
            ax.plot(
                new_pi2_amp.nominal_value,
                fit_res_qb.model.func(
                    new_pi2_amp.nominal_value,
                    **fit_res_qb.best_values,
                ),
                "sk",
                zorder=3,
                markersize=plt.rcParams["lines.markersize"] + 1,
            )
            # textbox
            old_pi_amp = qubit_parameters["old_parameter_values"][q.uid][
                f"{options.transition}_drive_amplitude_pi"
            ]
            old_pi2_amp = qubit_parameters["old_parameter_values"][q.uid][
                f"{options.transition}_drive_amplitude_pi2"
            ]
            textstr = (
                "$A_{\\pi}$: "
                f"{new_pi_amp.nominal_value:.4f} $\\pm$ "
                f"{new_pi_amp.std_dev:.4f}"
            )
            textstr += "\nOld $A_{\\pi}$: " + f"{old_pi_amp:.4f}"
            ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)
            textstr = (
                "$A_{\\pi/2}$: "
                f"{new_pi2_amp.nominal_value:.4f} $\\pm$ "
                f"{new_pi2_amp.std_dev:.4f}"
            )
            textstr += "\nOld $A_{\\pi/2}$: " + f"{old_pi2_amp:.4f}"
            ax.text(
                0.69,
                -0.15,
                textstr,
                ha="left",
                va="top",
                transform=ax.transAxes,
            )
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
