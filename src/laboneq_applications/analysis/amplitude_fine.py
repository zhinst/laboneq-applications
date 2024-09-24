"""This module defines the analysis for an amplitude-fine experiment.

The experiment is defined in laboneq_applications.experiments.

In this analysis, we first interpret the raw data into qubit populations using
principle-component analysis or rotation and projection on the measured calibration
states. Then we fit a cosine model to the qubit population and extract ...
. Finally, we plot the data and the fit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc

from laboneq_applications.analysis.cal_trace_rotation import calculate_qubit_population
from laboneq_applications.analysis.fitting_helpers import cosine_oscillatory_fit
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
    amplification_qop: str,
    repetitions: QubitSweepPoints,
    target_angle: float,
    phase_offset: float,
    parameter_to_update: str | None = None,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The amplitude-fine analysis Workflow.

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
        amplification_qop:
            str to identify the quantum operation to repeat to produce error
            amplification.
        repetitions:
            Number of qop repetitions to sweep over. If `qubits` is a
            single qubit, `repetitions` must be a list of integers. Otherwise
            it must be a list of lists of integers.
        target_angle:
            target angle the specified quantum operation shuould rotate.
            The target_angle is used as initial guess for fitting.
        phase_offset:
            initial guess for phase_offset of fit.
        parameter_to_update:
            str that defines the qubit parameter to be updated.
        options:
            The options for building the workflow, passed as an instance of
                [TuneUpAnalysisWorkflowOptions]. See the docstring of this class
                for more information.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        options = TuneUpAnalysisWorkflowOptions()
        result = analysis_workflow(
            results=results
            qubits=[q0, q1],
            amplification_qop='x180',
            repetitions=[
                [1,2,3,4],
                [1,2,3,4],
            ],
            options=options,
        ).run()
        ```
    """
    processed_data_dict = calculate_qubit_population(qubits, result, repetitions)
    fit_results = fit_data(qubits, processed_data_dict, target_angle, phase_offset)
    processed_fit_results = process_fit_results(qubits, fit_results, target_angle)
    qubit_parameters = extract_qubit_parameters(
        qubits,
        fit_results,
        processed_fit_results,
        parameter_to_update,
    )
    with if_(options.do_plotting):
        with if_(options.do_raw_data_plotting):
            xlabel = xaxis_label(amplification_qop)
            plot_raw_complex_data_1d(
                qubits,
                result,
                repetitions,
                xlabel=xlabel,
            )
        with if_(options.do_qubit_population_plotting):
            plot_population(
                qubits,
                processed_data_dict,
                amplification_qop,
                fit_results,
                processed_fit_results,
                parameter_to_update,
                qubit_parameters,
            )


@task
def xaxis_label(amplification_qop: str) -> str:
    """Small task to create the x-axis label based on amplification_qop.

    Arguments:
        amplification_qop:
            str to identify the quantum operation to repeat to produce error
            amplification.

    Returns:
        strings with the x-axis label for the amplitude-fine experiment
    """
    return f"Number of Repetitions {amplification_qop}"


@task
def fit_data(
    qubits: Qubits,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    target_angle: float,
    phase_offset: float,
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, lmfit.model.ModelResult]:
    """Perform a fit of a cosine model to the data.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
        processed_data_dict: the processed data dictionary returned by process_raw_data
        target_angle:
            target angle the specified quantum operation shuould rotate.
            The target_angle is used as initial guess for fitting.
        phase_offset:
            initial guess for phase_offset of fit.
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
            if opts.fit_parameters_hints is None:
                opts.fit_parameters_hints = {
                    "frequency": {"value": target_angle, "min": 0},
                    "phase": {"value": phase_offset},
                    "amplitude": {"value": 0.5, "vary": False},
                    "offset": {"value": 0.5, "vary": False},
                }
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
def process_fit_results(
    qubits: Qubits,
    fit_results: dict[str, lmfit.model.ModelResult] | None,
    target_angle: float,
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, unc.core.Variable]:
    """Process fit results to extract correction factor.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
        fit_results:
            the fit-results dictionary returned by fit_data
        target_angle:
            target angle the specified quantum operation shuould rotate.
            The target_angle is used as initial guess for fitting.
        options:
            The options for processing the raw data.
            See [TuneupAnalysisOptions], [TuneupExperimentOptions] and
            [BaseExperimentOptions] for accepted options.
            Overwrites the options from [TuneupAnalysisOptions],
            [TuneupExperimentOptions] and [BaseExperimentOptions].

    Returns:
        dict with qubit UIDs as keys and the processed fit results for each qubit
        as keys.
    """
    opts = TuneupAnalysisOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    processed_fit_results = {}
    if not opts.do_fitting:
        return processed_fit_results

    for q in qubits:
        processed_fit_results[q.uid] = {}
        # Extract and store the correction factor from the fit
        correction_factor = unc.ufloat(
            fit_results[q.uid].params["frequency"].value / target_angle,
            fit_results[q.uid].params["frequency"].stderr / target_angle,
        )
        processed_fit_results[q.uid]["correction_factor"] = correction_factor

    return processed_fit_results


@task
def extract_qubit_parameters(
    qubits: Qubits,
    fit_results: dict[str, lmfit.model.ModelResult],
    processed_fit_results: dict[str, unc.core.Variable] | None,
    parameter_to_update: str | None = None,
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
        processed_fit_results:
            the processed fit-results dictionary returned by process_fit_data
        parameter_to_update:
            str that defines the paramter to be updated.
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

        if parameter_to_update is not None:
            old_parameter_value = (
                getattr(q.parameters, f"ef_{parameter_to_update}")
                if "f" in opts.transition
                else getattr(q.parameters, f"ge_{parameter_to_update}")
            )
            qubit_parameters["old_parameter_values"][q.uid] = {
                f"{opts.transition}_{parameter_to_update}": old_parameter_value,
            }

            # Calculate new parameter value from correction factor
            new_parameter_value = unc.ufloat(
                old_parameter_value
                / processed_fit_results[q.uid]["correction_factor"].nominal_value,
                processed_fit_results[q.uid]["correction_factor"].std_dev
                * old_parameter_value
                / processed_fit_results[q.uid]["correction_factor"].nominal_value,
            )

            qubit_parameters["new_parameter_values"][q.uid] = {
                f"{opts.transition}_{parameter_to_update}": new_parameter_value,
            }

    return qubit_parameters


@task
def plot_population(
    qubits: Qubits,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    amplification_qop: str,
    fit_results: dict[str, lmfit.model.ModelResult] | None,
    processed_fit_results: dict[str, unc.core.Variable] | None,
    parameter_to_update: str,
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
        processed_data_dict:
            the processed data dictionary returned by process_raw_data
        fit_results:
            the fit-results dictionary returned by fit_data
        amplification_qop:
            str to identify the quantum operation to repeat.
        processed_fit_results:
            the processed fit-results dictionary returned by process_fit_data
        parameter_to_update:
            str that defines the paramter to be updated.
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
        base_name = f"Amplitude Fine {amplification_qop} {q.uid}"
        sweep_points = processed_data_dict[q.uid]["sweep_points"]
        data = processed_data_dict[q.uid][
            "population" if opts.do_rotation else "data_raw"
        ]
        num_cal_traces = processed_data_dict[q.uid]["num_cal_traces"]

        fig, ax = plt.subplots()
        ax.set_title(base_name)  # add timestamp here
        ax.set_xlabel(xaxis_label(amplification_qop))
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

            # corrected parameters
            if parameter_to_update is not None:
                param_name = (
                    "$A_{\\pi/2}$" if "pi2" in parameter_to_update else "$A_{\\pi}$"
                )
                new_parameter_value = qubit_parameters["new_parameter_values"][q.uid][
                    f"{options.transition}_{parameter_to_update}"
                ]
                # textbox
                old_parameter_value = qubit_parameters["old_parameter_values"][q.uid][
                    f"{options.transition}_{parameter_to_update}"
                ]
                textstr = (
                    f"{param_name} = "
                    f"{new_parameter_value.nominal_value:.4f} $\\pm$ "
                    f"{new_parameter_value.std_dev:.4f}"
                )
                textstr += f"\nOld {param_name} = " + f"{old_parameter_value:.4f}"
                ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)
            else:
                correction_factor = processed_fit_results[q.uid]["correction_factor"]
                textstr = (
                    "Pulse-Amplitude Correction Factor, $c$ = "
                    f"{correction_factor.nominal_value:.4f} $\\pm$ "
                    f"{correction_factor.std_dev:.4f}"
                )
                ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            handlelength=1.5,
            frameon=False,
        )

        if opts.save_figures:
            save_artifact(f"Amplitude_Fine_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures
