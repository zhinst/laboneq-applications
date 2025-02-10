# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the analysis for a time-rabi experiment.

The experiment is defined in laboneq_applications.experiments.

In this analysis, we first interpret the raw data into qubit populations using
principle-component analysis or rotation and projection on the measured calibration
states. Then we fit a cosine model to the qubit population and extract the pi pulse
lengths from the fit. Finally, we plot the data and the fit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from laboneq.workflow import (
    comment,
    if_,
    save_artifact,
    task,
    workflow,
)

from laboneq_applications.analysis.amplitude_rabi import (
    fit_data,
)
from laboneq_applications.analysis.calibration_traces_rotation import (
    calculate_qubit_population,
)
from laboneq_applications.analysis.fitting_helpers import (
    get_pi_pi2_xvalues_on_cos,
)
from laboneq_applications.analysis.options import (
    ExtractQubitParametersTransitionOptions,
    PlotPopulationOptions,
    TuneUpAnalysisWorkflowOptions,
)
from laboneq_applications.analysis.plotting_helpers import plot_raw_complex_data_1d
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps

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
    lengths: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The Time Rabi analysis Workflow.

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
        lengths:
            The lengths that were swept over in the time-rabi experiment for
            each qubit. If `qubits` is a single qubit, `lengths` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
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
        options = analysis_workflow.options()
        result = analysis_workflow(
            results=results
            qubits=[q0, q1],
            lengths=[
                np.linspace(10e-9, 100e-9, 11),
                np.linspace(10e-9, 100e-9, 11),
            ],
            options=options,
        ).run()
        ```
    """
    processed_data_dict = calculate_qubit_population(qubits, result, lengths)
    fit_results = fit_data(qubits, processed_data_dict)
    qubit_parameters = extract_qubit_parameters(
        qubits, processed_data_dict, fit_results
    )
    with if_(options.do_plotting):
        with if_(options.do_raw_data_plotting):
            plot_raw_complex_data_1d(
                qubits,
                result,
                lengths,
                xlabel="Drive-Pulse Length, $\\tau$ (ns)",
                xscaling=1e9,
            )
        with if_(options.do_qubit_population_plotting):
            plot_population(qubits, processed_data_dict, fit_results, qubit_parameters)


@task
def extract_qubit_parameters(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, lmfit.model.ModelResult],
    options: ExtractQubitParametersTransitionOptions | None = None,
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
    Raises:
        ValueError:
            If fit_results are empty (have length 0).
    """
    opts = ExtractQubitParametersTransitionOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    qubit_parameters = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
    }

    for q in qubits:
        # Store the old pi and pi-half pulse amplitude values
        old_drive_length = (
            q.parameters.ef_drive_length
            if "f" in opts.transition
            else q.parameters.ge_drive_length
        )
        qubit_parameters["old_parameter_values"][q.uid] = {
            f"{opts.transition}_drive_length": old_drive_length,
        }

        if opts.do_fitting and q.uid in fit_results:
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
                maxima,
                minima,
                rise,
                fall,
            ) = get_pi_pi2_xvalues_on_cos(swpts_fit, freq_fit, phase_fit)

            # if pca is done, it can happen that the pi-pulse amplitude
            # is in pi_amps_bottom and the pi/2-pulse amplitude in pi2_amps_fall
            pi_drive_lengths = np.sort(np.concatenate([maxima, minima]))
            pi2_drive_lengths = np.sort(np.concatenate([rise, fall]))
            try:
                pi2_drive_length = pi2_drive_lengths[0]
                pi_drive_length = pi_drive_lengths[pi_drive_lengths > pi2_drive_length][
                    0
                ]
            except IndexError:
                comment(f"Could not extract pi- and pi/2-pulse amplitudes for {q.uid}.")
                continue

            qubit_parameters["new_parameter_values"][q.uid] = {
                f"{opts.transition}_drive_length": pi_drive_length,
            }

    return qubit_parameters


@task
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
    """Create the time-Rabi plots.

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
            The options class for this task as an instance of [PlotPopulationOptions].
            See the docstring of this class for accepted options.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
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
        ax.set_title(f"Time Rabi {q.uid}")  # add timestamp here
        ax.set_xlabel("Drive-Pulse Length, $\\tau$ (ns)")
        ax.set_ylabel(
            "Principal Component (a.u)"
            if (num_cal_traces == 0 or opts.do_pca)
            else f"$|{opts.cal_states[-1]}\\rangle$-State Population",
        )
        ax.plot(sweep_points * 1e9, data, "o", zorder=2, label="data")
        if processed_data_dict[q.uid]["num_cal_traces"] > 0:
            # plot lines at the calibration traces
            xlims = np.array(ax.get_xlim())
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

            # plot fit
            swpts_fine = np.linspace(sweep_points[0], sweep_points[-1], 501)
            ax.plot(
                swpts_fine * 1e9,
                fit_res_qb.model.func(swpts_fine, **fit_res_qb.best_values),
                "r-",
                zorder=1,
                label="fit",
            )

            if len(qubit_parameters["new_parameter_values"][q.uid]) > 0:
                new_pi_length = qubit_parameters["new_parameter_values"][q.uid][
                    f"{opts.transition}_drive_length"
                ]
                # point at pi-pulse length
                ax.plot(
                    new_pi_length.nominal_value * 1e9,
                    fit_res_qb.model.func(
                        new_pi_length.nominal_value,
                        **fit_res_qb.best_values,
                    ),
                    "sk",
                    zorder=3,
                    markersize=plt.rcParams["lines.markersize"] + 1,
                )
                # textbox
                old_pi_length = qubit_parameters["old_parameter_values"][q.uid][
                    f"{opts.transition}_drive_length"
                ]
                textstr = (
                    "$\\tau_{\\pi}$ = "
                    f"{new_pi_length.nominal_value*1e9:.4f} $\\pm$ "
                    f"{new_pi_length.std_dev*1e9:.4f} ns"
                )
                textstr += "\nOld $\\tau_{\\pi}$ = " + f"{old_pi_length*1e9:.4f} ns"
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
