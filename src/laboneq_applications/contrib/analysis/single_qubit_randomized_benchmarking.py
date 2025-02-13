# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the analysis for single qubit randomized benchmarking experiment.

The experiment is defined in laboneq_applications.experiments.

In this analysis, we first interpret the raw data into qubit populations using
principal component analysis or rotation and projection on the measured calibration
states. Then we fit a exponential decay to the qubit population and extract the gate
fidelity from the fit. Finally, we plot the data and the fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from laboneq.analysis import fitting as fit_mods
from laboneq.workflow import (
    comment,
    if_,
    option_field,
    save_artifact,
    task,
    task_options,
    workflow,
)
from uncertainties.umath import exp

from laboneq_applications.analysis.calibration_traces_rotation import (
    CalculateQubitPopulationOptions,
    calculate_population_1d,
    extract_raw_data_dict,
)
from laboneq_applications.analysis.fitting_helpers import (
    fit_data_lmfit,
)
from laboneq_applications.analysis.options import (
    BasePlottingOptions,
    FitDataOptions,
    TuneUpAnalysisWorkflowOptions,
)
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_result,
)

if TYPE_CHECKING:
    import lmfit
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import QuantumElements


@task_options(base_class=BasePlottingOptions)
class PlotPopulationRBOptions:
    """Options for the `plot_population` task of the RB analysis.

    Attributes:
        do_fitting:
            Whether to perform the fit.
            Default: `True`.
        do_rotation:
            Whether to rotate the raw data based on calibration traces or principal
            component analysis.
            Default: `True`.
        do_pca:
            Whether to perform principal component analysis on the raw data independent
            of whether there were calibration traces in the experiment.
            Default: `False`.
        cal_states:
            The states to prepare in the calibration traces. Can be any
            string or tuple made from combining the characters 'g', 'e', 'f'.
            Default: same as transition

    Additional attributes from `BasePlottingOptions`:
        save_figures:
            Whether to save the figures.
            Default: `True`.
        close_figures:
            Whether to close the figures.
            Default: `True`.
    """

    do_fitting: bool = option_field(True, description="Whether to perform the fit.")
    do_rotation: bool = option_field(
        True,
        description="Whether to rotate the raw data based on calibration traces or "
        "principal component analysis.",
    )
    do_pca: bool = option_field(
        False,
        description="Whether to perform principal component analysis on the raw data"
        " independent of whether there were calibration traces in the experiment.",
    )
    cal_states: str | tuple = option_field(
        "ge", description="The states to prepare in the calibration traces."
    )


@workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    length_cliffords: list,
    variations: int,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The Time Rabi analysis Workflow.

    The workflow consists of the following steps:

    - [calculate_qubit_population_rb]()
    - [fit_data]()
    - [plot_population]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        length_cliffords:
            list of numbers of Clifford gates to sweep
        variations:
            Number of random seeds for RB.
        options:
            The options for building the workflow, passed as an instance of
            [TuneUpAnalysisWorkflowOptions]. See the docstring of this class for more
            details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    """
    processed_data_dict = calculate_qubit_population_rb(
        qubits, result, length_cliffords, variations
    )
    fit_results = fit_data(qubits, processed_data_dict)
    with if_(options.do_plotting):
        with if_(options.do_qubit_population_plotting):
            plot_population(qubits, processed_data_dict, fit_results)


@task
def calculate_qubit_population_rb(
    qubits: QuantumElements,
    result: RunExperimentResults,
    length_cliffords: list,
    variations: int,
    options: CalculateQubitPopulationOptions | None = None,
) -> dict[str, dict[str, ArrayLike]]:
    """Processes the raw data.

     The data is processed in the following way:

     - If calibration traces were used in the experiment, the raw data is rotated based
     on the calibration traces.
     See calibration_traces_rotation.py/rotate_data_to_cal_trace_results for more
     details.
     - If no calibration traces were used in the experiment, or do_pca = True is passed
     in options, principal component analysis is performed on the data.
     See calibration_traces_rotation.py/principal_component_analysis for more details.

    Arguments:
        qubits:
            The qubits on which the amplitude-Rabi experiments was run. May be either
            a single qubit or a list of qubits.
        result: the result of the experiment, returned by the run_experiment task.
        length_cliffords:
            list of numbers of Clifford gates to sweep
        variations:
            Number of random seeds for RB.
        options:
            The options for building the workflow as an instance of
            [CalculateQubitPopulationOptions]. See the docstrings of this class for more
            details.

    Returns:
        dict with qubit UIDs as keys and the dictionary of processed data for each qubit
        as values. See [calibration_traces_rotation.py/calculate_population_1d] for what
        this dictionary looks like.

    Raises:
        TypeError:
            If result is not an instance of RunExperimentResults.
    """
    validate_result(result)
    opts = CalculateQubitPopulationOptions() if options is None else options
    cliffords = np.concatenate([length_cliffords for i in range(variations)])
    if isinstance(qubits, Sequence):
        cliffords = [cliffords for _ in qubits]

    qubits, cliffords = validate_and_convert_qubits_sweeps(qubits, cliffords)
    processed_data_dict = {}
    for q, cliffs in zip(qubits, cliffords):
        raw_data = result[q.uid].result.data
        if opts.use_cal_traces:
            calibration_traces = [
                result[q.uid].cal_trace[cs].data for cs in opts.cal_states
            ]
            do_pca = opts.do_pca
        else:
            calibration_traces = []
            do_pca = True

        if opts.do_rotation:
            data_dict = calculate_population_1d(
                raw_data,
                cliffs,
                calibration_traces,
                do_pca=do_pca,
            )
        else:
            data_dict = extract_raw_data_dict(raw_data, cliffs, calibration_traces)
        processed_data_dict[q.uid] = data_dict
    return processed_data_dict


@task
def fit_data(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: FitDataOptions | None = None,
) -> dict[str, lmfit.model.ModelResult]:
    """Perform a fit of an exponential-decay model to the qubit e-state population.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in processed_data_dict
        processed_data_dict: the processed data dictionary returned by process_raw_data
        options:
            The options class for this task as an instance of [FitDataOptions]. See
            the docstring of this class for accepted options.

    Returns:
        dict with qubit UIDs as keys and the fit results for each qubit as keys.
    """
    opts = FitDataOptions() if options is None else options
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
            "amplitude": {"value": -0.5},
            "decay_rate": {"value": 1 / 50},
            "offset": {"value": 0.5, "vary": False},
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
            comment(f"Fit failed for {q.uid}: {err}.")

    return fit_results


@task
def plot_population(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, lmfit.model.ModelResult] | None,
    options: PlotPopulationRBOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the time-Rabi plots.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict and fit_results parameters.
        processed_data_dict:
            The processed data dictionary returned by process_raw_data.
        fit_results:
            The fit-results dictionary returned by fit_data.
        options:
            The options class for this task as an instance of [PlotPopulationRBOptions].
            See the docstring of this class for accepted options.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = PlotPopulationRBOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}
    for q in qubits:
        sweep_points = processed_data_dict[q.uid]["sweep_points"]
        data = processed_data_dict[q.uid][
            "population" if opts.do_rotation else "data_raw"
        ]
        num_cal_traces = processed_data_dict[q.uid]["num_cal_traces"]

        fig, ax = plt.subplots()
        ax.set_title(f"Randomized Benchmarking {q.uid}")
        ax.set_xlabel("Number of Cliffords")
        ax.set_ylabel(
            "Principal Component (a.u)"
            if (num_cal_traces == 0 or opts.do_pca)
            else f"$|{opts.cal_states[-1]}\\rangle$-State Population",
        )
        ax.plot(sweep_points, data, "o", zorder=2, label="data")
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
                swpts_fine,
                fit_res_qb.model.func(swpts_fine, **fit_res_qb.best_values),
                "r-",
                zorder=1,
                label="fit",
            )
            decay_fit = unc.ufloat(
                fit_res_qb.params["decay_rate"].value,
                fit_res_qb.params["decay_rate"].stderr,
            )
            fidelity_fit = exp(-decay_fit)
            textstr = (
                "gate fidelity = "
                f"{fidelity_fit.nominal_value:.4f} $\\pm$ "
                f"{fidelity_fit.std_dev:.4f}"
            )
            ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            handlelength=1.5,
            frameon=False,
        )

        if opts.save_figures:
            save_artifact(f"RB_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures


# can be replaced by function in fitting_helpers
def exponential_decay_fit(
    x: ArrayLike,
    data: ArrayLike,
    param_hints: dict | None = None,
) -> lmfit.model.ModelResult:
    """Performs a fit of an exponential-decay model to data.

    Arguments:
        data: the data to be fitted
        x: the independent variable
        param_hints: dictionary of guesses for the fit parameters. See the lmfit
            docstring for details on the form of the parameter hints dictionary:
            https://lmfit.github.io/lmfit-py/model.html#lmfit.model.Model.set_param_hint

    Returns:
        The lmfit result
    """
    if not param_hints:
        param_hints = {
            "decay_rate": {"value": 2 / (3 * np.max(x))},
            "amplitude": {
                "value": abs(np.max(data) - np.min(data)) / 2,
                "min": 0,
            },
            "offset": {"value": 0, "vary": False},
        }

    return fit_data_lmfit(
        fit_mods.exponential_decay,
        x,
        data,
        param_hints=param_hints,
    )
