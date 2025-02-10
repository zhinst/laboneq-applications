# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the analysis for Ramsey experiment.

The experiment is defined in laboneq_applications.experiments.

In this analysis, we first interpret the raw data into qubit population using
principle-component analysis or rotation and projection on the measured calibration
states. Then we fit an exponentially decaying cosine model to the qubit population and
extract the frequency and the qubit T2_star time from the fit. Then we calculate the
new qubit frequency from the old value and the frequency of the oscillations extracted
from the fit. Finally, we plot the data and the fit.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from laboneq import workflow

from laboneq_applications.analysis.calibration_traces_rotation import (
    calculate_qubit_population,
)
from laboneq_applications.analysis.fitting_helpers import cosine_oscillatory_decay_fit
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
class FitDataRamseyOptions:
    """Options for the `fit_data` task of the Ramsey analysis.

    See [FitDataOptions] for additional accepted options.

    Attributes:
        do_pca:
            Whether to perform principal component analysis on the raw data independent
            of whether there were calibration traces in the experiment.
            Default: `False`.
        transition:
            Transition to perform the experiment on. May be any
            transition supported by the quantum operations.
            Default: `"ge"` (i.e. ground to first excited state).
    """

    do_pca: bool = workflow.option_field(
        False,
        description="Whether to perform principal component analysis on the raw data"
        " independent of whether there were calibration traces in the experiment.",
    )
    transition: Literal["ge", "ef"] = workflow.option_field(
        "ge",
        description="Transition to perform the experiment on. May be any"
        " transition supported by the quantum operations.",
    )


def validate_and_convert_detunings(
    qubits: QuantumElements,
    detunings: float | Sequence[float] | None = None,
) -> Sequence[float]:
    """Validate the detunings used in a Ramsey experiment, and convert them to iterable.

    Check for the following conditions:
        - qubits must be a sequence.
        - detunings must be a sequence
        - detunings must have the same length as qubits
    If detunings is None, it is instantiated to a list of zeros with the same length
    as qubits.

    Args:
        qubits: the qubits used in the experiment/analysis
        detunings:
            The detuning in Hz introduced in order to generate oscillations of the qubit
            state vector around the Bloch sphere. This detuning and the frequency of the
            fitted oscillations is used to calculate the true qubit resonance frequency.
            `detunings` is a list of float values for each qubit following the order
            in qubits.

    Returns:
        a list containing the validated detunings
    """
    if not isinstance(qubits, Sequence):
        qubits = [qubits]

    if detunings is None:
        detunings = [0] * len(qubits)

    if not isinstance(detunings, Sequence):
        detunings = [detunings]

    if len(detunings) != len(qubits):
        raise ValueError(
            f"Length of qubits and detunings must be the same but "
            f"currently len(qubits) = {len(qubits)} and "
            f"len(detunings) = {len(detunings)}."
        )

    return detunings


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    delays: QubitSweepPoints,
    detunings: float | Sequence[float] | None = None,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The Ramsey analysis Workflow.

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
        delays:
            The delays that were swept over in the Ramsey experiment for
            each qubit. If `qubits` is a single qubit, `delays` must be an array of
            numbers. Otherwise, it must be a list of arrays of numbers.
        detunings:
            The detuning in Hz introduced in order to generate oscillations of the qubit
            state vector around the Bloch sphere. This detuning and the frequency of the
            fitted oscillations is used to calculate the true qubit resonance frequency.
            `detunings` is a list of float values for each qubit following the order
            in qubits.
        options:
            The options for building the workflow, passed as an instance of
            [TuneUpAnalysisWorkflowOptions]. See the docstring of this class for
            more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        result = analysis_workflow(
            results=results
            qubits=[q0, q1],
            delays=[
                np.linspace(0, 20e-6, 51),
                np.linspace(0, 30e-6, 52),
            ],
            detunings = [1e6, 1.346e6],
            options=analysis_workflow.options()
        ).run()
        ```
    """
    processed_data_dict = calculate_qubit_population(qubits, result, delays)
    fit_results = fit_data(qubits, processed_data_dict)
    qubit_parameters = extract_qubit_parameters(qubits, fit_results, detunings)
    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_raw_data_plotting):
            plot_raw_complex_data_1d(
                qubits,
                result,
                delays,
                xlabel="Pulse Separation, $\\tau$ ($\\mu$s)",
                xscaling=1e6,
            )
        with workflow.if_(options.do_qubit_population_plotting):
            plot_population(
                qubits, processed_data_dict, fit_results, qubit_parameters, detunings
            )
    workflow.return_(qubit_parameters)


@workflow.task
def fit_data(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: FitDataRamseyOptions | None = None,
) -> dict[str, lmfit.model.ModelResult]:
    """Perform a fit of an exponentially decaying cosine model to the data.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
        processed_data_dict: the processed data dictionary returned by process_raw_data
        options:
            The options class for this task as an instance of [FitDataRamseyOptions].
            See the docstring of this class for accepted options.

    Returns:
        dict with qubit UIDs as keys and the fit results for each qubit as values.
    """
    opts = FitDataRamseyOptions() if options is None else options
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
            "amplitude": {"value": 0.5, "vary": opts.do_pca},
            "oscillation_offset": {"value": 0, "vary": "f" in opts.transition},
        }
        param_hints_user = opts.fit_parameters_hints
        if param_hints_user is None:
            param_hints_user = {}
        param_hints.update(param_hints_user)
        try:
            fit_res = cosine_oscillatory_decay_fit(
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
    detunings: dict[str, float] | None = None,
    options: ExtractQubitParametersTransitionOptions | None = None,
) -> dict[str, dict[str, dict[str, int | float | unc.core.Variable | None]]]:
    """Extract the qubit parameters from the fit results.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits.
        fit_results: the fit-results dictionary returned by fit_data
        detunings:
            The detuning in Hz introduced in order to generate oscillations of the qubit
            state vector around the Bloch sphere. This detuning and the frequency of the
            fitted oscillations is used to calculate the true qubit resonance frequency.
            `detunings` is a list of float values for each qubit following the order
            in qubits.
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
    detunings = validate_and_convert_detunings(qubits, detunings)
    qubit_parameters = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
    }

    for i, q in enumerate(qubits):
        # Store the qubit frequency and T2_star values
        old_qb_freq = (
            q.parameters.resonance_frequency_ef
            if "f" in opts.transition
            else q.parameters.resonance_frequency_ge
        )
        old_t2_star = (
            q.parameters.ef_T2_star
            if "f" in opts.transition
            else q.parameters.ge_T2_star
        )
        qubit_parameters["old_parameter_values"][q.uid] = {
            f"resonance_frequency_{opts.transition}": old_qb_freq,
            f"{opts.transition}_T2_star": old_t2_star,
        }

        if opts.do_fitting and q.uid in fit_results:
            # Extract and store the new pi and pi-half pulse amplitude values
            fit_res = fit_results[q.uid]
            freq_fit = unc.ufloat(
                fit_res.params["frequency"].value,
                fit_res.params["frequency"].stderr,
            )
            introduced_detuning = detunings[i]
            qb_freq = old_qb_freq + introduced_detuning - freq_fit
            t2_star = unc.ufloat(
                fit_res.params["decay_time"].value,
                fit_res.params["decay_time"].stderr,
            )

            qubit_parameters["new_parameter_values"][q.uid] = {
                f"resonance_frequency_{opts.transition}": qb_freq,
                f"{opts.transition}_T2_star": t2_star,
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
    detunings: dict[str, float] | None = None,
    options: PlotPopulationOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the Ramsey plots.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
        processed_data_dict: the processed data dictionary returned by process_raw_data
        fit_results: the fit-results dictionary returned by fit_data
        qubit_parameters: the qubit-parameters dictionary returned by
            extract_qubit_parameters
        detunings:
            The detuning in Hz introduced in order to generate oscillations of the qubit
            state vector around the Bloch sphere. This detuning and the frequency of the
            fitted oscillations is used to calculate the true qubit resonance frequency.
            `detunings` is a list of float values for each qubit following the order
            in qubits.
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
    detunings = validate_and_convert_detunings(qubits, detunings)
    figures = {}
    for i, q in enumerate(qubits):
        sweep_points = processed_data_dict[q.uid]["sweep_points"]
        data = processed_data_dict[q.uid][
            "population" if opts.do_rotation else "data_raw"
        ]
        num_cal_traces = processed_data_dict[q.uid]["num_cal_traces"]

        fig, ax = plt.subplots()
        ax.set_title(timestamped_title(f"Ramsey {q.uid}"))
        ax.set_xlabel("Pulse Separation, $\\tau$ ($\\mu$s)")
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
            # plot fit
            swpts_fine = np.linspace(sweep_points[0], sweep_points[-1], 501)
            ax.plot(
                swpts_fine * 1e6,
                fit_res_qb.model.func(swpts_fine, **fit_res_qb.best_values),
                "r-",
                zorder=1,
                label="fit",
            )

            if (
                qubit_parameters is not None
                and len(qubit_parameters["new_parameter_values"][q.uid]) > 0
            ):
                # textbox
                old_qb_freq = qubit_parameters["old_parameter_values"][q.uid][
                    f"resonance_frequency_{opts.transition}"
                ]
                old_t2_star = qubit_parameters["old_parameter_values"][q.uid][
                    f"{opts.transition}_T2_star"
                ]
                new_qb_freq = qubit_parameters["new_parameter_values"][q.uid][
                    f"resonance_frequency_{opts.transition}"
                ]
                new_t2_star = qubit_parameters["new_parameter_values"][q.uid][
                    f"{opts.transition}_T2_star"
                ]
                freq_fit = fit_res_qb.best_values["frequency"]
                freq_fit_err = fit_res_qb.params["frequency"].stderr
                introduced_detuning = detunings[i]
                textstr = (
                    f"New qubit frequency: {new_qb_freq.nominal_value / 1e9:.6f} GHz "
                    f"$\\pm$ {new_qb_freq.std_dev / 1e6:.4f} MHz"
                )
                textstr += f"\nOld qubit frequency: {old_qb_freq / 1e9:.6f} GHz"
                textstr += (
                    f"\nDiff new-old qubit frequency: "
                    f"{(new_qb_freq - old_qb_freq) / 1e6:.6f} MHz"
                )
                textstr += f"\nIntroduced detuning: {introduced_detuning / 1e6:.2f} MHz"
                textstr += (
                    f"\nFitted frequency: {freq_fit / 1e6:.6f} "
                    f"$\\pm$ {freq_fit_err / 1e6:.4f} MHz"
                )
                textstr += (
                    f"\n$T_2^*$: {new_t2_star.nominal_value * 1e6:.4f} $\\mu$s $\\pm$ "
                    f"{new_t2_star.std_dev * 1e6:.4f} $\\mu$s"
                )
                textstr += f"\nOld $T_2^*$: {old_t2_star * 1e6:.4f} $\\mu$s"
                ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            handlelength=1.5,
            frameon=False,
        )

        if opts.save_figures:
            workflow.save_artifact(f"Ramsey_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures
