# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

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
from laboneq import workflow
from laboneq.simple import dsl

from laboneq_applications.analysis.fitting_helpers import lorentzian_fit
from laboneq_applications.analysis.options import (
    BasePlottingOptions,
    DoFittingOption,
)
from laboneq_applications.analysis.plotting_helpers import (
    plot_raw_complex_data_1d,
    timestamped_title,
)
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_result,
)

if TYPE_CHECKING:
    from typing import Any, Literal

    import attr
    import lmfit
    import matplotlib as mpl
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.workflow_options
class QubitSpectroscopyAnalysisWorkflowOptions:
    """Option class for qubit spectroscopy analysis workflows.

    Attributes:
        do_plotting:
            Whether to make plots.
            Default: 'True'.
        do_raw_data_plotting:
            Whether to plot the raw data.
            Default: True.
        do_plotting_qubit_spectroscopy:
            Whether to plot the final qubit spectroscopy plot.
            Default: True.
    """

    do_plotting: bool = workflow.option_field(
        True, description="Whether to make plots."
    )
    do_raw_data_plotting: bool = workflow.option_field(
        True, description="Whether to plot the raw data."
    )
    do_plotting_qubit_spectroscopy: bool = workflow.option_field(
        True, description="Whether to plot the final qubit spectroscopy plot."
    )


def _frequency_filters_validator(
    inst: Any,  # noqa: ANN401
    attr: attr.Attribute,
    value: dict[str, tuple[float | None]] | None,
) -> None:
    if value is None:
        return

    if not isinstance(value, dict):
        raise TypeError("frequency_filters must be a dictionary.")

    for q_uid, freq_filter in value.items():
        if freq_filter is not None and len(freq_filter) != 2:  # noqa: PLR2004
            raise ValueError(
                f"The frequency filter must have two entries, but that is not the "
                f"case for qubit {q_uid}: {freq_filter}."
            )


def _spectral_feature_validator(
    inst: Any,  # noqa: ANN401
    attr: attr.Attribute,
    value: Literal["peak", "dip"] | None,
) -> None:
    if value is None:
        return

    if value not in ["peak", "dip", "auto"]:
        raise ValueError(
            "Invalid spectral_feature. Please choose 'auto', 'peak', or 'dip'."
        )


@workflow.task_options(base_class=DoFittingOption)
class FitDataQubitSpecOptions:
    """Options for the `fit_data` task of the qubit spectroscopy analysis.

    Attributes:
        frequency_filters:
            Information on how to filter the first-dimensional sweep points
            (the frequency) for each qubit before performing the Lorentzian fit;
            for example, to fit the data only in the range f < 6.8 GHz.
            See the description in the options field for more details.
        spectral_feature:
            Whether to perform the fit assuming the Lorentzian is pointing
            upwards ("peak") or downwards ("dip"). By default, this parameter is "auto",
            in which case, the `lorentzian_fit` routine in `fitting_helpers.py` tries
            to work out the orientation of the Lorentzian feature.
            Default: None
        fit_parameters_hints:
            Parameters hints accepted by lmfit
            Default: None.

    Additional attributes from `DoFittingOption`:
        do_fitting:
            Whether to perform the fit.
            Default: `True`.
    """

    frequency_filters: dict[str, tuple[float | None]] | None = workflow.option_field(
        None,
        description="Information on how to filter the first-dimensional sweep points "
        "(the frequency) for each qubit before performing the Lorentzian fit; for "
        "example, to fit the data only in the range f < 6.8 GHz. The frequency_filters "
        "option field is either None (in which case no filter is applied), or a "
        "dictionary with qubit UIDs as keys and the corresponding filtering "
        "information as values. The latter is specified as a tuple with two entries: "
        "(None | lower limit, None | upper limit). The filter is applied as, "
        "frequencies > lower limit, frequencies < upper limit. Set `None` for either "
        "the upper or the lower limit to remove them from the filter.",
        validators=[_frequency_filters_validator],
    )
    spectral_feature: Literal["peak", "dip", "auto"] = workflow.option_field(
        "auto",
        description="Whether to perform the fit assuming the Lorentzian is pointing "
        "upwards ('peak') or downwards ('dip'). By default, this parameter is 'auto', "
        "in which case, the `lorentzian_fit` routine in `fitting_helpers.py` tries to "
        "work out the orientation of the Lorentzian feature.",
        validators=[_spectral_feature_validator],
    )
    fit_parameters_hints: dict | None = workflow.option_field(
        None, description="Parameters hints accepted by lmfit."
    )


@workflow.task_options
class PlotQubitSpectroscopyOptions(DoFittingOption, BasePlottingOptions):
    """Options for the `plot_qubit_spectroscopy` task of the qubit spec. analysis.

    Attributes from `DoFittingOption`:
        do_fitting:
            Whether to perform the fit.
            Default: `True`.

    Attributes from `BasePlottingOptions`:
        save_figures:
            Whether to save the figures.
            Default: `True`.
        close_figures:
            Whether to close the figures.
            Default: `True`.
    """


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
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
    qubits: QuantumElements,
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
        raw_data = result[dsl.handles.result_handle(q.uid)].data
        proc_data_dict[q.uid] = {
            "sweep_points": freqs,
            "data_raw": raw_data,
            "magnitude": np.abs(raw_data),
            "phase": np.angle(raw_data),
        }

    return proc_data_dict


def _get_data_to_fit(
    qubit: QuantumElement,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    frequency_filters: dict[str, tuple[float | None]],
) -> tuple[ArrayLike, ArrayLike]:
    """Extracts the data for the fit and processes it based on frequency_filters.

    Args:
        qubit:
            The qubit for which to filter the frequencies, which must be the ones that
            were swept for this qubit.
            If filtering of the frequencies for this qubit is desired, then the qubit
            uid must exist in frequency_filters.
        processed_data_dict: the processed data dictionary returned by
            calculate_signal_magnitude_and_phase
        frequency_filters:
            A dict with qubit uids as keys and the filtering information as values.
            The filtering information is passed as a tuple with two entries:
            (None | lower limit, None | upper limit). The filter is applied as,
            frequencies > lower limit, frequencies < upper limit. Set `None` for either
            the upper or the lower limit to remove them from the filter.

    Returns:
        The arrays of independent variable (frequencies) and dependent variable (signal
        magnitude) for the Lorentzian fit.

        If the qubit uid is found in frequency_filters and the filtering information for
        this qubit is not None, then the frequencies and signal magnitude array are
        filtered based on the information in frequency_filters[qubit.uid].
    """
    if frequency_filters is None:
        frequency_filters = {}

    if not isinstance(frequency_filters, dict):
        raise TypeError("frequency_filters must be a dictionary.")

    frequencies = processed_data_dict[qubit.uid]["sweep_points"]
    magnitude = processed_data_dict[qubit.uid]["magnitude"]

    if qubit.uid in frequency_filters and frequency_filters.get(qubit.uid) is not None:
        freq_filter = list(frequency_filters[qubit.uid])
        if freq_filter[0] is None:
            freq_filter[0] = min(frequencies)
        if freq_filter[1] is None:
            freq_filter[1] = max(frequencies)
        if freq_filter[0] > freq_filter[1]:
            raise ValueError(
                f"The first entry in the frequency filter cannot be larger than the "
                f"second entry, but this is so for qubit {qubit.uid}: {freq_filter}."
            )
        mask = np.logical_and(
            frequencies > freq_filter[0]
            if freq_filter[0] is not None
            else frequencies >= min(frequencies),
            frequencies < freq_filter[1]
            if freq_filter[1] is not None
            else frequencies <= max(frequencies),
        )
    else:
        mask = np.ones_like(frequencies, dtype=bool)

    return frequencies[mask], magnitude[mask]


@workflow.task
def fit_data(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: FitDataQubitSpecOptions | None = None,
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
            The options class for this task as an instance of [FitDataQubitSpecOptions].
            See the docstring of this class for accepted options.

    Returns:
        dict with qubit UIDs as keys and the fit results for each qubit as keys.
    """
    opts = FitDataQubitSpecOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    fit_results = {}
    if not opts.do_fitting:
        return fit_results

    for q in qubits:
        swpts_fit, data_to_fit = _get_data_to_fit(
            q, processed_data_dict, opts.frequency_filters
        )
        try:
            fit_res = lorentzian_fit(
                swpts_fit,
                data_to_fit,
                spectral_feature=opts.spectral_feature,
                param_hints=opts.fit_parameters_hints,
            )
            fit_results[q.uid] = fit_res
        except ValueError as err:
            workflow.log(logging.ERROR, "Fit failed for %s: %s.", q.uid, err)

    return fit_results


@workflow.task
def extract_qubit_parameters(
    qubits: QuantumElements,
    fit_results: dict[str, lmfit.model.ModelResult] | None,
    options: DoFittingOption | None = None,
) -> dict[str, dict[str, dict[str, int | float | unc.core.Variable | None]]]:
    """Extract the qubit parameters from the fit results.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
        fit_results: the fit-results dictionary returned by fit_data
        options:
            The options for this task as an instance of [DoFittingOption].
            See the docstring of this class for more details.

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
    opts = DoFittingOption() if options is None else options
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
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, lmfit.model.ModelResult] | None,
    qubit_parameters: dict[
        str, dict[str, dict[str, int | float | unc.core.Variable | None]]
    ],
    options: PlotQubitSpectroscopyOptions | None = None,
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
            The options for this task as an instance of [PlotQubitSpectroscopyOptions].
            See the docstring of this class for more details.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = PlotQubitSpectroscopyOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}

    for q in qubits:
        sweep_points = processed_data_dict[q.uid]["sweep_points"]
        magnitude = processed_data_dict[q.uid]["magnitude"]

        fig, ax = plt.subplots()
        ax.set_title(timestamped_title(f"Qubit Spectroscopy {q.uid}"))
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
            swpts_fit = fit_res.userkws["x"]

            # Plot fit of the magnitude
            swpts_fine = np.linspace(swpts_fit[0], swpts_fit[-1], 501)
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
