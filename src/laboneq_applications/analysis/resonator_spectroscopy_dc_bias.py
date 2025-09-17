# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the analysis for a resonator-spectroscopy dc-bias experiment.

In this experiment, the readout resonator frequency is swept in real time, and the
voltage of an external DC voltage source is swept in near-time using a callback
function. The resulting data is two-dimensional.

In this analysis, we first interpret the raw data into the signal magnitude and phase,
and extract the readout-resonator frequencies as the highest peaks
(`options.find_peaks == True`) or the deepest dips (`options.find_peaks == False`)
in the signal magnitude. Optionally, we apply first apply a filter
(`options.frequency_filter`) to only extract the readout-resonator frequencies in a
certain range.

Then, we fit a cosine model to the readout-resonator frequencies vs the corresponding
voltage points, and extract all the possible upper and lower sweet spots from this
curve. The sweet-spots are defined as the points on the cosine curve of voltage vs
readout-resonator frequency where the frequency is second-order sensitive to the
voltage (i.e. where cos(x) = +/-1). The upper (lower) sweet-spots are the points on the
cosine curve of voltage vs readout-resonator frequency where cos(x) = -1 (+1).

We then extract the qubit parameters to be updated, corresponding either to the upper
or lower sweet spot, depending on `options.parking_sweet_spot`.

Finally, we produce three plots: the raw data, the signal magnitude with the fit
results, and the signal phase.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from laboneq import workflow
from laboneq.simple import dsl

from laboneq_applications.analysis.fitting_helpers import (
    cosine_oscillatory_fit,
    get_pi_pi2_xvalues_on_cos,
)
from laboneq_applications.analysis.options import BasePlottingOptions
from laboneq_applications.analysis.plotting_helpers import (
    plot_data_2d,
    plot_raw_complex_data_2d,
)
from laboneq_applications.core import validation

if TYPE_CHECKING:
    import attr
    import lmfit
    import matplotlib as mpl
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike


@workflow.workflow_options
class ResonatorSpectroscopyDCBiasAnalysisWorkflowOptions:
    """Option class for resonator-spectroscopy-dc-bias analysis workflow.

    Attributes:
        do_plotting:
            Whether to create plots.
            Default: 'True'.
        do_raw_data_plotting:
            Whether to plot the raw data.
            Default: True.
        do_plotting_magnitude:
            Whether to make the plot of the signal magnitude, the extracted "
            "readout-resonator peak/dips for each voltage value, and the fit "
            "results if a fit was performed.
            Default: True.
        do_plotting_phase:
            Whether to make the plot the phase of the signal.
            Default: True.
    """

    do_plotting: bool = workflow.option_field(
        True, description="Whether to create plots."
    )
    do_raw_data_plotting: bool = workflow.option_field(
        True, description="Whether to plot the raw data."
    )
    do_plotting_magnitude: bool = workflow.option_field(
        True,
        description="Whether to make the plot of the signal magnitude, the extracted "
        "readout-resonator peak/dips for each voltage value, and the fit "
        "results if a fit was performed.",
    )
    do_plotting_phase: bool = workflow.option_field(
        True, description="Whether to make the plot the phase of the signal."
    )


def _frequency_filter_validator(
    inst: None, attr: attr.Attribute, value: tuple[float | None] | None
) -> None:
    if value is None:
        return

    if not len(value) == 2:  # noqa: PLR2004
        raise ValueError("frequency_filter must have two entries.")


@workflow.task_options
class ProcessRawDataOptions:
    """Options for the `process_raw_data` task of the resonator spectroscopy analysis.

    Attributes:
        find_peaks:
            Whether to extract the readout-resonator frequencies from the highest peaks
            in the signal magnitude (True) or from the lowest dips in the signal
            magnitude (False).
            Default: `False`.
        frequency_filter:
            Information on how to filter the first-dimensional sweep points
            (the frequency) before extracting the readout-resonator frequencies;
            for example to extract the readout-resonator frequencies only in the range
            f > 6.8 GHz.
            The frequency_filter is either None (in which case no filter is applied),
            or a tuple with two entries: (None | lower limit, None | upper limit). The
            filter is applied as, frequencies > lower limit, frequencies < upper limit.
            Set `None` for either the upper or the lower limit to remove them from the
            filter.
            Default: None.
    """

    find_peaks: bool = workflow.option_field(
        False,
        description="Whether to extract the readout-resonator frequencies from the "
        "highest peaks in the signal magnitude (True) or from the lowest dips in the "
        "signal magnitude (False).",
    )
    frequency_filter: tuple[float | None] | None = workflow.option_field(
        None,
        description="Information on how to filter the first-dimensional sweep points "
        "(the frequency) before extracting the readout-resonator frequencies; "
        "for example to extract the readout-resonator frequencies only in the range "
        "f > 6.8 GHz. The frequency_filter is either None (in which case no filter is "
        "applied), or a tuple with two entries: "
        "(None | lower limit, None | upper limit). The filter is applied as, "
        "frequencies > lower limit, frequencies < upper limit. Set `None` for either "
        "the upper or the lower limit to remove them from the filter.",
        validators=[_frequency_filter_validator],
    )


@workflow.task_options
class FitDataOptions:
    """Options for the `fit_data` task of the resonator spectroscopy analysis.

    Attributes:
        do_fitting:
            Whether to perform the fit of the voltages vs. readout-resonator frequencies
            Default: `False`.
        fit_parameters_hints:
            Parameters hints accepted by lmfit
            Default: None.
    """

    do_fitting: bool = workflow.option_field(
        True,
        description="Whether to perform the fit of the voltages vs. readout-resonator "
        "frequencies.",
    )
    fit_parameters_hints: dict[str, dict[str, float | bool | str]] | None = (
        workflow.option_field(None, description="Parameters hints accepted by lmfit")
    )


@workflow.task_options
class ExtractQubitParametersOptions:
    """Options for the `extract_qubit_parameters` task in this module.

    Attributes:
        parking_sweet_spot:
            Specifies which sweet spot to choose for the parking, either `uss` for
            'lower sweep-spot' or `uss` for 'upper sweep-spot'
            Default: `False`.
    """

    parking_sweet_spot: Literal["uss", "lss"] = workflow.option_field(
        default="uss",
        description="Specifies which sweet spot to choose for the qubit parking, "
        "either `uss` for 'lower sweep-spot' or `uss` for 'upper sweep-spot'. The "
        "value of this option determines which parking parameters extracted from the "
        "fit will be used to update the qubit parameter if `update==True` in the "
        "experiment-workflow options.",
    )


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubit: QuantumElement,
    frequencies: ArrayLike,
    voltages: ArrayLike,
    options: ResonatorSpectroscopyDCBiasAnalysisWorkflowOptions | None = None,
) -> None:
    """The analysis Workflow for a resonator-spectroscopy-dc-bias experiment.

    The workflow consists of the following steps:

    - [process_raw_data]()
    - [fit_data]()
    - [process_fit_results]()
    - [extract_qubit_parameters]()
    - [plot_raw_complex_data_1d]()
    - [plot_signal_magnitude]()
    - [plot_signal_phase]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubit:
            The qubit on which to run the analysis. The UID of this qubit must exist
            in the result.
        frequencies:
            The array of frequencies that were swept over in the experiment.
        voltages:
            The array of DC bias voltages that were swept over in the experiment.
        options:
            The options for building the workflow, passed as an instance of
            [ResonatorSpectroscopyDCBiasAnalysisWorkflowOptions]. See the docstring
            of [ResonatorSpectroscopyDCBiasAnalysisWorkflowOptions] for more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        result = analysis_workflow(
            results=results
            qubit=q0,
            frequencies=np.linspace(7.0, 7.1, 101),
            voltages=np.linspace(0, 1, 11),
            options=analysis_workflow.options(),
        ).run()
        ```
    """
    processed_data_dict = process_raw_data(qubit, result, frequencies, voltages)
    fit_result = fit_data(processed_data_dict)
    processed_fit_results = process_fit_results(fit_result)
    qubit_parameters = extract_qubit_parameters(
        qubit, processed_data_dict, processed_fit_results
    )
    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_raw_data_plotting):
            plot_raw_complex_data_2d(
                qubits=qubit,
                result=result,
                sweep_points_1d=frequencies,
                sweep_points_2d=voltages,
                label_sweep_points_1d="Readout Frequency, $f_{\\mathrm{RO}}$ (GHz)",
                label_sweep_points_2d="DC Bias Voltage $V$ (V)",
                scaling_sweep_points_1d=1e-9,
            )
        with workflow.if_(options.do_plotting_magnitude):
            plot_signal_magnitude(
                qubit=qubit,
                processed_data_dict=processed_data_dict,
                fit_result=fit_result,
                processed_fit_results=processed_fit_results,
            )
        with workflow.if_(options.do_plotting_phase):
            plot_signal_phase(
                qubit=qubit,
                processed_data_dict=processed_data_dict,
            )
    workflow.return_(qubit_parameters)


@workflow.task
def process_raw_data(
    qubit: QuantumElement,
    result: RunExperimentResults,
    frequencies: ArrayLike,
    voltages: ArrayLike,
    options: ProcessRawDataOptions | None = None,
) -> dict[str, ArrayLike]:
    """Processes the raw data and extracts the arrays needed for the analysis.

    This task calculates the following quantities based on the raw data and the options:

     - the magnitude of the transmission signal;
     - the phase of the transmission signal;
     - the filtered frequencies that were swept over in the experiment, calculated by
     applying `opts.frequency_filter` on the first-dimension sweep points;
     - the readout-resonator frequencies (`rr_frequencies`) as the highest peaks
     (`opts.find_peaks == True`) or the deepest dips (`opts.find_peaks == False`) in the
     signal magnitude.
        The length of the `rr_frequencies` array will be equal to the number of voltages
        (second-dimension sweep points) that were swept in the experiment

    Arguments:
        result:
            The experiment results returned by the  run_experiment` task.
        qubit:
            The qubit on which to run this task. The UID of this qubit must exist
            in the result.
        frequencies:
            The array of frequencies that were swept over in the experiment.
        voltages:
            The array of DC bias voltages that were swept over in the experiment.
        options:
            The options for this task as an instance of [ProcessRawDataOptions].
            See the docstring of this class for more details.

    Returns:
        dictionary with the following data:
            sweep_points_1d
            sweep_points_1d_filtered
            sweep_points_2d
            rr_frequencies
            data_raw
            magnitude
            phase
    """
    opts = ProcessRawDataOptions() if options is None else options
    validation.validate_result(result)
    _, frequencies = validation.validate_and_convert_single_qubit_sweeps(
        qubit, frequencies
    )
    qubit, voltages = validation.validate_and_convert_single_qubit_sweeps(
        qubit, voltages
    )

    raw_data = result[dsl.handles.result_handle(qubit.uid)].data
    magnitude = np.abs(raw_data)
    phase = np.angle(raw_data)
    take_extremum_function = np.argmax if opts.find_peaks else np.argmin
    if opts.frequency_filter is None:
        frequencies_filtered = None
        # find the resonator resonances by taking the extremum of the spectrum for each
        # voltage value
        rr_frequencies = frequencies[take_extremum_function(magnitude, axis=1)]
    else:
        freq_filter = list(opts.frequency_filter)
        if freq_filter[0] is None:
            freq_filter[0] = min(frequencies)
        if freq_filter[1] is None:
            freq_filter[1] = max(frequencies)
        if freq_filter[0] > freq_filter[1]:
            raise ValueError(
                "The first entry in the frequency_filter cannot be larger than the "
                "second entry."
            )
        mask = np.logical_and(
            frequencies > freq_filter[0]
            if freq_filter[0] is not None
            else frequencies >= min(frequencies),
            frequencies < freq_filter[1]
            if freq_filter[1] is not None
            else frequencies <= max(frequencies),
        )
        frequencies_filtered = frequencies[mask]
        # find the resonator resonances by taking the extremum of the spectrum for each
        # voltage value
        rr_frequencies = frequencies_filtered[
            take_extremum_function(magnitude[:, mask], axis=1)
        ]

    return {
        "sweep_points_1d": frequencies,
        "sweep_points_1d_filtered": frequencies_filtered,
        "sweep_points_2d": voltages,
        "rr_frequencies": rr_frequencies,
        "data_raw": raw_data,
        "magnitude": magnitude,
        "phase": phase,
    }


@workflow.task
def fit_data(
    processed_data_dict: dict[str, ArrayLike],
    options: FitDataOptions | None = None,
) -> lmfit.model.ModelResult | None:
    """Perform a fit of a cosine model to the voltages vs readout-resonator frequencies.

    Arguments:
        processed_data_dict: the processed data dictionary returned by
            the `process_raw_data` task in this module.
        options:
            The options for this task as an instance of [FitDataOptions].
            See the docstring of this class for more details.

    Returns:
        The fit result as an instance of `lmfit.model.ModelResult`.
    """
    opts = FitDataOptions() if options is None else options
    fit_result = None
    if not opts.do_fitting:
        return fit_result

    rr_frequencies = processed_data_dict["rr_frequencies"]
    voltages = processed_data_dict["sweep_points_2d"]
    try:
        fit_result = cosine_oscillatory_fit(
            x=voltages,
            data=rr_frequencies,
            param_hints=opts.fit_parameters_hints,
        )
    except ValueError as err:
        workflow.log(
            logging.ERROR,
            "Fit failed. Choosing the sweet-spot as the maximum extracted "
            "readout-resonator frequency if the voltage vs readout-resonator "
            "frequencies curve is convex, or the minimum if the curve is concave. %s",
            err,
        )

    return fit_result


@workflow.task
def process_fit_results(
    fit_result: lmfit.model.ModelResult | None,
) -> dict[str, dict[str, ArrayLike | float]]:
    """Extract information about the upper and lower sweet-spots from the fit result.

    This task calculates all the possible upper sweet-spot (uss) and lower sweet-spot
    (uss) points on the cosine curve, and chooses the upper and lower sweet-spot parking
    voltages as the lowest out of all the possibilities.

    The sweet-spots are defined as the points on the cosine curve of voltage vs
    readout-resonator frequency where the frequency is second-order sensitive to the
    voltage (i.e. where cos(x) = +/-1).
    The upper (lower) sweet-spots are the points on the cosine curve of voltage vs
    readout-resonator frequency where cos(x) = -1 (+1).

    Args:
        fit_result: The fit results returned by `fit_data` in this module.

    Returns:
        A dictionary with the following data:

        uss
            voltages
            frequencies
            rr_frequency_parking
            dc_voltage_parking
        lss
            voltages
            frequencies
            rr_frequency_parking
            dc_voltage_parking
    """
    processed_fit_results = {}
    if fit_result is None:
        return processed_fit_results

    freq_fit = unc.ufloat(
        fit_result.params["frequency"].value,
        fit_result.params["frequency"].stderr,
    )
    phase_fit = unc.ufloat(
        fit_result.params["phase"].value,
        fit_result.params["phase"].stderr,
    )
    voltages_fit = fit_result.userkws["x"]
    # extract all the possible upper and lower sweep-spot values on the cosine fit curve
    voltages_uss, voltages_lss, _, _ = get_pi_pi2_xvalues_on_cos(
        voltages_fit, freq_fit, phase_fit
    )
    v_uss_values = np.array([v_uss.nominal_value for v_uss in voltages_uss])
    v_lss_values = np.array([v_lss.nominal_value for v_lss in voltages_lss])
    f_uss_values = fit_result.model.func(v_uss_values, **fit_result.best_values)
    f_lss_values = fit_result.model.func(v_lss_values, **fit_result.best_values)

    if len(v_uss_values) > 0:
        # choose the lowest voltage value as the upper-sweet spot parking voltage
        uss_idx = np.argsort(abs(v_uss_values))[0]
        v_uss, f_uss = voltages_uss[uss_idx], f_uss_values[uss_idx]
        processed_fit_results["uss"] = {
            "voltages": list(voltages_uss),
            "frequencies": f_uss_values,
            "rr_frequency_parking": f_uss,
            "dc_voltage_parking": v_uss,
        }

    if len(v_lss_values) > 0:
        # choose the lowest voltage value as the lower-sweet spot parking voltage
        lss_idx = np.argsort(abs(v_lss_values))[0]
        v_lss, f_lss = voltages_lss[lss_idx], f_lss_values[lss_idx]
        processed_fit_results["lss"] = {
            "voltages": list(voltages_lss),
            "frequencies": f_lss_values,
            "rr_frequency_parking": f_lss,
            "dc_voltage_parking": v_lss,
        }

    return processed_fit_results


@workflow.task
def extract_qubit_parameters(
    qubit: QuantumElement,
    processed_data_dict: dict[str, ArrayLike],
    processed_fit_results: dict[str, dict[str, ArrayLike | float]],
    options: ExtractQubitParametersOptions | None = None,
) -> dict[str, dict[str, dict[str, int | float | unc.core.Variable | None]]]:
    """Extract the qubit parameters from the fit results.

    Arguments:
        qubit:
            The qubit on which to run this task.
        processed_data_dict: the processed-data dictionary returned by
            the `process_raw_data` task in this module.
        processed_fit_results: the dictionary returned by the `process_fit_results`
            task in this module.
        options:
            The options for this task as an instance of [ExtractQubitParametersOptions].
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
        If the fit_results is None, the new_parameter_values entry for the qubit is
        left empty.
    """
    opts = ExtractQubitParametersOptions() if options is None else options
    qubit = validation.validate_and_convert_single_qubit_sweeps(qubit)

    qubit_parameters = {
        "old_parameter_values": {qubit.uid: {}},
        "new_parameter_values": {qubit.uid: {}},
    }

    # Store the readout resonator frequency value
    qubit_parameters["old_parameter_values"][qubit.uid] = {
        "readout_resonator_frequency": qubit.parameters.readout_resonator_frequency,
        "dc_voltage_parking": qubit.parameters.dc_voltage_parking,
    }

    if len(processed_fit_results) > 0:  # fit failed or skipped
        # take the parking point from the fit
        if opts.parking_sweet_spot not in processed_fit_results:
            sweet_spot_name = (
                "upper sweet spot (uss)"
                if opts.parking_sweet_spot == "uss"
                else "lower sweet spot (lss)"
            )
            other_sweet_spot = "uss" if opts.parking_sweet_spot == "lss" else "lss"
            raise ValueError(
                f"The {sweet_spot_name} was not found in the chosen voltage sweep "
                f"range. Please set `options.parking_sweet_spot({other_sweet_spot})`."
            )
        proc_fit_dict = processed_fit_results[opts.parking_sweet_spot]
        rr_freq = proc_fit_dict["rr_frequency_parking"]
        v_parking = proc_fit_dict["dc_voltage_parking"].nominal_value
        qubit_parameters["new_parameter_values"][qubit.uid] = {
            "readout_resonator_frequency": rr_freq,
            "dc_voltage_parking": v_parking,
        }

    return qubit_parameters


@workflow.task
def plot_signal_magnitude(
    qubit: QuantumElement,
    processed_data_dict: dict[str, ArrayLike],
    fit_result: lmfit.model.ModelResult | None,
    processed_fit_results: dict[str, dict[str, ArrayLike | float]],
    options: BasePlottingOptions | None = None,
) -> mpl.figure.Figure | None:
    """Plot the magnitude of the signal and the fit results.

    Arguments:
        qubit:
            The qubit on which to run this task.
        fit_result: The fit results returned by `fit_data` in this module.
        processed_data_dict: the processed data dictionary returned by
            `process_raw_data` task in this module.
        processed_fit_results: the dictionary returned by the `process_fit_results`
            task in this module.
        options:
            The options for this task as an instance of [BasePlottingOptions].
            See the docstring of this class for more details.

    Returns:
        The matplotlib figure.

        If there are no new_parameter_values for the qubit, then fit result and the
        textbox with the extracted parking information are not plotted.
    """
    opts = BasePlottingOptions() if options is None else options
    qubit = validation.validate_and_convert_single_qubit_sweeps(qubit)

    frequencies = processed_data_dict["sweep_points_1d"]
    voltages = processed_data_dict["sweep_points_2d"]
    magnitude = processed_data_dict["magnitude"]

    # plot magnitude
    fig, ax = plot_data_2d(
        x_values=frequencies,
        y_values=voltages,
        z_values=magnitude,
        label_x_values="Readout Frequency, $f_{\\mathrm{RO}}$ (GHz)",
        label_y_values="DC Bias Voltage $V$ (V)",
        label_z_values="Transmission Signal\nMagnitude, $|S_{21}|$ (a.u.)",
        scaling_x_values=1e-9,
        close_figures=False,
    )

    if fit_result is not None:
        # add fit line and data points of voltages vs readout-resonator frequencies
        rr_frequencies = processed_data_dict["rr_frequencies"]
        voltages_fit = fit_result.userkws["x"]
        ax.plot(rr_frequencies / 1e9, voltages_fit, "ow")
        swpts_fine = np.linspace(voltages_fit[0], voltages_fit[-1], 501)
        ax.plot(
            fit_result.model.func(swpts_fine, **fit_result.best_values) / 1e9,
            swpts_fine,
            "w-",
        )

        if "uss" in processed_fit_results:
            f_uss_values = processed_fit_results["uss"]["frequencies"]
            v_uss_unc_floats = processed_fit_results["uss"]["voltages"]
            v_uss_values = np.array([v_uss.nominal_value for v_uss in v_uss_unc_floats])
            (line_uss,) = ax.plot(f_uss_values / 1e9, v_uss_values, "bo")

            v_uss = processed_fit_results["uss"]["dc_voltage_parking"]
            f_uss = processed_fit_results["uss"]["rr_frequency_parking"]
            textstr = "Smallest USS voltage:\n"
            textstr += f"{v_uss.nominal_value:.4f} V $\\pm$ {v_uss.std_dev:.4f} V"
            textstr += f"\nParking frequency:\n{f_uss / 1e9:.4f} GHz"
            ax.text(
                1,
                -0.15,
                textstr,
                ha="right",
                va="top",
                c=line_uss.get_c(),
                transform=ax.transAxes,
            )

        if "lss" in processed_fit_results:
            f_lss_values = processed_fit_results["lss"]["frequencies"]
            v_lss_unc_floats = processed_fit_results["lss"]["voltages"]
            v_lss_values = np.array([v_lss.nominal_value for v_lss in v_lss_unc_floats])
            (line_lss,) = ax.plot(f_lss_values / 1e9, v_lss_values, "go")

            v_lss = processed_fit_results["lss"]["dc_voltage_parking"]
            f_lss = processed_fit_results["lss"]["rr_frequency_parking"]
            textstr = "Smallest LSS voltage:\n"
            textstr += f"{v_lss.nominal_value:.4f} V $\\pm$ {v_lss.std_dev:.4f} V"
            textstr += f"\nParking frequency:\n{f_lss / 1e9:.4f} GHz"
            ax.text(
                0,
                -0.15,
                textstr,
                ha="left",
                va="top",
                c=line_lss.get_c(),
                transform=ax.transAxes,
            )

    if opts.save_figures:
        workflow.save_artifact(f"Magnitude_Phase_{qubit.uid}", fig)

    if opts.close_figures:
        plt.close(fig)

    return fig


@workflow.task
def plot_signal_phase(
    qubit: QuantumElement,
    processed_data_dict: dict[str, ArrayLike],
    options: BasePlottingOptions | None = None,
) -> mpl.figure.Figure | None:
    """Plot the phase of the transmission signal.

    Arguments:
        qubit:
            The qubit on which to run this task.
        processed_data_dict: the processed data dictionary returned by
            `process_raw_data` in this module.
        options:
            The options for this task as an instance of [BasePlottingOptions].
            See the docstring of this class for more details.

    Returns:
        the matplotlib figure

        If there are no new_parameter_values for the qubit, then fit result and the
        textbox with the extracted readout resonator frequency are not plotted.
    """
    opts = BasePlottingOptions() if options is None else options
    qubit = validation.validate_and_convert_single_qubit_sweeps(qubit)

    frequencies = processed_data_dict["sweep_points_1d"]
    voltages = processed_data_dict["sweep_points_2d"]
    phase = processed_data_dict["phase"]

    fig, axs = plot_data_2d(
        x_values=frequencies,
        y_values=voltages,
        z_values=phase,
        label_x_values="Readout Frequency, $f_{\\mathrm{RO}}$ (GHz)",
        label_y_values="DC Bias Voltage $V$ (V)",
        label_z_values="Transmission Signal\nPhase, $|S_{21}|$ (a.u.)",
        scaling_x_values=1e-9,
        close_figures=False,
    )

    if opts.save_figures:
        workflow.save_artifact(f"Phase_{qubit.uid}", fig)

    if opts.close_figures:
        plt.close(fig)

    return fig
