# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the analysis for a TWPA tuneup experiment.

The experiment is defined in laboneq_applications.twpa_experiments.

In this analysis, we plot the phase diagram of the TWPA,
sweeping pump frequency (x-axis) and pump power (y-axis) to obtain
the gain of readout signal from a QA channel output into a QA channel input.
The SHFPPC is used to operate the TWPA.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.simple import dsl
from scipy.ndimage import maximum_filter, minimum_filter

from laboneq_applications.analysis.options import (
    BasePlottingOptions,
    TuneUpAnalysisWorkflowOptions,
)
from laboneq_applications.analysis.plotting_helpers import timestamped_title
from laboneq_applications.core import validation

if TYPE_CHECKING:
    import matplotlib as mpl
    import uncertainties as unc
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.qpu_types.twpa import TWPA


@workflow.workflow_options(base_class=TuneUpAnalysisWorkflowOptions)
class TWPATuneUpAnalysisWorkflowOptions:
    """Options for the TWPA tune-up analysis workflow."""

    do_snr: bool = workflow.option_field(
        False, description="Whether to run SNR measurement."
    )
    use_probe_from_ppc: bool = workflow.option_field(
        True, description="Whether to use the PPC for sending the probe tone."
    )


@workflow.task_options
class DoSNROption:
    """The `do_snr` option for the TWPA tune-up analysis.

    Attributes:
        do_snr:
            Whether to run SNR measurement.
            Default: `False`.
    """

    do_snr: bool = workflow.option_field(
        False, description="Whether to run SNR measurement."
    )


@workflow.task_options
class Plot2DTWPAOptions(DoSNROption, BasePlottingOptions):
    """Options for the `plot_2d` task of the TWPA tune-up analysis workflow.

    Attributes:
        do_fitting:
            Whether to perform the fit.
            Default: `True`.

    Additional attributes from `TWPADoSNROption`:
        do_snr:
            Whether to run SNR measurement.
            Default: `False`.

    Additional attributes from `BasePlottingOptions`:
        save_figures:
            Whether to save the figures.
            Default: `True`.
        close_figures:
            Whether to close the figures.
            Default: `True`.
    """

    do_fitting: bool = workflow.option_field(
        True, description="Whether to perform the fit."
    )


@workflow.workflow
def analysis_workflow(
    signal_pump_on: RunExperimentResults,
    signal_pump_off: RunExperimentResults,
    parametric_amplifier: TWPA,
    pump_frequency: ArrayLike,
    pump_power: ArrayLike,
    noise_pump_on: RunExperimentResults | None = None,
    noise_pump_off: RunExperimentResults | None = None,
    options: TWPATuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The TWPA tune-up analysis workflow."""
    with workflow.if_(options.do_snr):
        signal_data = calculate_data_psd(
            parametric_amplifier, signal_pump_on, signal_pump_off
        )
        noise_data = calculate_data_psd(
            parametric_amplifier, noise_pump_on, noise_pump_off
        )
        fit_results = fit_data(pump_frequency, pump_power, signal_data, noise_data)
        parametric_amplifier_parameters = extract_parametric_amplifier_parameters(
            parametric_amplifier, fit_results
        )
        with workflow.if_(options.do_plotting):
            plot_2d(
                parametric_amplifier,
                fit_results,
                parametric_amplifier_parameters,
                pump_frequency,
                pump_power,
                signal_data,
                noise_data,
            )
    with workflow.else_():
        with workflow.if_(options.use_probe_from_ppc):
            signal_data = calculate_data_psd(
                parametric_amplifier, signal_pump_on, signal_pump_off
            )
        with workflow.else_():
            signal_data = calculate_data(
                parametric_amplifier, signal_pump_on, signal_pump_off
            )
        noise_data = None
        fit_results = fit_data(pump_frequency, pump_power, signal_data, noise_data)
        parametric_amplifier_parameters = extract_parametric_amplifier_parameters(
            parametric_amplifier, fit_results
        )
        with workflow.if_(options.do_plotting):
            plot_2d(
                parametric_amplifier,
                fit_results,
                parametric_amplifier_parameters,
                pump_frequency,
                pump_power,
                signal_data,
                noise_data,
            )

    workflow.return_(parametric_amplifier_parameters)


@workflow.task
def calculate_data_psd(
    parametric_amplifier: TWPA,
    result_pump_on: RunExperimentResults,
    result_pump_off: RunExperimentResults,
) -> dict[str, ArrayLike]:
    """Calculate the power spectral density."""
    data_pump_on = result_pump_on[
        dsl.handles.result_handle(parametric_amplifier.uid)
    ].data
    data_pump_on_dbm = 10 * np.log10(
        (1 / parametric_amplifier.parameters.readout_length)
        * np.abs(data_pump_on)
        / 50
        / 0.001
    )

    data_pump_off = result_pump_off[
        dsl.handles.result_handle(parametric_amplifier.uid)
    ].data
    data_pump_off_dbm = 10 * np.log10(
        (1 / parametric_amplifier.parameters.readout_length)
        * np.abs(data_pump_off)
        / 50
        / 0.001
    )

    return {
        "data_pump_on_dbm": data_pump_on_dbm,
        "data_pump_off_dbm": data_pump_off_dbm,
    }


@workflow.task
def calculate_data(
    parametric_amplifier: TWPA,
    result_pump_on: RunExperimentResults,
    result_pump_off: RunExperimentResults,
) -> dict[str, ArrayLike]:
    """Calculate the power spectral density."""
    data_pump_on = result_pump_on[
        dsl.handles.result_handle(parametric_amplifier.uid)
    ].data
    data_pump_on_dbm = 20 * np.log10(np.abs(data_pump_on))

    data_pump_off = result_pump_off[
        dsl.handles.result_handle(parametric_amplifier.uid)
    ].data
    data_pump_off_dbm = 20 * np.log10(np.abs(data_pump_off))

    return {
        "data_pump_on_dbm": data_pump_on_dbm,
        "data_pump_off_dbm": data_pump_off_dbm,
    }


@workflow.task
def fit_data(
    pump_frequency: ArrayLike,
    pump_power: ArrayLike,
    signal_dict: dict,
    noise_dict: dict | None = None,
    options: DoSNROption | None = None,
) -> dict[str, ArrayLike]:
    """Fit the data to obtain the maximum gain and SNR."""
    opts = DoSNROption() if options is None else options
    fit_results = {}
    signal_pump_on = signal_dict["data_pump_on_dbm"]
    signal_pump_off = signal_dict["data_pump_off_dbm"]
    x = pump_frequency
    y = pump_power
    z = signal_pump_on - signal_pump_off
    max_gain = maximum_filter(z, size=z.shape)
    for i, j in zip(*np.where(max_gain == z)):
        fit_results["max_gain_pump_freq"] = x[j]
        fit_results["max_gain_pump_power"] = y[i]
        fit_results["max_gain"] = z[i, j]

    if opts.do_snr:
        noise_pump_on = noise_dict["data_pump_on_dbm"]
        noise_pump_off = noise_dict["data_pump_off_dbm"]
        z_noise = noise_pump_on - noise_pump_off
        z_snr = z - z_noise

        min_nr = minimum_filter(z_noise, size=z_noise.shape)
        for i, j in zip(*np.where(z_noise == min_nr)):
            fit_results["min_noise_pump_freq"] = x[j]
            fit_results["min_noise_pump_power"] = y[i]
            fit_results["min_noise"] = z[i, j]

        max_snr = maximum_filter(z_snr, size=z_snr.shape)
        for i, j in zip(*np.where(max_snr == z_snr)):
            fit_results["max_SNR_pump_freq"] = x[j]
            fit_results["max_SNR_pump_power"] = y[i]
            fit_results["max_snr"] = z_snr[i, j]

    return fit_results


@workflow.task
def extract_parametric_amplifier_parameters(
    parametric_amplifier: TWPA,
    fit_results: dict,
    options: DoSNROption | None = None,
) -> dict[str, dict[str, dict[str, int | float | unc.core.Variable | None]]]:
    """Extract the parametric amplifier parameters."""
    opts = DoSNROption() if options is None else options
    parametric_amplifier = validation.validate_and_convert_single_qubit_sweeps(
        parametric_amplifier
    )
    parametric_amplifier_parameters = {
        "old_parameter_values": {parametric_amplifier.uid: {}},
        "new_parameter_values": {parametric_amplifier.uid: {}},
    }

    # Store the readout resonator frequency value
    parametric_amplifier_parameters["old_parameter_values"][
        parametric_amplifier.uid
    ] = {
        "pump_frequency": parametric_amplifier.parameters.pump_frequency,
        "pump_power": parametric_amplifier.parameters.pump_power,
    }

    if opts.do_snr:
        parametric_amplifier_parameters["new_parameter_values"][
            parametric_amplifier.uid
        ] = {
            "pump_frequency": fit_results["max_SNR_pump_freq"],
            "pump_power": fit_results["max_SNR_pump_power"],
        }
    else:
        parametric_amplifier_parameters["new_parameter_values"][
            parametric_amplifier.uid
        ] = {
            "pump_frequency": fit_results["max_gain_pump_freq"],
            "pump_power": fit_results["max_gain_pump_power"],
        }

    return parametric_amplifier_parameters


@workflow.task
def plot_2d(  # noqa: PLR0915
    parametric_amplifier: TWPA,
    fit_results: dict,
    parametric_amplifier_parameters: TWPA,
    pump_frequency: ArrayLike,
    pump_power: ArrayLike,
    signal_dict: dict,
    noise_dict: dict | None = None,
    options: Plot2DTWPAOptions | None = None,
) -> mpl.figure.Figure | None:
    """Plot the 2D phase diagram of the TWPA."""
    opts = Plot2DTWPAOptions() if options is None else options
    signal_pump_on = signal_dict["data_pump_on_dbm"]
    signal_pump_off = signal_dict["data_pump_off_dbm"]
    x = pump_frequency / 1e9
    y = pump_power
    z = signal_pump_on - signal_pump_off
    if opts.do_snr:
        noise_pump_on = noise_dict["data_pump_on_dbm"]
        noise_pump_off = noise_dict["data_pump_off_dbm"]
        z_noise = noise_pump_on - noise_pump_off
        z_snr = z - z_noise

        fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(5, 5 / 1.6 * 3), sharex=True)
        fig.tight_layout()
        y0 = axs[0].pcolor(x, y, z, cmap="inferno")
        cbar = fig.colorbar(y0, ax=axs[0], orientation="vertical")
        cbar.ax.set_ylabel("Signal gain $S_{p,on} / S_{p,off}$ (dB)")
        axs[0].set_ylabel("Pump Power (dBm)")
        axs[0].set_title(timestamped_title("TWPA phase diagram"))
        if opts.do_fitting:
            axs[0].plot(
                fit_results["max_gain_pump_freq"] / 1e9,
                fit_results["max_gain_pump_power"],
                marker="+",
                markersize=8,
                color="green",
            )

        y1 = axs[1].pcolor(x, y, z_noise, cmap="inferno")
        cbar = fig.colorbar(y1, ax=axs[1], orientation="vertical")
        cbar.ax.set_ylabel("Noise Rise $N_{p,on} / N_{p,off}$ (dB)")
        axs[1].set_ylabel("Pump Power (dBm)")
        if opts.do_fitting:
            axs[1].plot(
                fit_results["min_noise_pump_freq"] / 1e9,
                fit_results["min_noise_pump_power"],
                marker="+",
                markersize=8,
                color="green",
            )

        y2 = axs[2].pcolor(x, y, z_snr, cmap="inferno")
        cbar = fig.colorbar(y2, ax=axs[2], orientation="vertical")
        cbar.ax.set_ylabel("$SNR_{p,on} / SNR_{p,off}$ (dB)")
        axs[2].set_xlabel("Pump Frequency (GHz)")
        axs[2].set_ylabel("Pump Power (dBm)")
        if opts.do_fitting:
            axs[2].plot(
                fit_results["max_SNR_pump_freq"] / 1e9,
                fit_results["max_SNR_pump_power"],
                marker="+",
                markersize=8,
                color="green",
            )
        # Textbox
        old_rr_freq = parametric_amplifier_parameters["old_parameter_values"][
            parametric_amplifier.uid
        ]["pump_frequency"]
        old_rr_power = parametric_amplifier_parameters["old_parameter_values"][
            parametric_amplifier.uid
        ]["pump_power"]
        rr_freq = parametric_amplifier_parameters["new_parameter_values"][
            parametric_amplifier.uid
        ]["pump_frequency"]
        rr_power = parametric_amplifier_parameters["new_parameter_values"][
            parametric_amplifier.uid
        ]["pump_power"]
        textstr = f"Pump frequency: {rr_freq / 1e9:.4f} GHz "
        textstr += f"\nPrevious value: {old_rr_freq / 1e9:.4f} GHz"
        textstr += f"\nPump power: {rr_power:.2f} dBm "
        textstr += f"\nPrevious value: {old_rr_power:.2f} dBm"
        axs[2].text(0, -0.35, textstr, ha="left", va="top", transform=axs[2].transAxes)

    else:
        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(5, 5 / 1.6))
        fig.tight_layout()
        y = axs.pcolor(x, y, z, cmap="inferno")
        cbar = fig.colorbar(y, ax=axs, orientation="vertical")
        cbar.ax.set_ylabel("Signal Gain (dB)")
        axs.set_xlabel("Pump Frequency (GHz)")
        axs.set_ylabel("Pump Power (dBm)")
        axs.set_title("Gain Landscape")
        if opts.do_fitting:
            axs.plot(
                fit_results["max_gain_pump_freq"] / 1e9,
                fit_results["max_gain_pump_power"],
                marker="+",
                markersize=8,
                color="green",
            )
        # Textbox
        old_rr_freq = parametric_amplifier_parameters["old_parameter_values"][
            parametric_amplifier.uid
        ]["pump_frequency"]
        old_rr_power = parametric_amplifier_parameters["old_parameter_values"][
            parametric_amplifier.uid
        ]["pump_power"]
        rr_freq = parametric_amplifier_parameters["new_parameter_values"][
            parametric_amplifier.uid
        ]["pump_frequency"]
        rr_power = parametric_amplifier_parameters["new_parameter_values"][
            parametric_amplifier.uid
        ]["pump_power"]
        textstr = f"Pump frequency: {rr_freq / 1e9:.4f} GHz "
        textstr += f"\nPrevious value: {old_rr_freq / 1e9:.4f} GHz"
        textstr += f"\nPump power: {rr_power:.2f} dBm  "
        textstr += f"\nPrevious value: {old_rr_power:.2f} dBm"
        axs.text(0, -0.35, textstr, ha="left", va="top", transform=axs.transAxes)

    if opts.save_figures:
        workflow.save_artifact("twpa_tune_up", fig)

    if opts.close_figures:
        plt.close(fig)

    return fig
