# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the analysis for a TWPA gain curve experiment.

After optimizing the pump tone parameters, this analysis shows the readout spectrum
as a function of the pump power.

The experiment is defined in laboneq_applications.contrib.experiments.measure_gain_curve
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.simple import dsl
from scipy.ndimage import maximum_filter

from laboneq_applications.analysis.plotting_helpers import timestamped_title
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import (
    TuneupAnalysisOptions,
    TuneUpAnalysisWorkflowOptions,
)

if TYPE_CHECKING:
    import matplotlib as mpl
    import uncertainties as unc
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.qpu_types.twpa import TWPAParameters


@workflow.workflow_options(base_class=TuneUpAnalysisWorkflowOptions)
class TWPAGainCurveAnalysisWorkflowOptions:
    """Options for the TWPA gain curve analysis workflow."""

    use_probe_from_ppc: bool = workflow.option_field(
        True, description="Whether to use the PPC for sending the probe tone."
    )


@workflow.workflow
def analysis_workflow(
    result_pump_on: RunExperimentResults,
    result_pump_off: RunExperimentResults,
    parametric_amplifier: TWPAParameters,
    probe_frequency: ArrayLike,
    pump_power: ArrayLike,
    selected_indexes: list | None = None,
    options: TWPAGainCurveAnalysisWorkflowOptions | None = None,
) -> None:
    """The gain curve analysis workflow."""
    with workflow.if_(options.use_probe_from_ppc):
        signal_data = calculate_data_PSD(
            parametric_amplifier, result_pump_on, result_pump_off
        )
    with workflow.else_():
        signal_data = calculate_data(
            parametric_amplifier, result_pump_on, result_pump_off
        )
    fit_results = fit_data(
        parametric_amplifier, signal_data, probe_frequency, pump_power
    )
    parametric_amplifier_parameters = extract_parametric_amplifier_parameters(
        parametric_amplifier, fit_results
    )

    with workflow.if_(options.do_plotting):
        plot_2D(
            parametric_amplifier,
            parametric_amplifier_parameters,
            fit_results,
            signal_data,
            probe_frequency,
            pump_power,
        )
        plot_1D(
            parametric_amplifier,
            parametric_amplifier_parameters,
            signal_data,
            probe_frequency,
            pump_power,
            selected_indexes,
        )
    workflow.return_(parametric_amplifier_parameters)


@workflow.task
def calculate_data_PSD(  # noqa: N802
    parametric_amplifier: TWPAParameters,
    result_pump_on: RunExperimentResults,
    result_pump_off: RunExperimentResults,
) -> dict[str, ArrayLike]:
    """Calculate the PSD data."""
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
    parametric_amplifier: TWPAParameters,
    result_pump_on: RunExperimentResults,
    result_pump_off: RunExperimentResults,
) -> dict[str, ArrayLike]:
    """Calculate the data."""
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
    parametric_amplifier: TWPAParameters,
    signal_dict: dict,
    probe_frequency: ArrayLike,
    pump_power: ArrayLike,
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, ArrayLike]:
    """Fit the data."""
    fit_results = {}
    data = signal_dict["data_pump_on_dbm"]
    ref = signal_dict["data_pump_off_dbm"]
    x = probe_frequency
    y = pump_power
    z = data - ref

    max_gain = maximum_filter(z, size=z.shape)
    for i, j in zip(*np.where(max_gain == z)):
        fit_results["max_gain_probe_freq"] = x[j]
        fit_results["max_gain_pump_power"] = y[i]
        fit_results["max_gain"] = z[i, j]

    return fit_results


@workflow.task
def extract_parametric_amplifier_parameters(
    parametric_amplifier: TWPAParameters,
    fit_results: dict,
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, dict[str, dict[str, int | float | unc.core.Variable | None]]]:
    """Extract the parametric amplifier parameters."""
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
        "probe_frequency": parametric_amplifier.parameters.probe_frequency,
        "pump_power": parametric_amplifier.parameters.pump_power,
    }
    parametric_amplifier_parameters["new_parameter_values"][
        parametric_amplifier.uid
    ] = {
        "probe_frequency": fit_results["max_gain_probe_freq"],
        "pump_power": fit_results["max_gain_pump_power"],
    }

    return parametric_amplifier_parameters


@workflow.task
def plot_2D( # noqa: N802
    parametric_amplifier: TWPAParameters,
    parametric_amplifier_parameters: dict,
    fit_results: dict,
    signal_dict: dict,
    probe_frequency: ArrayLike,
    pump_power: ArrayLike,
    options: TuneupAnalysisOptions | None = None,
) -> mpl.figure.Figure | None:
    """Plot the 2D gain diagram."""
    opts = TuneupAnalysisOptions() if options is None else options

    data = signal_dict["data_pump_on_dbm"]
    ref = signal_dict["data_pump_off_dbm"]
    x = np.array(probe_frequency) / 1e9
    y = pump_power
    z = data - ref

    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(5, 5 / 1.6))
    fig.tight_layout()
    y = axs.pcolor(x, y, z, cmap="inferno")
    cbar = fig.colorbar(y, ax=axs, orientation="vertical")
    cbar.ax.set_ylabel("Signal gain (dB)")
    axs.set_xlabel("Probe frequency (GHz)")
    axs.set_ylabel("Pump power (dBm)")
    axs.set_title(timestamped_title("Gain curve"))
    axs.plot(
        fit_results["max_gain_probe_freq"] / 1e9,
        fit_results["max_gain_pump_power"],
        marker="+",
        markersize=8,
        color="green",
    )
    # Textbox
    old_rr_freq = parametric_amplifier_parameters["old_parameter_values"][
        parametric_amplifier.uid
    ]["probe_frequency"]
    old_rr_power = parametric_amplifier_parameters["old_parameter_values"][
        parametric_amplifier.uid
    ]["pump_power"]
    rr_freq = parametric_amplifier_parameters["new_parameter_values"][
        parametric_amplifier.uid
    ]["probe_frequency"]
    rr_power = parametric_amplifier_parameters["new_parameter_values"][
        parametric_amplifier.uid
    ]["pump_power"]
    textstr = f"Probe frequency: " f"{rr_freq / 1e9:.4f} GHz "
    textstr += f"\nPrevious value: {old_rr_freq / 1e9:.4f} GHz"
    textstr += f"\nProbe power: " f"{rr_power:.4f} dBm "
    textstr += f"\nPrevious value: {old_rr_power:.4f} dBm"
    axs.text(0, -0.35, textstr, ha="left", va="top", transform=axs.transAxes)
    if opts.save_figures:
        workflow.save_artifact("2D Gain_diagram", fig)

    if opts.close_figures:
        plt.close(fig)

    return fig


@workflow.task
def plot_1D( # noqa: N802
    parametric_amplifier: TWPAParameters,
    parametric_amplifier_parameters: dict,
    signal_dict: dict,
    probe_frequency: ArrayLike,
    pump_power: ArrayLike,
    selected_indexes: list | None = None,
    options: TuneupAnalysisOptions | None = None,
) -> mpl.figure.Figure | None:
    """Plot the 1D gain diagram."""
    opts = TuneupAnalysisOptions() if options is None else options
    data = signal_dict["data_pump_on_dbm"]
    ref = signal_dict["data_pump_off_dbm"]
    x = np.array(probe_frequency) / 1e9
    y = pump_power
    z = data - ref

    fig, axs = plt.subplots(figsize=(5.3, 3))

    # Plot z vs x for each selected y value
    for idx in selected_indexes:
        # Find the index of the closest y value
        axs.plot(x, z[idx, :], label=f"Pump power = {y[idx]} dBm")

    axs.set_xlabel("Probe frequency (GHz)")
    axs.set_ylabel("Signal gain (dB)")
    axs.legend(loc="best")
    axs.grid(True)

    if opts.save_figures:
        workflow.save_artifact("1D_Gain_plot", fig)

    if opts.close_figures:
        plt.close(fig)

    return fig
