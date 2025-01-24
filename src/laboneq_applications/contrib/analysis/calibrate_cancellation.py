# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the analysis for a cancellation tone calibration experiment.

The experiment is defined in laboneq_applications.experiments.

In this analysis, we first interpret the raw data into the signal magnitude and phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.simple import dsl
from scipy.ndimage import minimum_filter

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

    from laboneq_applications.qpu_types.twpa import TWPA


@workflow.workflow
def analysis_workflow(
    result_data: RunExperimentResults,
    result_ref: RunExperimentResults,
    parametric_amplifier: TWPA,
    cancel_phaseshift: ArrayLike,
    cancel_attenuation: ArrayLike,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The cancellation tone calibration analysis workflow."""
    signal_data = calculate_data_PSD(parametric_amplifier, result_data, result_ref)

    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_fitting):
            fit_results = fit_data(signal_data, cancel_phaseshift, cancel_attenuation)
            pa_parameters = extract_cancellation_parameters(
                parametric_amplifier, fit_results
            )
        plot_2D(
            parametric_amplifier,
            pa_parameters,
            fit_results,
            signal_data,
            cancel_phaseshift,
            cancel_attenuation,
        )

    workflow.return_(pa_parameters)


@workflow.task
def calculate_data_PSD(  # noqa: N802
    parametric_amplifier: TWPA,
    result_data: RunExperimentResults,
    result_ref: RunExperimentResults,
) -> dict[str, ArrayLike]:
    """Calculate the power spectral density of the pump tone residual."""
    ref = result_ref[dsl.handles.result_handle(parametric_amplifier.uid)].data
    ref_dbm = 10 * np.log10(
        (1 / parametric_amplifier.parameters.readout_length) * np.abs(ref) / 50 / 0.001
    )

    data = result_data[dsl.handles.result_handle(parametric_amplifier.uid)].data
    data_dbm = 10 * np.log10(
        (1 / parametric_amplifier.parameters.readout_length) * np.abs(data) / 50 / 0.001
    )

    return {
        "data_dbm": data_dbm,
        "ref_dbm": ref_dbm,
    }


@workflow.task
def fit_data(
    signal_dict: dict,
    cancel_phaseshift: ArrayLike,
    cancel_attenuation: ArrayLike,
) -> dict[str, ArrayLike]:
    """Fit data."""
    fit_results = {}

    data = signal_dict["data_dbm"]
    ref = signal_dict["ref_dbm"]
    x = cancel_phaseshift
    y = cancel_attenuation
    z = data - ref

    min_pump_tone = minimum_filter(z, size=z.shape)
    for i, j in zip(*np.where(min_pump_tone == z)):
        fit_results["cancel_phaseshift"] = x[j]
        fit_results["cancel_attenuation"] = y[i]
        fit_results["max_cancel"] = z[i, j]

    return fit_results


@workflow.task
def extract_cancellation_parameters(
    parametric_amplifier: TWPA,
    fit_results: dict,
) -> dict[str, dict[str, dict[str, int | float | unc.core.Variable | None]]]:
    """Extract the cancellation parameters."""
    parametric_amplifier = validation.validate_and_convert_single_qubit_sweeps(
        parametric_amplifier
    )
    parametric_amplifier_parameters = {
        "old_parameter_values": {parametric_amplifier.uid: {}},
        "new_parameter_values": {parametric_amplifier.uid: {}},
    }

    # Store the readout resonator frequency value
    cancellation_attenuation = parametric_amplifier.parameters.cancellation_attenuation
    parametric_amplifier_parameters["old_parameter_values"][
        parametric_amplifier.uid
    ] = {
        "cancellation_phase": parametric_amplifier.parameters.cancellation_phase,
        "cancellation_attenuation": cancellation_attenuation,
    }

    parametric_amplifier_parameters["new_parameter_values"][
        parametric_amplifier.uid
    ] = {
        "cancellation_phase": fit_results["cancel_phaseshift"],
        "cancellation_attenuation": fit_results["cancel_attenuation"],
    }

    return parametric_amplifier_parameters


@workflow.task
def plot_2D(  # noqa: N802
    parametric_amplifier: TWPA,
    pa_parameters: dict,
    fit_results: dict,
    signal_dict: dict,
    cancel_phaseshift: ArrayLike,
    cancel_attenuation: ArrayLike,
    options: TuneupAnalysisOptions | None = None,
) -> mpl.figure.Figure | None:
    """Plot the 2D plot of the pump tone residual."""
    opts = TuneupAnalysisOptions() if options is None else options

    data = signal_dict["data_dbm"]
    ref = signal_dict["ref_dbm"]

    x = cancel_phaseshift
    y = cancel_attenuation
    z = data - ref

    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(5, 5 / 1.6))
    fig.tight_layout()
    y = axs.pcolor(x, y, z, cmap="inferno")
    cbar = fig.colorbar(y, ax=axs, orientation="vertical")
    cbar.ax.set_ylabel("$P_{cancel,on} / P_{cancel,off} $ (dB)")
    axs.set_xlabel("Cancellation phase shift (rad)")
    axs.set_ylabel("Cancellation attenuation (dB)")
    axs.set_title(timestamped_title("Pump tone residual"))
    if opts.do_fitting:
        axs.plot(
            fit_results["cancel_phaseshift"],
            fit_results["cancel_attenuation"],
            marker="+",
            markersize=8,
            color="green",
        )
    # Textbox
    suppression = fit_results["max_cancel"]
    old_rr_ph = pa_parameters["old_parameter_values"][parametric_amplifier.uid][
        "cancellation_phase"
    ]
    old_rr_att = pa_parameters["old_parameter_values"][parametric_amplifier.uid][
        "cancellation_attenuation"
    ]
    rr_ph = pa_parameters["new_parameter_values"][parametric_amplifier.uid][
        "cancellation_phase"
    ]
    rr_att = pa_parameters["new_parameter_values"][parametric_amplifier.uid][
        "cancellation_attenuation"
    ]
    textstr = f"Pump tone can be suppressed by {suppression:.2f} dB"
    textstr += f"\nOptimal cancellation phase: {rr_ph:.2f} rad"
    textstr += f"\nPrevious value: {old_rr_ph:.2f} rad"
    textstr += f"\nOptimal cancellation attenuation: {rr_att:.1f} dB"
    textstr += f"\nPrevious value: {old_rr_att:.1f} dB"

    axs.text(0, -0.35, textstr, ha="left", va="top", transform=axs.transAxes)

    if opts.save_figures:
        workflow.save_artifact("Cancellation_calibration", fig)

    if opts.close_figures:
        plt.close(fig)

    return fig
