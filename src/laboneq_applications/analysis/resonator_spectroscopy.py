"""This module defines the analysis for a resonator-spectroscopy experiment.

The experiment is defined in laboneq_applications.experiments.

In this analysis, we first interpret the raw data into the signal magnitude and phase.
Then we either extract the frequency corresponding to the min or max of the magnitude
data, or we fit a Lorentzian model to the signal magnitude and extract frequency
corresponding to the peak of the Lorentzian from the fit. Finally, we plot the data and
the fit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc

from laboneq_applications import dsl, workflow
from laboneq_applications.analysis.fitting_helpers import lorentzian_fit
from laboneq_applications.analysis.plotting_helpers import (
    plot_raw_complex_data_1d,
    timestamped_title,
)
from laboneq_applications.core.validation import validate_result
from laboneq_applications.experiments.options import (
    ResonatorSpectroscopyExperimentOptions,
    WorkflowOptions,
)
from laboneq_applications.workflow import option_field, options

if TYPE_CHECKING:
    import lmfit
    import matplotlib as mpl
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from numpy.typing import ArrayLike

    from laboneq_applications.tasks.run_experiment import RunExperimentResults


@options
class ResonatorSpectroscopyAnalysisOptions(ResonatorSpectroscopyExperimentOptions):
    """Options for the analysis of the resonator spectroscopy experiment.

    Attributes:
        fit_lorentzian:
            Whether to fit a Lorentzian model to the data.
            Default: `False`.
        fit_parameters_hints:
            Parameters hints accepted by lmfit
            Default: None.
        find_peaks:
            Whether to search for peaks (True) or dips (False) in the spectrum.
            Default: `False`.
        save_figures:
            Whether to save the figures.
            Default: `True`.
        close_figures:
            Whether to close the figures.
            Default: `True`.

    """

    fit_lorentzian: bool = option_field(
        False, description="Whether to fit a Lorentzian model to the data."
    )
    fit_parameters_hints: dict[str, dict[str, float | bool | str]] | None = (
        option_field(None, description="Parameters hints accepted by lmfit")
    )
    find_peaks: bool = option_field(
        False,
        description="Whether to search for peaks (True) or dips (False) "
        "in the spectrum.",
    )
    save_figures: bool = option_field(True, description="Whether to save the figures.")
    close_figures: bool = option_field(
        True, description="Whether to close the figures."
    )


@options
class ResonatorSpectroscopyAnalysisWorkflowOptions(WorkflowOptions):
    """Option class for spectroscopy analysis workflows.

    Attributes:
        do_plotting:
            Whether to create plots.
            Default: 'True'.
        do_raw_data_plotting:
            Whether to plot the raw data.
            Default: True.
        do_plotting_magnitude_phase:
            Whether to plot the magnitude and phase.
            Default: True.
        do_plotting_real_imaginary:
            Whether to plot the real and imaginary data.
            Default: True.
    """

    do_plotting: bool = option_field(True, description="Whether to create plots.")
    do_raw_data_plotting: bool = option_field(
        True, description="Whether to plot the raw data."
    )
    do_plotting_magnitude_phase: bool = option_field(
        True, description="Whether to plot the magnitude and phase."
    )
    do_plotting_real_imaginary: bool = option_field(
        True, description="Whether to plot the real and imaginary data."
    )


@workflow.workflow(name="resonator_spectroscopy_analysis")
def analysis_workflow(
    result: RunExperimentResults,
    qubit: QuantumElement,
    frequencies: ArrayLike,
    options: ResonatorSpectroscopyAnalysisWorkflowOptions | None = None,
) -> None:
    """The Resonator Spectroscopy analysis Workflow.

    The workflow consists of the following steps:

    - [calculate_signal_magnitude_and_phase]()
    - [fit_data]()
    - [extract_qubit_parameters]()
    - [plot_raw_complex_data_1d]()
    - [plot_magnitude_phase]()
    - [plot_real_imaginary]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubit:
            The qubit on which to run the analysis. The UID of this qubit must exist
            in the result.
        frequencies:
            The array of frequencies that were swept over in the experiment.
        options:
            The options for building the workflow, passed as an instance of
                [ResonatorSpectroscopyAnalysisWorkflowOptions]. See the docstring of
                [ResonatorSpectroscopyAnalysisWorkflowOptions] for more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        result = analysis_workflow(
            results=results
            qubit=q0,
            frequencies=np.linspace(7.0, 7.1, 101),
            options=analysis_workflow.options(),
        ).run()
        ```
    """
    processed_data_dict = calculate_signal_magnitude_and_phase(
        qubit, result, frequencies
    )
    fit_result = fit_data(processed_data_dict)
    qubit_parameters = extract_qubit_parameters(qubit, processed_data_dict, fit_result)
    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_raw_data_plotting):
            plot_raw_complex_data_1d(
                qubit,
                result,
                frequencies,
                xlabel="Readout Frequency, $f_{\\mathrm{RO}}$ (GHz)",
                xscaling=1e-9,
            )
        with workflow.if_(options.do_plotting_magnitude_phase):
            plot_magnitude_phase(
                qubit, processed_data_dict, fit_result, qubit_parameters
            )
        with workflow.if_(options.do_plotting_real_imaginary):
            plot_real_imaginary(qubit, result)
    workflow.return_(qubit_parameters)


@workflow.task
def calculate_signal_magnitude_and_phase(
    qubit: QuantumElement,
    result: RunExperimentResults,
    frequencies: ArrayLike,
) -> dict[str, ArrayLike]:
    """Calculates the magnitude and phase of the spectroscopy signal in result.

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubit:
            The qubit on which to run the analysis. The UID of this qubit must exist
            in the result.
        frequencies:
            The array of frequencies that were swept over in the experiment.

    Returns:
        dictionary with the following data:
            sweep_points
            data_raw
            magnitude
            phase
    """
    validate_result(result)
    qubit, frequencies = dsl.validation.validate_and_convert_single_qubit_sweeps(
        qubit, frequencies
    )

    raw_data = result[dsl.handles.result_handle(qubit.uid)].data
    return {
        "sweep_points": frequencies,
        "data_raw": raw_data,
        "magnitude": np.abs(raw_data),
        "phase": np.angle(raw_data),
    }


@workflow.task
def fit_data(
    processed_data_dict: dict[str, ArrayLike],
    options: ResonatorSpectroscopyAnalysisOptions | None = None,
) -> lmfit.model.ModelResult | None:
    """Perform a fit of a Lorentzian model to the data if fit_lorentzian == True.

    Arguments:
        processed_data_dict: the processed data dictionary returned by
            calculate_signal_magnitude_and_phase
        options:
            The options for processing the raw data.
            See [ResonatorSpectroscopyAnalysisOptions] and
            [BaseExperimentOptions] for accepted options.

    Returns:
        dict with qubit UIDs as keys and the fit results for each qubit as keys.
    """
    opts = ResonatorSpectroscopyAnalysisOptions() if options is None else options
    fit_result = None

    if opts.fit_lorentzian:
        swpts_fit = processed_data_dict["sweep_points"]
        data_to_fit = processed_data_dict["magnitude"]
        try:
            fit_res = lorentzian_fit(
                swpts_fit,
                data_to_fit,
                param_hints=opts.fit_parameters_hints,
            )
            fit_result = fit_res
        except ValueError as err:
            workflow.log(logging.ERROR, "Fit failed: %s", err)

    return fit_result


@workflow.task
def extract_qubit_parameters(
    qubit: QuantumElement,
    processed_data_dict: dict[str, ArrayLike],
    fit_result: lmfit.model.ModelResult | None,
    options: ResonatorSpectroscopyAnalysisOptions | None = None,
) -> dict[str, dict[str, dict[str, int | float | unc.core.Variable | None]]]:
    """Extract the qubit parameters from the fit results.

    Arguments:
        qubit:
            The qubit on which to run the analysis.
        processed_data_dict: the processed data dictionary returned by
            calculate_signal_magnitude_and_phase.
        fit_result: the lmfit ModelResults returned by fit_data
        options:
            The options for extracting the qubit parameters.
            See [ResonatorSpectroscopyAnalysisOptions] and
            [BaseExperimentOptions] for accepted options.

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
    opts = ResonatorSpectroscopyAnalysisOptions() if options is None else options
    qubit = dsl.validation.validate_and_convert_single_qubit_sweeps(qubit)

    qubit_parameters = {
        "old_parameter_values": {qubit.uid: {}},
        "new_parameter_values": {qubit.uid: {}},
    }

    # Store the readout resonator frequency value
    qubit_parameters["old_parameter_values"][qubit.uid] = {
        "readout_resonator_frequency": qubit.parameters.readout_resonator_frequency,
    }

    # Extract and store the readout resonator frequency value
    if fit_result is not None:
        rr_freq = unc.ufloat(
            fit_result.params["position"].value,
            fit_result.params["position"].stderr,
        )
    else:
        # find frequency at min or max of the signal magnitude
        take_extremum = np.argmax if opts.find_peaks else np.argmin
        freqs = processed_data_dict["sweep_points"]
        signal_magnitude = processed_data_dict["magnitude"]
        rr_freq = unc.ufloat(
            freqs[take_extremum(signal_magnitude)],
            0,
        )

    qubit_parameters["new_parameter_values"][qubit.uid] = {
        "readout_resonator_frequency": rr_freq
    }

    return qubit_parameters


@workflow.task
def plot_magnitude_phase(
    qubit: QuantumElement,
    processed_data_dict: dict[str, ArrayLike],
    fit_result: lmfit.model.ModelResult | None,
    qubit_parameters: dict[
        str, dict[str, dict[str, int | float | unc.core.Variable | None]]
    ],
    options: ResonatorSpectroscopyAnalysisOptions | None = None,
) -> mpl.figure.Figure | None:
    """Plot the magnitude and phase of the spectroscopy signal.

    Arguments:
        qubit:
            The qubit on which to run the analysis. qubit_parameters.
        processed_data_dict: the processed data dictionary returned by
            calculate_signal_magnitude_and_phase.
        fit_result: the lmfit ModelResults returned by fit_data
        qubit_parameters: the qubit-parameters dictionary returned by
            extract_qubit_parameters
        options:
            The options for extracting the qubit parameters.
            See [ResonatorSpectroscopyAnalysisOptions] and
            [BaseExperimentOptions] for accepted options.

    Returns:
        the matplotlib figure

        If there are no new_parameter_values for the qubit, then fit result and the
        textbox with the extracted readout resonator frequency are not plotted.
    """
    opts = ResonatorSpectroscopyAnalysisOptions() if options is None else options
    qubit = dsl.validation.validate_and_convert_single_qubit_sweeps(qubit)

    sweep_points = processed_data_dict["sweep_points"]
    magnitude = processed_data_dict["magnitude"]
    phase = processed_data_dict["phase"]

    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].set_title(timestamped_title(f"Magnitude-Phase {qubit.uid}"))
    axs[0].plot(
        sweep_points / 1e9,
        magnitude,
        "-",
        zorder=2,
        label="data",
    )
    axs[0].set_ylabel("Transmission Signal\nMagnitude, $|S_{21}|$ (a.u.)")
    axs[1].plot(sweep_points / 1e9, phase, "-", zorder=2, label="data")
    axs[1].set_ylabel("Transmission Signal\nPhase, $|S_{21}|$ (a.u.)")
    axs[1].set_xlabel("Readout Frequency, $f_{\\mathrm{RO}}$ (GHz)")
    fig.align_ylabels()
    fig.subplots_adjust(hspace=0.1)

    if opts.fit_lorentzian and fit_result is not None:
        # Plot fit of the magnitude
        swpts_fine = np.linspace(sweep_points[0], sweep_points[-1], 501)
        axs[0].plot(
            swpts_fine / 1e9,
            fit_result.model.func(swpts_fine, **fit_result.best_values),
            "r-",
            zorder=1,
            label="fit",
        )

    if len(qubit_parameters["new_parameter_values"][qubit.uid]) > 0:
        rr_freq = qubit_parameters["new_parameter_values"][qubit.uid][
            "readout_resonator_frequency"
        ]

        # Point at the extracted readout resonator frequency
        if opts.fit_lorentzian:
            # Point on the magnitude plot from the fit result
            axs[0].plot(
                rr_freq.nominal_value / 1e9,
                fit_result.model.func(
                    rr_freq.nominal_value,
                    **fit_result.best_values,
                ),
                "or",
                zorder=3,
                markersize=plt.rcParams["lines.markersize"] + 1,
            )
            # Legend
            axs[0].legend(
                loc="center left",
                bbox_to_anchor=(1, 0),
                handlelength=1.5,
                frameon=False,
            )
        else:
            take_extremum = np.argmax if opts.find_peaks else np.argmin
            # Point on the magnitude plot at the rr freq
            axs[0].plot(
                rr_freq.nominal_value / 1e9,
                magnitude[take_extremum(magnitude)],
                "or",
                zorder=3,
                markersize=plt.rcParams["lines.markersize"] + 1,
            )

        # Line on the phase plot corresponding to the rr freq
        ylims = axs[1].get_ylim()
        axs[1].vlines(
            rr_freq.nominal_value / 1e9,
            *ylims,
            linestyles="--",
            colors="r",
            zorder=0,
        )
        axs[1].set_ylim(ylims)

        # Textbox
        old_rr_freq = qubit_parameters["old_parameter_values"][qubit.uid][
            "readout_resonator_frequency"
        ]
        textstr = (
            f"Readout-resonator frequency: "
            f"{rr_freq.nominal_value / 1e9:.4f} GHz $\\pm$ "
            f"{rr_freq.std_dev / 1e6:.4f} MHz"
        )
        textstr += f"\nPrevious value: {old_rr_freq / 1e9:.4f} GHz"
        axs[1].text(0, -0.35, textstr, ha="left", va="top", transform=axs[1].transAxes)

    if opts.save_figures:
        workflow.save_artifact(f"Magnitude_Phase_{qubit.uid}", fig)

    if opts.close_figures:
        plt.close(fig)

    return fig


@workflow.task
def plot_real_imaginary(
    qubit: QuantumElement,
    result: RunExperimentResults,
    options: ResonatorSpectroscopyAnalysisOptions | None = None,
) -> mpl.figure.Figure | None:
    """Create the amplitude-Rabi plots.

    Arguments:
        qubit:
            The qubit on which to run the analysis.
        result:
            The experiment results returned by the run_experiment task.
        options:
            The options for extracting the qubit parameters.
            See [ResonatorSpectroscopyAnalysisOptions] and
            [BaseExperimentOptions] for accepted options.

    Returns:
        the matplotlib figure
    """
    opts = ResonatorSpectroscopyAnalysisOptions() if options is None else options
    validate_result(result)
    qubit = dsl.validation.validate_and_convert_single_qubit_sweeps(qubit)

    raw_data = result[dsl.handles.result_handle(qubit.uid)].data

    fig, ax = plt.subplots()
    ax.set_title(timestamped_title(f"Real-Imaginary {qubit.uid}"))
    ax.set_xlabel("Real Transmission Signal, Re($S_{21}$) (a.u.)")
    ax.set_ylabel("Imaginary Transmission Signal, Im($S_{21}$) (a.u.)")
    ax.plot(
        np.real(raw_data),
        np.imag(raw_data),
        "o",
        zorder=2,
        label="data",
    )

    if opts.save_figures:
        workflow.save_artifact(f"Real_Imaginary_{qubit.uid}", fig)

    if opts.close_figures:
        plt.close(fig)

    return fig
