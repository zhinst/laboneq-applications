"""This module defines the analysis for a time-traces experiment.

The experiment is defined in laboneq_applications.experiments.

In this analysis, we extract the optimal integration kernels from the raw time traces.
The kernels allow maximal discrimination between the qubit states which were used to
measure the time traces. Then, we optionally apply a low-pass filter to the kernels with
a cut-off frequency chosen by the user to remove any spurious signals in the kernels
(for example, the down-converted TWPA pump tone). We then create the dictionary with
qubit parameters to update, and finally we plot the time traces and the kernels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from laboneq.analysis import calculate_integration_kernels_thresholds

from laboneq_applications import dsl, workflow
from laboneq_applications.analysis.plotting_helpers import timestamped_title
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.experiments.options import TaskOptions, WorkflowOptions
from laboneq_applications.workflow import option_field, options

if TYPE_CHECKING:
    from collections.abc import Sequence

    import matplotlib as mpl
    from numpy.typing import ArrayLike, NDArray

    from laboneq_applications.tasks.run_experiment import RunExperimentResults
    from laboneq_applications.typing import Qubits


@options
class TimeTracesAnalysisWorkflowOptions(WorkflowOptions):
    """Option class for the time-traces analysis workflow.

    Attributes:
        filter_kernels:
            Whether to filter the integration kernels.
            Default: 'False'.
        do_fitting:
            Whether to perform the fit.
            Default: `True`.
        do_plotting:
            Whether to create plots.
            Default: 'True'.
        do_plotting_time_traces:
            Whether to create the time-traces plots.
            Default: 'True'.
        do_plotting_kernels_traces:
            Whether to create the integration-kernel plots.
            Default: 'True'.
        do_plotting_kernels_fft:
            Whether to create the kernels-FFT plots.
            Default: 'True'.
    """

    filter_kernels: bool = option_field(
        False, description="Whether to filter the kernels."
    )
    do_fitting: bool = option_field(True, description="Whether to perform the fit.")
    do_plotting: bool = option_field(True, description="Whether to create plots.")
    do_plotting_time_traces: bool = option_field(
        True, description="Whether to create time-traces plots."
    )
    do_plotting_kernels_traces: bool = option_field(
        True, description="Whether to create the integration-kernel plots."
    )
    do_plotting_kernels_fft: bool = option_field(
        True, description="Whether to create the kernels-FFT plots."
    )


@options
class TimeTracesAnalysisOptions(TaskOptions):
    """Option class for the tasks in the time-traces analysis workflows.

    Attributes:
        granularity:
            The granularity of the acquisition instrument. Used to truncate the time
            traces to align them to the granularity grid.
            Default: 16.
        filter_cutoff_frequency:
            The cut-off frequency of the low-pass filter for the kernels. Only used
            if filter_kernels is True in TimeTracesAnalysisWorkflowOptions.
            Default: `None`.
        sampling_rate:
            The sampling rate of the acquisition instrument that was used to measure the
            time-traces. The sampling_rate is used when applying a low-pass filter to
            the kernels if filter_kernels is True in TimeTracesAnalysisWorkflowOptions.
            Default: 2e9.
        do_fitting:
            Whether to perform the fit.
            Default: `True`.
        save_figures:
            Whether to save the figures.
            Default: `True`.
        close_figures:
            Whether to close the figures.
            Default: `True`.
    """

    granularity: int = option_field(
        16, description="The granularity of the acquisition."
    )
    filter_cutoff_frequency: float | None = option_field(
        None, description="The cut-off frequency."
    )
    sampling_rate: float = option_field(2e9, description="The sampling rate.")
    do_fitting: bool = option_field(True, description="Whether to perform the fit.")
    save_figures: bool = option_field(True, description="Whether to save the figures.")
    close_figures: bool = option_field(
        True, description="Whether to close the figures."
    )


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: Qubits,
    states: Sequence[Literal["g", "e", "f"]],
    options: TimeTracesAnalysisWorkflowOptions | None = None,
) -> None:
    """The time-traces analysis Workflow.

    The workflow consists of the following tasks:

    - [truncate_time_traces]()
    - [extract_kernels_thresholds]()
    - [filter_integration_kernels]()
    - [extract_qubit_parameters]()
    - [plot_time_traces]()
    - [plot_kernels_traces]()
    - [plot_kernels_fft]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        states:
            The qubit basis states for which the time traces were measured. May be
            either a string, e.g. "gef", or a list of letters, e.g. ["g","e","f"].
        options:
            The options for building the workflow as an instance of
            [TimeTracesAnalysisWorkflowOptions]. See the docstring of this class for
            more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        options = analysis_workflow.options()
        options.close_figures(False)
        result = analysis_workflow(
            results=results
            qubits=[q0, q1],
            states="ge",
            options=options,
        ).run()
        ```
    """
    # truncate_time_traces
    truncated_time_traces = truncate_time_traces(qubits, result, states)

    # extract_kernels_thresholds
    out = extract_kernels_thresholds(qubits, truncated_time_traces)
    integration_kernels, discrimination_thresholds = out[0], out[1]

    # filter_integration_kernels
    integration_kernels_filtered = None
    with workflow.if_(options.filter_kernels):
        integration_kernels_filtered = filter_integration_kernels(
            qubits, integration_kernels
        )

    # extract_qubit_parameters
    qubit_parameters = extract_qubit_parameters(
        qubits,
        discrimination_thresholds,
        integration_kernels,
        integration_kernels_filtered,
    )

    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_plotting_time_traces):
            # plot_time_traces
            plot_time_traces(qubits, states, truncated_time_traces)
        with workflow.if_(options.do_fitting):
            with workflow.if_(options.do_plotting_kernels_traces):
                # plot_kernels_traces
                plot_kernels_traces(
                    qubits,
                    discrimination_thresholds,
                    integration_kernels,
                    integration_kernels_filtered,
                )
            with workflow.if_(options.do_plotting_kernels_fft):
                # plot_kernels_fft
                plot_kernels_fft(
                    qubits,
                    discrimination_thresholds,
                    integration_kernels,
                    integration_kernels_filtered,
                )

    workflow.return_(qubit_parameters)


@workflow.task
def truncate_time_traces(
    qubits: Qubits,
    result: RunExperimentResults,
    states: Sequence[Literal["g", "e", "f"]],
    options: TimeTracesAnalysisOptions | None = None,
) -> dict[str, list[ArrayLike]]:
    """Truncate the time traces to align on the granularity grid.

    The granularity is passed via the options and is typically 16 samples (default).

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits.
        result:
            The experiment results returned by the run_experiment task.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].
        options:
            The options for building the workflow as an instance of
            [TimeTracesAnalysisOptions]. See the docstring of this class for more
            details.

    Returns:
        dict with qubit UIDs as keys and the list of truncated time-traces for each
        qubit as keys.
    """
    qubits = validate_and_convert_qubits_sweeps(qubits)
    opts = TimeTracesAnalysisOptions() if options is None else options

    truncated_time_traces = {q.uid: [] for q in qubits}
    for q in qubits:
        raw_traces = []
        for s in states:
            trace = result[dsl.handles.result_handle(q.uid, suffix=s)].data
            raw_traces += [trace[: (len(trace) // opts.granularity) * opts.granularity]]
        truncated_time_traces[q.uid] = raw_traces

    return truncated_time_traces


@workflow.task
def extract_kernels_thresholds(
    qubits: Qubits,
    truncated_time_traces: dict[str, list[NDArray]],
    options: TimeTracesAnalysisOptions | None = None,
) -> tuple[dict[str, list[ArrayLike]] | None, dict[str, list[float]] | None]:
    """Extract the integration kernels and discrimination thresholds.

    Args:
        qubits:
            The qubits on which to run the task. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the keys of the
            truncated_time_traces dictionary.
        truncated_time_traces:
            The dictionary of truncated time traces for each qubit as returned by
            truncate_time_traces.
        options:
            The options for building the workflow as an instance of
            [TimeTracesAnalysisOptions]. See the docstring of this class for more
            details.

    Returns:
        a tuple with the list of integration kernel arrays and a list with the
        corresponding discrimination thresholds.
    """
    opts = TimeTracesAnalysisOptions() if options is None else options
    if not opts.do_fitting:
        return None, None

    qubits = validate_and_convert_qubits_sweeps(qubits)
    integration_kernels, discrimination_thresholds = {}, {}
    for q in qubits:
        # below, kernels is a list of PulseSampledComplex pulse functionals and
        # thresholds is a list of floats
        kernels, thresholds = calculate_integration_kernels_thresholds(
            truncated_time_traces[q.uid]
        )
        integration_kernels[q.uid] = [krn.samples for krn in kernels]
        discrimination_thresholds[q.uid] = thresholds

    return integration_kernels, discrimination_thresholds


@workflow.task
def filter_integration_kernels(
    qubits: Qubits,
    integration_kernels: dict[str, list[ArrayLike]],
    options: TimeTracesAnalysisOptions | None = None,
) -> dict[str, list]:
    """Applies a low-pass filter to the kernels.

    Args:
        qubits:
            The qubits on which to run the task. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the keys of the
            integration_kernels dictionary.
        integration_kernels:
            A dictionary with qubit uids as keys and the list of arrays of optimal
            integration kernels as values.
        options:
            The options for building the workflow as an instance of
            [TimeTracesAnalysisOptions]. See the docstring of this class for more
            details.

    Returns:
        a list with the arrays of filtered integration kernels

    Raises:
        ValueError:
            If the filter_cutoff_frequency is None.
    """
    opts = TimeTracesAnalysisOptions() if options is None else options
    if opts.filter_cutoff_frequency is None:
        raise ValueError("Please provide the filter_cutoff_frequency.")
    qubits = validate_and_convert_qubits_sweeps(qubits)
    integration_kernels_filtered = {q.uid: [] for q in qubits}
    for q in qubits:
        for krn in integration_kernels[q.uid]:
            poles = 5
            sos = sp.signal.butter(
                poles,
                opts.filter_cutoff_frequency,
                "lowpass",
                fs=opts.sampling_rate,
                output="sos",
            )
            integration_kernels_filtered[q.uid] += [sp.signal.sosfiltfilt(sos, krn)]

    return integration_kernels_filtered


@workflow.task
def extract_qubit_parameters(
    qubits: Qubits,
    discrimination_thresholds: dict[str, list] | None,
    integration_kernels: dict[str, list] | None,
    integration_kernels_filtered: dict[str, list] | None,
) -> dict[str, dict[str, dict[str, list]]]:
    """Extract the qubit parameters to be updated.

    These parameters are `readout_integration_kernels` and
    `readout_integration_discrimination_thresholds`.

    Arguments:
        qubits:
            The qubits on which to run the task. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the keys of the
            dictionaries discrimination_thresholds, integration_kernels and
            integration_kernels_filtered.
        discrimination_thresholds:
            The dictionary with the discrimination thresholds corresponding to the
            integration kernels, as returned by extract_kernels_thresholds.
        integration_kernels:
            The dictionary with the arrays of integration kernels for each qubit as
            returned by extract_kernels_thresholds.
        integration_kernels_filtered:
            The dictionary with the arrays of filtered integration kernels for each
            qubit as returned by filter_integration_kernels.

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
        If integration_kernels, integration_kernels_filtered, discrimination_thresholds
        are all None, then the new_parameter_values are not extracted and the function
        only returns the old_parameter_values.
    """
    qubits = validate_and_convert_qubits_sweeps(qubits)
    qubit_parameters = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
    }

    for q in qubits:
        # Store the old integration kernels and discrimination thresholds
        old_kernels = q.parameters.readout_integration_kernels
        old_kerneltype = q.parameters.readout_integration_kernels_type
        old_thresholds = q.parameters.readout_integration_discrimination_thresholds
        qubit_parameters["old_parameter_values"][q.uid] = {
            "readout_integration_kernels": old_kernels,
            "readout_integration_kernels_type": old_kerneltype,
            "readout_integration_discrimination_thresholds": old_thresholds,
        }

        # Extract and store the new integration kernels and discrimination
        # thresholds
        if integration_kernels is not None:
            qubit_parameters["new_parameter_values"][q.uid][
                "readout_integration_kernels"
            ] = [
                {"function": "sampled_pulse", "samples": krn}
                for krn in integration_kernels[q.uid]
            ]
            qubit_parameters["new_parameter_values"][q.uid][
                "readout_integration_kernels_type"
            ] = "optimal"
        if integration_kernels_filtered is not None:
            # Overwrite the integration kernels with the filtered ones
            qubit_parameters["new_parameter_values"][q.uid][
                "readout_integration_kernels"
            ] = [
                {"function": "sampled_pulse", "samples": krn}
                for krn in integration_kernels_filtered[q.uid]
            ]
            qubit_parameters["new_parameter_values"][q.uid][
                "readout_integration_kernels_type"
            ] = "optimal"
        if discrimination_thresholds is not None:
            qubit_parameters["new_parameter_values"][q.uid][
                "readout_integration_discrimination_thresholds"
            ] = discrimination_thresholds[q.uid]

    return qubit_parameters


@workflow.task
def plot_time_traces(
    qubits: Qubits,
    states: Sequence[str],
    truncated_time_traces: dict[str, list],
    options: TimeTracesAnalysisOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the time-traces plots.

    Arguments:
        qubits:
            The qubits on which to run the task. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the keys of the
            truncated_time_traces dictionary.
        states:
            The qubit basis states for which the time traces were measured. May be
            either a string, e.g. "gef", or a list of letters, e.g. ["g","e","f"].
        truncated_time_traces:
            The dictionary of truncated time traces for each qubit as returned by
            truncate_time_traces.
        options:
            The options for building the workflow as an instance of
            [TimeTracesAnalysisOptions]. See the docstring of this class for more
            details.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = TimeTracesAnalysisOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}
    for q in qubits:
        # plot traces and kernel
        fig_size = plt.rcParams["figure.figsize"]
        fig, axs = plt.subplots(
            nrows=len(states),
            sharex=True,
            figsize=(fig_size[0], fig_size[1] * 1.5),
        )
        fig.align_ylabels()
        axs[0].set_title(timestamped_title(f"Time Traces {q.uid}"))
        axs[-1].set_xlabel("Samples, $N$")
        ymax, ymin = 0, 0
        for i, s in enumerate(states):
            time_trace = truncated_time_traces[q.uid][i]
            axs[i].plot(np.real(time_trace), label=f"{s}: I")
            axs[i].plot(np.imag(time_trace), label=f"{s}: Q")
            axs[i].set_ylabel("Voltage, $V$ (a.u.)")
            axs[i].legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
            ymax = max(ymax, axs[i].get_ylim()[1])
            ymin = min(ymin, axs[i].get_ylim()[0])

        for ax in axs:
            # all panels have the same y-axis range
            ax.set_ylim(ymin, ymax)

        if opts.save_figures:
            workflow.save_artifact(f"Time_traces_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures


@workflow.task
def plot_kernels_traces(
    qubits: Qubits,
    discrimination_thresholds: dict[str, list],
    integration_kernels: dict[str, list],
    integration_kernels_filtered: dict[str, list] | None = None,
    options: TimeTracesAnalysisOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the plots of the integration kernels.

    Arguments:
        qubits:
            The qubits on which to run the task. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the keys of the
            dictionaries discrimination_thresholds, integration_kernels and
            integration_kernels_filtered.
        discrimination_thresholds:
            The dictionary with the discrimination thresholds corresponding to the
            integration kernels, as returned by extract_kernels_thresholds.
        integration_kernels:
            The dictionary with the arrays of integration kernels for each qubit as
            returned by extract_kernels_thresholds.
        integration_kernels_filtered:
            The dictionary with the arrays of filtered integration kernels for each
            qubit as returned by filter_integration_kernels. If None, only the
            integration_kernels are plotted.
        options:
            The options for building the workflow as an instance of
            [TimeTracesAnalysisOptions]. See the docstring of this class for more
            details.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = TimeTracesAnalysisOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}
    for q in qubits:
        thresholds = discrimination_thresholds[q.uid]
        kernels = integration_kernels[q.uid]
        kernels_to_plot = [kernels]
        if integration_kernels_filtered is not None:
            kernels_to_plot += [integration_kernels_filtered[q.uid]]

        # plot traces and kernel
        fig_size = plt.rcParams["figure.figsize"]
        nrows = len(kernels) * len(kernels_to_plot)
        fig, axs = plt.subplots(
            nrows=nrows,
            sharex=True,
            figsize=(fig_size[0], fig_size[1] * len(kernels_to_plot)),
        )
        if not isinstance(axs, np.ndarray):
            axs = [axs]
        fig.align_ylabels()
        axs[0].set_title(timestamped_title(f"Integration Kernels {q.uid}"))
        axs[-1].set_xlabel("Samples, $N$")

        ymax, ymin = 0, 0
        for i, krns in enumerate(kernels_to_plot):
            for ii, krn_vals in enumerate(krns):
                ax = axs[ii if len(kernels_to_plot) == 1 else i + 2 * ii]
                ax.plot(
                    np.real(krn_vals),
                    label=f"w{ii + 1} {'filt.' if i == 1 else ''}: I",
                    zorder=len(krns) + 1 - ii,
                )
                ax.plot(
                    np.imag(krn_vals),
                    label=f"w{ii + 1} {'filt.' if i == 1 else ''}: Q",
                    zorder=len(krns) + 1 - ii,
                )
                ax.text(
                    0.975,
                    0.95,
                    f"$v_{{th}}$: {thresholds[ii]:.2f}",
                    ha="right",
                    va="top",
                    transform=ax.transAxes,
                )
                ax.set_ylabel("Voltage, $V$ (a.u.)")
                ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
                ymax = max(ymax, ax.get_ylim()[1])
                ymin = min(ymin, ax.get_ylim()[0])

        for ax in axs:
            # all panels have the same y-axis range
            ax.set_ylim(ymin, ymax)

        if opts.save_figures:
            workflow.save_artifact(f"Integration_kernels_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures


@workflow.task
def plot_kernels_fft(
    qubits: Qubits,
    discrimination_thresholds: dict[str, list],
    integration_kernels: dict[str, list],
    integration_kernels_filtered: dict[str, list] | None = None,
    options: TimeTracesAnalysisOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the plots of the FFT of the integration kernels.

    Arguments:
        qubits:
            The qubits on which to run the task. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the keys of the
            dictionaries discrimination_thresholds, integration_kernels and
            integration_kernels_filtered.
        discrimination_thresholds:
            The dictionary with the discrimination thresholds corresponding to the
            integration kernels, as returned by extract_kernels_thresholds.
        integration_kernels:
            The dictionary with the arrays of integration kernels for each qubit as
            returned by extract_kernels_thresholds.
        integration_kernels_filtered:
            The dictionary with the arrays of filtered integration kernels for each
            qubit as returned by filter_integration_kernels. If None, only the
            integration_kernels are plotted.
        options:
            The options for building the workflow as an instance of
            [TimeTracesAnalysisOptions]. See the docstring of this class for more
            details.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = TimeTracesAnalysisOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}
    for q in qubits:
        fig, axs = plt.subplots(nrows=len(integration_kernels[q.uid]), sharex=True)
        if not isinstance(axs, np.ndarray):
            axs = [axs]

        for i, ax in enumerate(axs):
            y = integration_kernels[q.uid][i]
            n = len(y)
            t = 0.5e-9
            yf = sp.fft.fft(y)
            xf = sp.fft.fftfreq(n, t)
            xf = sp.fft.fftshift(xf)
            yplot = sp.fft.fftshift(yf)
            ax.semilogy(
                xf / 1e6, 1.0 / n * np.abs(yplot), label=f"w{i + 1}: unfiltered"
            )

            if integration_kernels_filtered is not None:
                if opts.filter_cutoff_frequency is None:
                    raise ValueError("Please provide the filter_cutoff_frequency.")
                y = integration_kernels_filtered[q.uid][i]
                n = len(y)
                t = 0.5e-9
                yf = sp.fft.fft(y)
                xf = sp.fft.fftfreq(n, t)
                xf = sp.fft.fftshift(xf)
                yplot = sp.fft.fftshift(yf)
                ax.semilogy(
                    xf / 1e6,
                    1.0 / n * np.abs(yplot),
                    label=f"LPF: {opts.filter_cutoff_frequency / 1e6} MHz",
                )
                ax.text(
                    0.025,
                    0.95,
                    f"$v_{{th}}$: {discrimination_thresholds[q.uid][i]:.2f}",
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                )

        axs[-1].set_xlabel("Readout IF Frequency, $f_{IF}$ (MHz)")
        axs[0].set_ylabel("FFT")
        axs[-1].set_ylabel("FFT")
        axs[0].legend(frameon=False)
        axs[-1].legend(frameon=False)
        axs[0].set_title(timestamped_title(f"Integration Kernels {q.uid}"))
        fig.subplots_adjust(hspace=0.1)

        if opts.save_figures:
            workflow.save_artifact(f"Integration_kernels_fft_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures
