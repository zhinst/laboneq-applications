# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module contains helper function for experiment analyses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.data.experiment_results import AcquiredResult as AcquiredResultLegacy
from laboneq.simple import Results, dsl
from laboneq.workflow.tasks.run_experiment import (
    AcquiredResult as AcquiredResultRunExp,
)
from laboneq.workflow.tasks.run_experiment import AttributeWrapper, RunExperimentResults
from laboneq.workflow.timestamps import local_timestamp

from laboneq_applications.analysis.options import BasePlottingOptions
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_result,
)

if TYPE_CHECKING:
    from datetime import datetime

    import matplotlib as mpl
    from laboneq.simple import QuantumElement
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.options
class PlotRawDataOptions(workflow.TaskOptions):
    """Options for the plot_raw_complex_data_1d task.

    Attributes:
        use_cal_traces:
            Whether to plot the calibration traces.
            Note that in the case of 2D data, the calibration traces are included
            in the 2D plot together with the main data. To achieve this, the
            1D sweep points are extended by num_cal_traces * [sp_1d[-2] - sp_1d[-1]].
            Default: `True`.
        cal_states:
            The states to prepare in the calibration traces. Can be any
            string or tuple made from combining the characters 'g', 'e', 'f'.
            Default: same as transition
        figure_size_raw_data:
            The size of the figure.
        save_figures:
            Whether to save the figures.
            Default: `True`.
        close_figures:
            Whether to close the figures.
            Default: `True`.
    """

    use_cal_traces: bool = workflow.option_field(
        True, description="Whether to include calibration traces in the experiment."
    )
    cal_states: str | tuple = workflow.option_field(
        "ge",
        description="The states to prepare in the calibration traces."
        "Can be any string or tuple made from combining the characters 'g', 'e', 'f'.",
    )
    figure_size_raw_data: Sequence | None = workflow.option_field(
        None, description="The size of the figure."
    )
    save_figures: bool = workflow.option_field(
        True, description="Whether to save the figures."
    )
    close_figures: bool = workflow.option_field(
        True, description="Whether to close the figures."
    )


@workflow.task_options(base_class=BasePlottingOptions)
class PlotSignalMagnitudeAndPhase2DOptions:
    """Options for the plot_signal_magnitude_and_phase_2d task.

    Attributes:
        figure_size_magnitude_phase:
            The size of the figure.
    """

    figure_size_magnitude_phase: Sequence | None = workflow.option_field(
        None, description="The size of the figure."
    )


def timestamped_title(title: str, dt: datetime | None = None) -> str:
    """Return a plot title with a timestamp in the local timezone.

    Arguments:
        title:
            The title of the plot without a timestamp.
        dt:
            The time to use to create the timestamp. If None,
            the workflow start time is used. If there is no
            active workflow, the current time is used.

    Note:
        The timestamp is generated using the function `local_timestamp`
        and thus has the same format as the timestamps used by the
        `FolderStore` in the logbook folders it creates.

    Returns:
        The title with a timestamp, formatted as "TIMESTAMP - TITLE".
    """
    if dt is None:
        wf_info = workflow.execution_info()
        if wf_info is not None:
            dt = wf_info.start_time

    return f"{local_timestamp(dt)} - {title}"


def _get_raw_data_collection(
    result: RunExperimentResults, qubit: QuantumElement
) -> list[tuple[None | str, AcquiredResultLegacy | AcquiredResultRunExp]]:
    """Collect the measured data that can be either nested or not.

    Args:
        result: the result as an instance of RunExperimentResults.
        qubit: the qubit for which the data was measured.

    Returns:
        a list of tuples with the nested results key (the one that comes after
        "q.uid/result") and the corresponding acquired results.
        If the data does not contain any nested keys, the first entry in the tuple is
        None.
    """
    validate_result(result)
    res_handle = dsl.handles.result_handle(qubit.uid)
    if isinstance(
        result[res_handle],
        (AcquiredResultLegacy, AcquiredResultRunExp),
    ):
        raw_data_collection = [(None, result[res_handle])]
    elif isinstance(result[res_handle], AttributeWrapper):
        raw_data_collection = [(k, result[res_handle][k]) for k in result[res_handle]]
    else:
        raise TypeError(
            f"The result for the handle qubit {res_handle} has an unsupported type. "
            f"Only the following types are supported: "
            f"`laboneq.data.experiment_results.AcquiredResult`, "
            f"`laboneq.workflow.tasks.run_experiment.AcquiredResult`, "
            f"or a `dict` with values with those two types."
        )

    return raw_data_collection


@workflow.task
def plot_raw_complex_data_1d(
    qubits: QuantumElements,
    result: RunExperimentResults | Results,
    sweep_points: QubitSweepPoints,
    xlabel: str,
    xscaling: float = 1.0,
    options: PlotRawDataOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Creates plots of raw complex data acquired in integration mode.

    Arguments:
        qubits:
            The qubits on which to run this task. May be either a single qubit or a
            list of qubits.
        result: the result of the experiment. Can be either an instance of
            `RunExperimentResults` (returned by the `run_experiment` task), or an
            instance of `Results` (returned by `session.run()`).
        sweep_points:
            The sweep points used in the experiment for each qubit.
            If `qubits` is a single qubit, `sweep_points` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        xlabel: x-axis label
        xscaling: value by which to scale the sweep_points
        options:
            The options for processing the raw data as an instance of
            [PlotRawDataOptions].

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as keys.
    """
    opts = PlotRawDataOptions() if options is None else options
    qubits, sweep_points = validate_and_convert_qubits_sweeps(qubits, sweep_points)
    raw_result = result
    if isinstance(result, Results):
        raw_result = RunExperimentResults(data=result.acquired_results)

    figure_size_raw_data = opts.figure_size_raw_data
    if figure_size_raw_data is None:
        figsize_default = plt.rcParams["figure.figsize"]
        figure_size_raw_data = [0.75 * figsize_default[0], 1.5 * figsize_default[1]]
    figures = {}
    for q, swpts in zip(qubits, sweep_points):
        fig, axs = plt.subplots(nrows=2, figsize=figure_size_raw_data, sharex=True)
        fig.align_labels()
        fig.subplots_adjust(hspace=0.1)
        axs[0].set_title(timestamped_title(f"Raw data {q.uid}"))
        axs[1].set_xlabel(xlabel)

        raw_data_collection = _get_raw_data_collection(raw_result, q)
        for legend_name, acquired_results in raw_data_collection:
            raw_data = acquired_results.data
            # plot real
            axs[0].plot(swpts * xscaling, raw_data.real, "o-", label=legend_name)
            axs[0].set_ylabel("Real (arb.)")
            # plot imaginary
            axs[1].plot(swpts * xscaling, raw_data.imag, "o-")
            axs[1].set_ylabel("Imaginary (arb.)")

        # plot lines at calibration traces
        if (
            opts.use_cal_traces
            and dsl.handles.calibration_trace_handle(q.uid) in raw_result
        ):
            for j, cs in enumerate(opts.cal_states):
                for i, ax in enumerate(axs):
                    ct_handle = dsl.handles.calibration_trace_handle(q.uid, cs)
                    cal_trace = (
                        raw_result[ct_handle].data.real
                        if i == 0
                        else raw_result[ct_handle].data.imag
                    )
                    xlims = ax.get_xlim()
                    ax.hlines(
                        cal_trace,
                        *xlims,
                        linestyles="--",
                        colors="gray",
                        zorder=0,
                        label="calib.\ntraces" if i == 0 and j == 0 else None,
                    )
                    ax.text(
                        max(swpts * xscaling),
                        cal_trace,
                        f"Prep. $|{cs}\\rangle$",
                        ha="right",
                        va="bottom",
                        transform=ax.transData,
                    )
                    ax.set_xlim(xlims)

        # Add legend
        axs[0].legend(
            loc="center left",
            bbox_to_anchor=(1, 0),
            handlelength=1.5,
            frameon=False,
        )

        if opts.save_figures:
            workflow.save_artifact(f"Raw_data_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures


def _prepare_sweep_points_with_cal_traces_2d(
    qubit: QuantumElement,
    data: ArrayLike,
    result: RunExperimentResults,
    sweep_points_1d: ArrayLike,
    sweep_points_2d: ArrayLike,
    plot_cal_traces: bool = True,  # noqa: FBT001, FBT002
    cal_states: str = "ge",
) -> tuple[ArrayLike, ArrayLike]:
    """Checks whether to plot calibration traces and prepares the data to plot.

    This function first checks whether calibration traces exist in the result and then
    whether to plot them (`plot_cal_traces==True`).

    If the calibration traces exist in result, and they are to be plotted
    (`plot_cal_traces==True`), this function adds the calibration traces to the data
    array and extends the sweep_points_1d by num_cal_traces * [sp_1d[-2] - sp_1d[-1]],
    such that the extended data array can be plotted against the sweep points.

    Args:
        qubit:
            The qubit on which to run this task.
        data:
            The 2D array of the data to be plotted.
        result:
            The result object as an instance of `RunExperimentResults`
            (returned by the `run_experiment` task) that contains the data.
        sweep_points_1d:
            The sweep points corresponding to the innermost sweep.
            If `qubits` is a single qubit, `sweep_points_1d` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        sweep_points_2d:
            The sweep points corresponding to the outermost sweep.
            If `qubits` is a single qubit, `sweep_points_2d` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        plot_cal_traces:
            Whether to add the calibration traces to the plot.
        cal_states:
            The calibration states as either "ge" (default) or "ef".

    Returns:
        A tuple with the new data and sweep_points_1d arrays extended to include the
        calibration traces.
    """
    sp_1d_to_plot = sweep_points_1d
    if plot_cal_traces and dsl.handles.calibration_trace_handle(qubit.uid) in result:
        # add the calibration traces to the data array to be plotted
        for cs in cal_states:
            ct_handle = dsl.handles.calibration_trace_handle(qubit.uid, cs)
            ct_data = result[ct_handle].data
            if len(ct_data.shape) > 1:
                raise ValueError("ct_data must be a number or a 1D array.")
            if not isinstance(ct_data, (Sequence, np.ndarray)) or len(ct_data) == 1:
                ct_data = np.repeat(ct_data, len(sweep_points_2d))
            if len(ct_data) != data.shape[0]:
                raise ValueError(
                    "The number of data points for each calibration state must "
                    "equal the number of 2D sweep points. Please set "
                    "`options.use_cal_traces` to `False` for the "
                    "`plot_raw_complex_data_2d` task."
                )
            data = np.concatenate([data, ct_data[:, np.newaxis]], axis=1)
        # We artificially extend the 1d sweep points to the right in order
        # to plot the calibration traces data
        sp_cal_traces = np.array(
            [
                sweep_points_1d[-1]
                + (i + 1) * (sweep_points_1d[-1] - sweep_points_1d[-2])
                for i in range(len(cal_states))
            ]
        )
        sp_1d_to_plot = np.concatenate([sweep_points_1d, sp_cal_traces])

    return data, sp_1d_to_plot


@workflow.task
def plot_raw_complex_data_2d(
    qubits: QuantumElements,
    result: RunExperimentResults | Results,
    sweep_points_1d: QubitSweepPoints,
    sweep_points_2d: QubitSweepPoints,
    label_sweep_points_1d: str,
    label_sweep_points_2d: str,
    scaling_sweep_points_1d: float = 1.0,
    scaling_sweep_points_2d: float = 1.0,
    options: PlotRawDataOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Creates plots of two-dimensional raw complex data acquired in integration mode.

    Arguments:
        qubits:
            The qubits on which to run this task. May be either a single qubit or a
            list of qubits.
        result: the result of the experiment. Can be either an instance of
            `RunExperimentResults` (returned by the `run_experiment` task), or an
            instance of `Results` (returned by `session.run()`).
        sweep_points_1d:
            The sweep points corresponding to the innermost sweep.
            If `qubits` is a single qubit, `sweep_points_1d` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        sweep_points_2d:
            The sweep points corresponding to the outermost sweep.
            If `qubits` is a single qubit, `sweep_points_2d` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        label_sweep_points_1d:
            The label that will appear on the axis of the 1D sweep points.
            Passed to the `label_x_values` parameter of plot_data_2d.
        label_sweep_points_2d:
            The label that will appear on the axis of the 2D sweep points.
            Passed to the `label_y_values` parameter of plot_data_2d.
        scaling_sweep_points_1d:
            The scaling factor of the 1D sweep points.
            Passed to the `scaling_x_values` parameter of plot_data_2d.
            Default: 1.0.
        scaling_sweep_points_2d:
            The scaling factor of the 2D sweep points.
            Passed to the `scaling_y_values` parameter of plot_data_2d.
            Default: 1.0.
        options:
            The options for this task as an instance of [PlotRawDataOptions]. See the
            docstring of this class for more details.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as keys.
    """
    opts = PlotRawDataOptions() if options is None else options
    _, sweep_points_1d = validate_and_convert_qubits_sweeps(qubits, sweep_points_1d)
    qubits, sweep_points_2d = validate_and_convert_qubits_sweeps(
        qubits, sweep_points_2d
    )
    raw_result = result
    if isinstance(result, Results):
        raw_result = RunExperimentResults(data=result.acquired_results)

    figures = {}
    figure_size_raw_data = opts.figure_size_raw_data
    if figure_size_raw_data is None:
        figure_size_raw_data = plt.rcParams["figure.figsize"]
    for q, sp_1d, sp_2d in zip(qubits, sweep_points_1d, sweep_points_2d):
        raw_data_collection = _get_raw_data_collection(raw_result, q)
        figures[q.uid] = {}
        for k, acquired_results in raw_data_collection:
            # generate and configure figure
            plot_identifier = q.uid if k is None else f"{q.uid}_{k}"
            fig, axs = plt.subplots(
                nrows=2,
                figsize=figure_size_raw_data,
                sharex=True,
                constrained_layout=True,
            )
            fig.align_labels()
            fig.subplots_adjust(hspace=0.1)
            axs[0].set_title(timestamped_title(f"Raw data {plot_identifier}"))
            axs[1].set_xlabel(label_sweep_points_1d)

            # get raw data and validate it
            raw_data = acquired_results.data
            if len(raw_data.shape) != 2:  # noqa: PLR2004
                raise ValueError(f"The raw data for qubit {q.uid} is not a 2D array.")
            if raw_data.shape[0] != len(sp_2d) and raw_data.shape[1] != len(sp_1d):
                raise ValueError(
                    f"The raw data for {q.uid} does not have the shape "
                    "(len(sweep_points_2d), len(sweep_points_1d))."
                )

            # check if there are cal traces and prepare for plotting them
            raw_data, sp_1d_to_plot = _prepare_sweep_points_with_cal_traces_2d(
                qubit=q,
                data=raw_data,
                result=raw_result,
                sweep_points_1d=sp_1d,
                sweep_points_2d=sp_2d,
                plot_cal_traces=opts.use_cal_traces,
                cal_states=opts.cal_states,
            )

            # plot real part
            fig, axs[0] = plot_data_2d(
                x_values=sp_1d_to_plot,
                y_values=sp_2d,
                z_values=raw_data.real,
                label_y_values=label_sweep_points_2d,
                label_z_values="Real (a.u.)",
                scaling_x_values=scaling_sweep_points_1d,
                scaling_y_values=scaling_sweep_points_2d,
                figure=fig,
                axis=axs[0],
                close_figures=opts.close_figures,
            )
            # plot imaginary part
            fig, axs[1] = plot_data_2d(
                x_values=sp_1d_to_plot,
                y_values=sp_2d,
                z_values=raw_data.imag,
                label_y_values=label_sweep_points_2d,
                label_z_values="Imaginary (a.u.)",
                scaling_x_values=scaling_sweep_points_1d,
                scaling_y_values=scaling_sweep_points_2d,
                figure=fig,
                axis=axs[1],
                close_figures=opts.close_figures,
            )

            if opts.save_figures:
                workflow.save_artifact(f"Raw_data_{plot_identifier}", fig)

            if opts.close_figures:
                plt.close(fig)

            if k is None:
                figures[q.uid] = fig
            else:
                figures[q.uid][k] = fig

    return figures


@workflow.task
def plot_signal_magnitude_and_phase_2d(
    qubits: QuantumElements,
    result: RunExperimentResults | Results,
    sweep_points_1d: QubitSweepPoints,
    sweep_points_2d: QubitSweepPoints,
    label_sweep_points_1d: str,
    label_sweep_points_2d: str,
    scaling_sweep_points_1d: float = 1.0,
    scaling_sweep_points_2d: float = 1.0,
    options: PlotSignalMagnitudeAndPhase2DOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the qubit-spectroscopy plots.

    Arguments:
        qubits:
            The qubits on which to run this task. May be either a single qubit or a
            list of qubits.
        result: the result of the experiment. Can be either an instance of
            `RunExperimentResults` (returned by the `run_experiment` task), or an
            instance of `Results` (returned by `session.run()`).
        sweep_points_1d:
            The sweep points corresponding to the innermost sweep.
            If `qubits` is a single qubit, `sweep_points_1d` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        sweep_points_2d:
            The sweep points corresponding to the outermost sweep.
            If `qubits` is a single qubit, `sweep_points_2d` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        label_sweep_points_1d:
            The label that will appear on the axis of the 1D sweep points.
            Passed to the `label_x_values` parameter of plot_data_2d.
        label_sweep_points_2d:
            The label that will appear on the axis of the 2D sweep points.
            Passed to the `label_y_values` parameter of plot_data_2d.
        scaling_sweep_points_1d:
            The scaling factor of the 1D sweep points.
            Passed to the `scaling_x_values` parameter of plot_data_2d.
            Default: 1.0.
        scaling_sweep_points_2d:
            The scaling factor of the 2D sweep points.
            Passed to the `scaling_y_values` parameter of plot_data_2d.
            Default: 1.0.
        options:
            The options for this task as an instance of
            [PlotSignalMagnitudeAndPhase2DOptions]. See the docstring of this class
            for more details.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = PlotSignalMagnitudeAndPhase2DOptions() if options is None else options
    _, sweep_points_1d = validate_and_convert_qubits_sweeps(qubits, sweep_points_1d)
    qubits, sweep_points_2d = validate_and_convert_qubits_sweeps(
        qubits, sweep_points_2d
    )
    raw_result = result
    if isinstance(result, Results):
        raw_result = RunExperimentResults(result.acquired_results)

    figures = {}
    figure_size = opts.figure_size_magnitude_phase
    if figure_size is None:
        figure_size = plt.rcParams["figure.figsize"]
    for q, sp_1d, sp_2d in zip(qubits, sweep_points_1d, sweep_points_2d):
        raw_data_collection = _get_raw_data_collection(raw_result, q)

        figures[q.uid] = {}
        for k, acquired_results in raw_data_collection:
            plot_identifier = q.uid if k is None else f"{q.uid}_{k}"
            fig, axs = plt.subplots(
                nrows=2,
                figsize=figure_size,
                sharex=True,
                constrained_layout=True,
            )
            fig.align_labels()
            fig.subplots_adjust(hspace=0.1)
            axs[0].set_title(timestamped_title(f"Magnitude_Phase_{plot_identifier}"))
            axs[1].set_xlabel(label_sweep_points_1d)

            raw_data = acquired_results.data
            sp_1d_to_plot = sp_1d
            if len(raw_data.shape) != 2:  # noqa: PLR2004
                raise ValueError(f"The raw data for {q.uid} is not a 2D array.")
            if raw_data.shape[0] != len(sp_2d) and raw_data.shape[1] != len(sp_1d):
                raise ValueError(
                    f"The raw data for {q.uid} does not have the shape "
                    "(len(sweep_points_2d), len(sweep_points_1d))."
                )

            # plot magnitude
            fig, axs[0] = plot_data_2d(
                x_values=sp_1d_to_plot,
                y_values=sp_2d,
                z_values=np.abs(raw_data),
                label_y_values=label_sweep_points_2d,
                label_z_values="Transmission Signal\nMagnitude, $|S_{21}|$ (a.u.)",
                scaling_x_values=scaling_sweep_points_1d,
                scaling_y_values=scaling_sweep_points_2d,
                figure=fig,
                axis=axs[0],
                close_figures=opts.close_figures,
            )
            # plot phase
            fig, axs[1] = plot_data_2d(
                x_values=sp_1d_to_plot,
                y_values=sp_2d,
                z_values=np.angle(raw_data),
                label_y_values=label_sweep_points_2d,
                label_z_values="Transmission Signal\nPhase, $|S_{21}|$ (a.u.)",
                scaling_x_values=scaling_sweep_points_1d,
                scaling_y_values=scaling_sweep_points_2d,
                figure=fig,
                axis=axs[1],
                close_figures=opts.close_figures,
            )

            if opts.save_figures:
                workflow.save_artifact(f"Magnitude_Phase_{plot_identifier}", fig)

            if opts.close_figures:
                plt.close(fig)

            if k is None:
                figures[q.uid] = fig
            else:
                figures[q.uid][k] = fig

    return figures


@workflow.task
def plot_data_2d(  # noqa: C901, PLR0913
    x_values: ArrayLike,
    y_values: ArrayLike,
    z_values: ArrayLike,
    label_x_values: str = "",
    label_y_values: str = "",
    label_z_values: str = "",
    scaling_x_values: float = 1.0,
    scaling_y_values: float = 1.0,
    plot_title: str = "",
    figure_name: str = "2D_Data",
    save_figures: bool = False,  # noqa: FBT001, FBT002
    close_figures: bool = True,  # noqa: FBT001, FBT002
    figure: mpl.figure.Figure = None,
    axis: mpl.axes.Axes = None,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Create a 2D plot using pcolormesh.

    Arguments:
        x_values:
            The 1D array to plot on the x-axis.
        y_values:
            The 1D array to plot on the y-axis.
        z_values:
            The 2D array to plot on the z-axis. The shape of z_values must be
            (len(y_values), len(x_values)).
        label_x_values:
            The label that will appear on the x-axis.
            Default: ""
        label_y_values:
            The label that will appear on the y-axis.
            Default: ""
        label_z_values:
            The label that will appear on the colour bar of the z-axis.
            Default: ""
        scaling_x_values:
            The scaling factor of the x-values.
            Default: 1.0.
        scaling_y_values:
            The scaling factor of the y-values.
            Default: 1.0.
        plot_title:
            The plot title.
            Default: ""
        figure_name:
            The figure name.
            Default: "2D_Data".
        save_figures:
            Whether to save the figures.
            Default: `False`.
        close_figures:
            Whether to close the figures.
            Default: `True`.
        figure:
            The matplotlib figure on which to plot.
            Default: None.
        axis:
            The matplotlib axis on which to plot.
            Default: None.

    Raises:
        ValueError: If the x_values is not a 1D array.
        ValueError: If the y_values is not a 1D array.
        ValueError: If the z_values is not a 2D array.
        ValueError: If the z_values does not have the shape
            (len(y_values), len(x_values)).

    Returns:
        The matplotlib figure and axis.
    """
    if len(x_values.shape) > 1:
        raise ValueError("x_values must be a 1D array.")
    if len(y_values.shape) > 1:
        raise ValueError("y_values must be a 1D array.")
    if len(z_values.shape) != 2:  # noqa: PLR2004
        raise ValueError("z_values must be a 2D array.")
    if z_values.shape[0] != len(y_values) and z_values.shape[1] != len(x_values):
        raise ValueError("z_values must have the shape (len(y_values), len(x_values)).")

    x_values, y_values, z_values = sorted_mesh(
        x_values * scaling_x_values,
        y_values * scaling_y_values,
        z_values,
    )

    if figure is None and axis is None:
        figure, axis = plt.subplots(constrained_layout=True)
    mesh = axis.pcolormesh(x_values, y_values, z_values, cmap="magma")
    cbar = figure.colorbar(mesh)
    if len(label_z_values) > 0:
        cbar.set_label(label_z_values)
    if len(label_x_values) > 0:
        axis.set_xlabel(label_x_values)
    if len(label_y_values) > 0:
        axis.set_ylabel(label_y_values)
    if len(plot_title) > 0:
        axis.set_title(timestamped_title(plot_title))

    if save_figures:
        workflow.save_artifact(figure_name, figure)

    if close_figures:
        plt.close(figure)

    return figure, axis


@workflow.task
def sorted_mesh(
    x_values: ArrayLike, y_values: ArrayLike, z_values: ArrayLike
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Prepare the x, y, z arrays to be plotted with matplotlib pcolormesh.

    Ensures that the z values are sorted according to the values in x_values and
    y_values and creates np.meshgrid from x_values and y_values.

    Args:
        x_values: array of the values to be plotted on the x-axis: typically the
            real-time sweep points.
        y_values: array of the values to be plotted on the y-axis: typically the
            near-time sweep points.
        z_values: array of the values to be plotted on the z-axis: typically the data.

    Returns:
        the x, y, and z values to be passed directly to pcolormesh.
    """
    # First, we need to sort the data as otherwise we get odd plotting
    # artefacts. An example is e.g., plotting a fourier transform
    sorted_x_arguments = x_values.argsort()
    x_values = x_values[sorted_x_arguments]
    sorted_y_arguments = y_values.argsort()
    y_values = y_values[sorted_y_arguments]
    z_values_srt = z_values[:, sorted_x_arguments]
    z_values_srt = z_values_srt[sorted_y_arguments, :]

    xgrid, ygrid = np.meshgrid(x_values, y_values)

    return xgrid, ygrid, z_values_srt
