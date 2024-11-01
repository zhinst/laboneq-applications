"""This module contains helper function for experiment analyses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from laboneq import workflow
from laboneq.data.experiment_results import AcquiredResult as AcquiredResultLegacy
from laboneq.workflow import (
    option_field,
    options,
)
from laboneq.workflow.options import TaskOptions
from laboneq.workflow.tasks.run_experiment import (
    AcquiredResult as AcquiredResultRunExp,
)
from laboneq.workflow.timestamps import local_timestamp

from laboneq_applications import dsl
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_result,
)

if TYPE_CHECKING:
    from datetime import datetime

    from laboneq.simple import Results
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import Qubits, QubitSweepPoints


@options
class PlotRawDataOptions(TaskOptions):
    """Options for the plot_raw_complex_data_1d task.

    Attributes:
        use_cal_traces:
            Whether to include calibration traces in the experiment.
            Default: `True`.
        cal_states:
            The states to prepare in the calibration traces. Can be any
            string or tuple made from combining the characters 'g', 'e', 'f'.
            Default: same as transition
        save_figures:
            Whether to save the figures.
            Default: `True`.
        close_figures:
            Whether to close the figures.
            Default: `True`.
    """

    use_cal_traces: bool = option_field(
        True, description="Whether to include calibration traces in the experiment."
    )
    cal_states: str | tuple = option_field(
        "ge",
        description="The states to prepare in the calibration traces."
        "Can be any string or tuple made from combining the characters 'g', 'e', 'f'.",
    )
    save_figures: bool = option_field(True, description="Whether to save the figures.")
    close_figures: bool = option_field(
        True, description="Whether to close the figures."
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


@workflow.task
def plot_raw_complex_data_1d(
    qubits: Qubits,
    result: RunExperimentResults | tuple[RunExperimentResults, Results],
    sweep_points: QubitSweepPoints,
    xlabel: str,
    xscaling: float = 1.0,
    options: PlotRawDataOptions | None = None,
) -> dict[str, dict[str, ArrayLike]]:
    """Creates plots of raw complex data acquired in integration mode.

    Arguments:
        qubits:
            The qubits on which the amplitude-Rabi experiments was run. May be either
            a single qubit or a list of qubits.
        result: the result of the experiment, returned by the run_experiment task.
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
    validate_result(result)
    qubits, sweep_points = validate_and_convert_qubits_sweeps(qubits, sweep_points)
    figures = {}
    for q, swpts in zip(qubits, sweep_points):
        figsize = plt.rcParams["figure.figsize"]
        fig, axs = plt.subplots(
            nrows=2, figsize=[0.75 * figsize[0], 1.5 * figsize[1]], sharex=True
        )
        fig.align_ylabels()
        fig.subplots_adjust(hspace=0.1)
        axs[0].set_title(timestamped_title(f"Raw data {q.uid}"))
        axs[1].set_xlabel(xlabel)

        res_handle = dsl.handles.result_handle(q.uid)
        if isinstance(
            result[res_handle],
            (AcquiredResultLegacy, AcquiredResultRunExp),
        ):
            raw_data_collection = [("raw data", result[res_handle])]
        else:
            raw_data_collection = [
                (k, result[res_handle][k]) for k in result[res_handle]
            ]

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
            and dsl.handles.calibration_trace_handle(q.uid) in result
        ):
            for i, ax in enumerate(axs):
                for j, cs in enumerate(opts.cal_states):
                    ct_handle = dsl.handles.calibration_trace_handle(q.uid, cs)
                    cal_trace = (
                        result[ct_handle].data.real
                        if i == 0
                        else result[ct_handle].data.imag
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
