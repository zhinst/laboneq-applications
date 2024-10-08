"""This module contains helper function for experiment analyses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from laboneq_applications.core.utils import local_timestamp
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.workflow import (
    execution_info,
    save_artifact,
    task,
)
from laboneq_applications.workflow.options import TaskOptions

if TYPE_CHECKING:
    from datetime import datetime

    from laboneq.simple import Results
    from numpy.typing import ArrayLike

    from laboneq_applications.tasks.run_experiment import RunExperimentResults
    from laboneq_applications.typing import Qubits, QubitSweepPoints


class PlotRawDataOptions(TaskOptions):
    """Options for the plot_raw_complex_data_1d taks.

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

    use_cal_traces: bool = False
    cal_states: str | tuple = "ge"
    save_figures: bool = True
    close_figures: bool = True


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
        wf_info = execution_info()
        if wf_info is not None:
            dt = wf_info.start_time

    return f"{local_timestamp(dt)} - {title}"


@task
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
    qubits, sweep_points = validate_and_convert_qubits_sweeps(qubits, sweep_points)
    figures = {}
    for q, swpts in zip(qubits, sweep_points):
        raw_data = result.result[q.uid].data
        figsize = plt.rcParams["figure.figsize"]
        fig, axs = plt.subplots(
            nrows=2, figsize=[0.75 * figsize[0], 1.5 * figsize[1]], sharex=True
        )
        axs[0].set_title(f"Raw data {q.uid}")  # add timestamp here
        axs[1].set_xlabel(xlabel)

        # plot real
        axs[0].plot(swpts * xscaling, raw_data.real, "o-")
        axs[0].set_ylabel("Real (arb.)")
        # plot imaginary
        axs[1].plot(swpts * xscaling, raw_data.imag, "o-")
        axs[1].set_ylabel("Imaginary (arb.)")
        fig.align_ylabels()
        fig.subplots_adjust(hspace=0.1)

        # plot lines at calibration traces
        if opts.use_cal_traces and "cal_trace" in result:
            for i, ax in enumerate(axs):
                for cs in opts.cal_states:
                    cal_trace = (
                        result.cal_trace[q.uid][cs].data.real
                        if i == 0
                        else result.cal_trace[q.uid][cs].data.imag
                    )
                    xlims = ax.get_xlim()
                    ax.hlines(
                        cal_trace,
                        *xlims,
                        linestyles="--",
                        colors="gray",
                        zorder=0,
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

        if opts.save_figures:
            save_artifact(f"Raw_data_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures
