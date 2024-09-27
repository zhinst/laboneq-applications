"""This module defines the analysis for a time rabi chevron experiment.

The experiment is defined in laboneq_applications.experiments.

In this analysis, we first interpret the raw data into qubit populations using
principal component analysis or rotation and projection on the measured calibration
states. Then we plot the data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from laboneq_applications.contrib.analysis.amplitude_rabi_chevron import (
    calculate_qubit_population_2d,
)
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.experiments.options import (
    TuneupAnalysisOptions,
    TuneUpAnalysisWorkflowOptions,
)
from laboneq_applications.workflow import (
    if_,
    save_artifact,
    task,
    workflow,
)

if TYPE_CHECKING:
    import matplotlib as mpl
    from numpy.typing import ArrayLike

    from laboneq_applications.tasks.run_experiment import RunExperimentResults
    from laboneq_applications.typing import Qubits, QubitSweepPoints


@workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: Qubits,
    frequencies: QubitSweepPoints,
    lengths: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The Time Rabi analysis Workflow.

    The workflow consists of the following steps:

    - [calculate_qubit_population_2d]()
    - [plot_population]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        frequencies:
            The qubit frequencies to sweep over for the qubit drive pulse. If `qubits`
            is a single qubit, `frequencies` must be a list of numbers or an array.
            Otherwise, it must be a list of lists of numbers or arrays.
        lengths:
            The lengths to sweep over for each qubit drive pulse.  `lengths` must
            be a list of numbers or an array. Otherwise it must be a list of lists of
             numbers or arrays.
        options:
            The options for building the workflow, passed as an instance of
            [TuneUpAnalysisWorkflowOptions].

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        options = analysis_workflow.options()
        result = analysis_workflow(
            results=results
            qubits=[q0, q1],
            frequencies=[
                np.linspace(1.5e9, 2.0e9, 11),
                np.linsapce(1.5e9, 2.0e9, 11),
            ],
            lengths=[
                np.linspace(10e-9, 100e-9, 11),
                np.linspace(10e-9, 100e-9, 11),
            ],
            options=options,
        ).run()
        ```
    """
    processed_data_dict = calculate_qubit_population_2d(
        qubits, result, frequencies, lengths
    )
    with if_(options.do_plotting):
        with if_(options.do_qubit_population_plotting):
            plot_population(qubits, processed_data_dict)


@task
def plot_population(
    qubits: Qubits,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the time-Rabi plots.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict, fit_results and qubit_parameters.
        processed_data_dict: the processed data dictionary returned by process_raw_data
        options:
            The options for processing the raw data.
            See [TuneupAnalysisOptions], [TuneupExperimentOptions] and
            [BaseExperimentOptions] for accepted options.
            Overwrites the options from [TuneupAnalysisOptions],
            [TuneupExperimentOptions] and [BaseExperimentOptions].

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = TuneupAnalysisOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}
    for q in qubits:
        sweep_points_slow = 1e-9 * processed_data_dict[q.uid]["sweep_points_slow"]
        sweep_points_fast = 1e9 * processed_data_dict[q.uid]["sweep_points_fast"]
        data = processed_data_dict[q.uid]["population"]
        num_cal_traces = processed_data_dict[q.uid]["num_cal_traces"]

        fig, ax = plt.subplots()
        ax.set_title(f"Time Rabi Chevron {q.uid}")
        ax.set_xlabel("Drive Frequency, (GHz)")
        ax.set_ylabel("Drive Pulse Length, (ns)")
        x, y = np.meshgrid(sweep_points_slow, sweep_points_fast)
        im = ax.pcolormesh(x, y, data, cmap="viridis")
        colorbar_label = (
            "Principal Component (a.u)"
            if (num_cal_traces == 0 or opts.do_pca)
            else f"$|{opts.cal_states[-1]}\\rangle$-State Population"
        )
        fig.colorbar(im, orientation="vertical", label=colorbar_label)

        if opts.save_figures:
            save_artifact(f"Time_Rabi_Chevron_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures
