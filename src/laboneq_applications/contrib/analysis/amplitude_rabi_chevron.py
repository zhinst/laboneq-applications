"""This module defines the analysis for a amplitude Rabi chevron experiment.

The experiment is defined in laboneq_applications.contrib.experiments.

In this analysis, we first interpret the raw data into qubit populations using
principal component analysis or rotation and project onto the measured calibration
states. Then we plot the data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from laboneq_applications.analysis.cal_trace_rotation import calculate_population_1d
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

options = TuneUpAnalysisWorkflowOptions


@workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: Qubits,
    frequencies: QubitSweepPoints,
    amplitudes: QubitSweepPoints,
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
        amplitudes:
            The amplitudes to sweep over for each qubit drive pulse.  `amplitudes` must
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
        options = TuneUpAnalysisWorkflowOptions()
        result = analysis_workflow(
            results=results
            qubits=[q0, q1],
            lengths=[
                np.linspace(10e-9, 100e-9, 11),
                np.linspace(10e-9, 100e-9, 11),
            ],
            options=options,
        ).run()
        ```
    """
    processed_data_dict = calculate_qubit_population_2d(
        qubits, result, frequencies, amplitudes
    )
    with if_(options.do_plotting):
        with if_(options.do_qubit_population_plotting):
            plot_population(qubits, processed_data_dict)


@task
def calculate_qubit_population_2d(
    qubits: Qubits,
    result: RunExperimentResults,
    slow_axis: QubitSweepPoints,
    fast_axis: QubitSweepPoints,
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, dict[str, ArrayLike]]:
    """Processes the raw data.

     The data is processed in the following way:

     - If calibration traces were used in the experiment, the raw data is rotated based
     on the calibration traces.
     See [cal_trace_rotation.py/rotate_data_to_cal_trace_results] for more details.
     - If no calibration traces were used in the experiment, or do_pca = True is passed
     in options, principal-component analysis is performed on the data.
     See [cal_trace_rotation.py/principal_component_analysis] for more details.

    Arguments:
        qubits:
            The qubits on which the amplitude-Rabi experiments was run. May be either
            a single qubit or a list of qubits.
        result: the result of the experiment, returned by the run_experiment task.
        slow_axis:
            The slow_axis that was swept over in the experiment for
            each qubit. If `qubits` is a single qubit, `slow_axis` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        fast_axis:
            The fast_axis that was swept over in the experiment for
            each qubit. If `qubits` is a single qubit, `fast_axis` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        options:
            The options for processing the raw data.
            See [TuneupAnalysisOptions], [TuneupExperimentOptions] and
            [BaseExperimentOptions] for accepted options.
            Overwrites the options from [TuneupAnalysisOptions],
            [TuneupExperimentOptions] and [BaseExperimentOptions].

    Returns:
        dict with qubit UIDs as keys and the dictionary of processed data for each qubit
        as values. See [cal_trace_rotation.py/calculate_population_1d] for what this
        dictionary looks like.

    Raises:
        TypeError if result is not an instance of RunExperimentResults.
    """
    opts = TuneupAnalysisOptions() if options is None else options
    _, slow_axis = validate_and_convert_qubits_sweeps(qubits, slow_axis)
    qubits, fast_axis = validate_and_convert_qubits_sweeps(qubits, fast_axis)

    processed_data_dict = {}
    for q, q_slow, q_fast in zip(qubits, slow_axis, fast_axis):
        raw_data = result.result[q.uid].data
        if opts.use_cal_traces:
            calibration_traces = [
                np.mean(result.cal_trace[q.uid][cs].data) for cs in opts.cal_states
            ]
            do_pca = opts.do_pca
            num_cal_traces = len(opts.cal_states)
        else:
            calibration_traces = []
            do_pca = True
            num_cal_traces = 0

        linear_data = np.concatenate(raw_data)
        population = calculate_population_1d(
            linear_data,
            q_fast,
            calibration_traces,
            do_pca=do_pca,
        )["population"]
        population = np.reshape(population, [len(q_fast), len(q_slow)])
        processed_data_dict[q.uid] = {
            "sweep_points_slow": q_slow,
            "sweep_points_fast": q_fast,
            "population": population,
            "num_cal_traces": num_cal_traces,
        }
    return processed_data_dict


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
        sweep_points_fast = processed_data_dict[q.uid]["sweep_points_fast"]
        data = processed_data_dict[q.uid]["population"]
        num_cal_traces = processed_data_dict[q.uid]["num_cal_traces"]

        fig, ax = plt.subplots()
        ax.set_title(f"Amplitude Rabi Chevron {q.uid}")
        ax.set_xlabel("Drive Frequency, (GHz)")
        ax.set_ylabel("Drive Pulse Amplitude")
        x, y = np.meshgrid(sweep_points_slow, sweep_points_fast)
        im = ax.pcolormesh(x, y, data, cmap="viridis")
        colorbar_label = (
            "Principal Component (a.u)"
            if (num_cal_traces == 0 or opts.do_pca)
            else f"$|{opts.cal_states[-1]}\\rangle$-State Population"
        )
        fig.colorbar(im, orientation="vertical", label=colorbar_label)

        if opts.save_figures:
            save_artifact(f"Amplitude_Rabi_Chevron_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures
