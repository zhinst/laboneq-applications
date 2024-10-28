"""This module defines the analysis for a dispersive-shift experiment.

The experiment is defined in laboneq_applications.experiments. The goal of this
analysis is to extract the optimal qubit readout frequency at which we obtain the
largest distance in the IQ plane between the transmission signal of the readout
resonator when preparing the qubit in different states.

In this analysis, we first calculate the differences between absolute value of the
pair-wise differences between the complex transmission signals acquired for each
preparation state of the transmon. Then we extract the optimal qubit readout frequency.
Finally, we plot the acquired transmission signals for each state and the calculated
differences.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from laboneq_applications import dsl, workflow
from laboneq_applications.analysis.plotting_helpers import timestamped_title
from laboneq_applications.experiments.options import (
    TaskOptions,
    WorkflowOptions,
)
from laboneq_applications.workflow import option_field, options

if TYPE_CHECKING:
    from collections.abc import Sequence

    import matplotlib as mpl
    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from numpy.typing import ArrayLike

    from laboneq_applications.tasks.run_experiment import RunExperimentResults


@options
class DispersiveShiftAnalysisOptions(TaskOptions):
    """Base options for the analysis of a dispersive-shift experiment.

    Attributes:
        save_figures:
            Whether to save the figures.
            Default: `True`.
        close_figures:
            Whether to close the figures.
            Default: `True`.

    """

    save_figures: bool = option_field(True, description="Whether to save the figures.")
    close_figures: bool = option_field(
        True, description="Whether to close the figures."
    )


@options
class DispersiveShiftAnalysisWorkflowOptions(WorkflowOptions):
    """Option class for a dispersive-shift analysis workflows.

    Attributes:
        do_plotting:
            Whether to create the plots.
            Default: 'True'.
        do_plotting_dispersive_shift:
            Whether to create the dispersive shift plot, i.e. signal magnitudes vs
            frequency for every state.
            Default: True.
        do_plotting_signal_distances:
            Whether to create the plot for the pair-wise signal distances.
            Default: True.
    """

    do_plotting: bool = option_field(True, description="Whether to create the plots.")
    do_plotting_dispersive_shift: bool = option_field(
        True, description="Whether to create the dispersive shift plot."
    )
    do_plotting_signal_distances: bool = option_field(
        True,
        description="Whether to create the plot for the pair-wise signal distances.",
    )


@workflow.workflow(name="dispersive_shift_analysis")
def analysis_workflow(
    result: RunExperimentResults,
    qubit: QuantumElement,
    frequencies: ArrayLike,
    states: Sequence[str],
    options: DispersiveShiftAnalysisWorkflowOptions | None = None,
) -> None:
    """The dispersive-shift analysis Workflow.

    The workflow consists of the following steps:

    - [calculate_signal_differences]()
    - [extract_qubit_parameters]()
    - [plot_dispersive_shift]()
    - [plot_signal_distances]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubit:
            The qubit on which to run the analysis. The UID of this qubit must exist
            in the result.
        frequencies:
            The resonator frequencies to sweep over for the readout pulse
            sent to the resonator. Must be a list of numbers or an array.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].
        options:
            The options for building the workflow, passed as an instance of
                [DispersiveShiftAnalysisWorkflowOptions]. See the docstring of
                [DispersiveShiftAnalysisWorkflowOptions] for more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        options = analysis_workflow.options()
        result = analysis_workflow(
            results=results
            qubit=q0,
            frequencies=np.linspace(7.0, 7.1, 101),
            states="gef",
            options=options,
        ).run()
        ```
    """
    processed_data_dict = calculate_signal_differences(
        qubit, result, frequencies, states
    )
    qubit_parameters = extract_qubit_parameters(qubit, processed_data_dict)
    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_plotting_dispersive_shift):
            plot_dispersive_shift(qubit, result, frequencies, states)
        with workflow.if_(options.do_plotting_signal_distances):
            plot_signal_distances(qubit, frequencies, processed_data_dict)
    workflow.return_(qubit_parameters)


@workflow.task
def calculate_signal_differences(
    qubit: QuantumElement,
    result: RunExperimentResults,
    frequencies: ArrayLike,
    states: Sequence[str],
) -> dict[str, tuple[ArrayLike, ArrayLike, ArrayLike]]:
    """Calculates the pair-wise differences between the signals acquired for each state.

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubit:
            The qubit on which to run the task. The UID of this qubit must exist
            in the result.
        frequencies:
            The resonator frequencies to sweep over for the readout pulse (or CW)
            sent to the resonator. Must be a list of numbers or an array.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].

    Returns:
        dictionary with the magnitudes of the pair-wise differences between the signals
        acquired for each state

    Raises:
        TypeError:
            If the result is not an instance of RunExperimentResults.
    """
    qubit, frequencies = dsl.validation.validate_and_convert_single_qubit_sweeps(
        qubit, frequencies
    )
    dsl.validation.validate_result(result)
    all_state_combinations = combinations(list(states), 2)
    processed_data_dict = {}
    joined_states = ["".join(sc) for sc in all_state_combinations]
    # calculate the pair-wise differences between the magnitudes of the signals
    # acquired for each state
    for state_pair in joined_states:
        s0, s1 = state_pair
        s21_dist = abs(
            result[dsl.handles.result_handle(qubit.uid, suffix=s1)].data
            - result[dsl.handles.result_handle(qubit.uid, suffix=s0)].data
        )
        max_idx = np.argmax(s21_dist)
        processed_data_dict[state_pair] = (
            s21_dist,
            s21_dist[max_idx],
            frequencies[max_idx],
        )

    if len(states) > 2:  # noqa: PLR2004
        # Calculate the sum of all the signal differences. When more than two states are
        # measured, we take the optimal readout frequency as the one at which the sum of
        # all the signal difference is maximal
        s21_dist_sum = np.sum(
            [s21_data[0] for s21_data in processed_data_dict.values()], axis=0
        )
        max_idx = np.argmax(s21_dist_sum)
        processed_data_dict["sum"] = (
            s21_dist_sum,
            s21_dist_sum[max_idx],
            frequencies[max_idx],
        )

    return processed_data_dict


@workflow.task
def extract_qubit_parameters(
    qubit: QuantumElement,
    processed_data_dict: dict[str, tuple[ArrayLike, ArrayLike, ArrayLike]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Extract the optimal qubit readout resonator frequency.

    Arguments:
        qubit:
            The qubit on which to run the task. The UID of this qubit must exist
            in processed_data_dict.
        processed_data_dict: the dictionary returned by calculate_signal_differences,
            containing the magnitudes of the pair-wise differences between the signals
            acquired for each state.

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
    """
    qubit = dsl.validation.validate_and_convert_single_qubit_sweeps(qubit)
    qubit_parameters = {
        "old_parameter_values": {qubit.uid: {}},
        "new_parameter_values": {qubit.uid: {}},
    }

    # Store the readout resonator frequency value
    qubit_parameters["old_parameter_values"][qubit.uid] = {
        "readout_resonator_frequency": qubit.parameters.readout_resonator_frequency,
    }

    # Store the readout resonator frequency value
    # When more than two states are measured, we take the optimal readout frequency as
    # the one at which the sum of all the signal difference is maximal
    key = "sum" if "sum" in processed_data_dict else next(iter(processed_data_dict))
    optimal_frequency = processed_data_dict[key][2]
    qubit_parameters["new_parameter_values"][qubit.uid] = {
        "readout_resonator_frequency": optimal_frequency
    }

    return qubit_parameters


@workflow.task
def plot_dispersive_shift(
    qubit: QuantumElement,
    result: RunExperimentResults,
    frequencies: ArrayLike,
    states: Sequence[str],
    options: DispersiveShiftAnalysisOptions | None = None,
) -> mpl.figure.Figure | None:
    """Plot the magnitude of the transmission signals for each preparation state.

    Arguments:
        qubit:
            The qubit on which to run the task.
        result:
            The experiment results returned by the run_experiment task.
        frequencies:
            The resonator frequencies to sweep over for the readout pulse (or CW)
            sent to the resonator. Must be a list of numbers or an array.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].
        options:
            The options, passed as an instance of [DispersiveShiftAnalysisOptions]. See
            the docstring of this class for more details.

    Returns:
        the matplotlib figure

    Raises:
        TypeError:
            If the result is not an instance of RunExperimentResults.
    """
    opts = DispersiveShiftAnalysisOptions() if options is None else options
    qubit, frequencies = dsl.validation.validate_and_convert_single_qubit_sweeps(
        qubit, frequencies
    )
    dsl.validation.validate_result(result)

    # Plot S21 for each prep state
    fig, ax = plt.subplots()
    ax.set_xlabel("Readout Frequency, $f_{\\mathrm{RO}}$ (GHz)")
    ax.set_ylabel("Signal Magnitude, $|S_{21}|$ (a.u.)")
    ax.set_title(timestamped_title(f"Dispersive Shift {qubit.uid}"))
    for state in states:
        data_mag = abs(result[dsl.handles.result_handle(qubit.uid, suffix=state)].data)
        ax.plot(frequencies / 1e9, data_mag, label=state)
    ax.legend(frameon=False)

    if opts.save_figures:
        workflow.save_artifact(f"Dispersive_shift_{qubit.uid}", fig)

    if opts.close_figures:
        plt.close(fig)

    return fig


@workflow.task
def plot_signal_distances(
    qubit: QuantumElement,
    frequencies: ArrayLike,
    processed_data_dict: dict[str, ArrayLike],
    options: DispersiveShiftAnalysisOptions | None = None,
) -> mpl.figure.Figure | None:
    """Plot the pair-wise differences between the signals acquired for each state.

    Arguments:
        qubit:
            The qubit on which to run the rask. The UID of this qubit must exist
            in processed_data_dict.
        frequencies:
            The resonator frequencies to sweep over for the readout pulse (or CW)
            sent to the resonator. Must be a list of numbers or an array.
        processed_data_dict: the dictionary returned by calculate_signal_differences,
            containing the magnitudes of the pair-wise differences between the signals
            acquired for each state; returned by
        options:
            The options, passed as an instance of [DispersiveShiftAnalysisOptions]. See
            the docstring of this class for more details.

    Returns:
        the matplotlib figure
    """
    opts = DispersiveShiftAnalysisOptions() if options is None else options
    qubit, frequencies = dsl.validation.validate_and_convert_single_qubit_sweeps(
        qubit, frequencies
    )

    # Plot the S21 distances
    fig, ax = plt.subplots()
    ax.set_xlabel("Readout Frequency, $f_{\\mathrm{RO}}$ (GHz)")
    ax.set_ylabel("Signal-Magnitude Difference, $|\\Delta S_{21}|$ (a.u.)")
    ax.set_title(timestamped_title(f"Signal Differences {qubit.uid}"))
    for state_pairs, (
        s21_dist,
        opt_s21_dist,
        opt_freq,
    ) in processed_data_dict.items():
        legend_label = (
            f"{state_pairs}: $f_{{\\mathrm{{max}}}}$ = " f"{opt_freq / 1e9:.4f} GHz"
        )
        (line,) = ax.plot(frequencies / 1e9, s21_dist, label=legend_label)
        # add point at optimal frequency
        ax.plot(opt_freq / 1e9, opt_s21_dist, "o", c=line.get_c())
        # add vertical line at optimal frequency
        ax.vlines(opt_freq / 1e9, min(s21_dist), opt_s21_dist, colors=line.get_c())
    ax.legend(frameon=False, loc="upper left", handlelength=0.5)

    if opts.save_figures:
        workflow.save_artifact(f"Signal_distances_{qubit.uid}", fig)

    if opts.close_figures:
        plt.close(fig)

    return fig
