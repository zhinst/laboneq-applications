"""This module defines the analysis for an IQ-blob experiment.

The experiment is defined in laboneq_applications.experiments.

In this analysis, we collect the single shots acquire for each prepared stated and
then use LinearDiscriminantAnalysis from the sklearn library to classify the data into
the prepared states. From this classification, we calculate the correct-state-assignment
matrix and the correct-state-assignment fidelity. Finally, we plot the single shots for
each prepared state and the correct-state-assignment matrix.
"""

from __future__ import annotations

import logging
from itertools import product
from typing import TYPE_CHECKING

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import confusion_matrix

from laboneq_applications import workflow
from laboneq_applications.analysis.plotting_helpers import timestamped_title
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_result,
)
from laboneq_applications.experiments.options import TaskOptions, WorkflowOptions
from laboneq_applications.workflow import option_field, options

if TYPE_CHECKING:
    from collections.abc import Sequence

    import matplotlib as mpl
    from numpy.typing import ArrayLike

    from laboneq_applications.tasks.run_experiment import RunExperimentResults
    from laboneq_applications.typing import Qubits


@options
class IQBlobAnalysisWorkflowOptions(WorkflowOptions):
    """Option class for IQ-blob analysis workflows.

    Attributes:
        do_fitting:
            Whether to perform the fit.
            Default: `True`.
        do_plotting:
            Whether to create plots.
            Default: 'True'.
        do_plotting_iq_blobs:
            Whether to create the IQ-blob plots of the single shots.
            Default: 'True'.
        do_plotting_assignment_matrices:
            Whether to create the assignment matrix plots.
            Default: 'True'.
    """

    do_fitting: bool = option_field(True, description="Whether to perform the fit.")
    do_plotting: bool = option_field(True, description="Whether to create plots.")
    do_plotting_iq_blobs: bool = option_field(
        True, description="Whether to create the IQ-blob plots of the single shots."
    )
    do_plotting_assignment_matrices: bool = option_field(
        True, description="Whether to create the assignment matrix plots."
    )


@options
class IQBlobAnalysisOptions(TaskOptions):
    """Option class for the tasks in the iq-blob analysis workflows.

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


@workflow.workflow(name="iq_blobs_analysis")
def analysis_workflow(
    result: RunExperimentResults,
    qubits: Qubits,
    states: Sequence[str],
    options: IQBlobAnalysisWorkflowOptions | None = None,
) -> None:
    """The IQ Blobs analysis Workflow.

    The workflow consists of the following steps:

    - [collect_shots]()
    - [fit_data]()
    - [calculate_assignment_matrices]()
    - [calculate_assignment_fidelities]()
    - [plot_iq_blobs]()
    - [plot_assignment_matrices]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].
        options:
            The options for building the workflow as an instance of
            [IQBlobAnalysisWorkflowOptions]. See the docstring of this class for more
            details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        result = analysis_workflow(
            results=results
            qubits=[q0, q1],
            amplitudes=[
                np.linspace(0, 1, 11),
                np.linspace(0, 0.75, 11),
            ],
            options=analysis_workflow.options(),
        ).run()
        ```
    """
    processed_data_dict = collect_shots(qubits, result, states)
    fit_results = None
    assignment_matrices = None
    assignment_fidelities = None
    with workflow.if_(options.do_fitting):
        fit_results = fit_data(qubits, processed_data_dict)
        assignment_matrices = calculate_assignment_matrices(
            qubits, processed_data_dict, fit_results
        )
        assignment_fidelities = calculate_assignment_fidelities(
            qubits, assignment_matrices
        )

    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_plotting_iq_blobs):
            plot_iq_blobs(qubits, states, processed_data_dict, fit_results)
        with workflow.if_(options.do_plotting_assignment_matrices):
            with workflow.if_(options.do_fitting):
                plot_assignment_matrices(
                    qubits, states, assignment_matrices, assignment_fidelities
                )
    workflow.return_(assignment_fidelities)


@workflow.task
def collect_shots(
    qubits: Qubits,
    result: RunExperimentResults,
    states: Sequence[str],
) -> dict[str, dict[str, ArrayLike | dict]]:
    """Collect the single shots acquired for each preparation state in states.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in result.
        result:
            The experiment results returned by the run_experiment task.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].

    Returns:
        dict with qubit UIDs as keys and values as a dict with the following keys:
              shots_per_state - dict with states as keys and raw single shots as values.
              shots_combined - list of the real and imaginary part of the shots in
                shots_per_state, in a form expected by LinearDiscriminantAnalysis.
              ideal_states_shots - list of the same shape as shots_combined with ints
                specifying the state (0, 1, 2) the qubit is expected to be found in
                ideally for each shot.
    """
    qubits = validate_and_convert_qubits_sweeps(qubits)
    validate_result(result)
    states_map = {"g": 0, "e": 1, "f": 2}
    processed_data_dict = {q.uid: {} for q in qubits}
    for q in qubits:
        shots = {}
        shots_combined = []
        ideal_states = []

        for s in states:
            shots[s] = result.result[q.uid][s].data
            shots_combined += [
                np.concatenate(
                    [
                        np.real(shots[s])[:, np.newaxis],
                        np.imag(shots[s])[:, np.newaxis],
                    ],
                    axis=1,
                )
            ]

            ideal_states += [states_map[s] * np.ones(len(shots[s]))]

        processed_data_dict[q.uid] = {
            "shots_per_state": shots,
            "shots_combined": np.concatenate(shots_combined, axis=0),
            "ideal_states_shots": np.concatenate(ideal_states),
        }

    return processed_data_dict


@workflow.task
def fit_data(
    qubits: Qubits,
    processed_data_dict: dict[str, dict[str, ArrayLike | dict]],
) -> dict | dict[str, None]:
    """Perform a classification of the shots using LinearDiscriminantAnalysis.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
        processed_data_dict: the processed data dictionary returned by collect_shots

    Returns:
        dict with qubit UIDs as keys and the classification result for each qubit as
        keys.
    """
    qubits = validate_and_convert_qubits_sweeps(qubits)
    fit_results = {}

    for q in qubits:
        shots_combined = processed_data_dict[q.uid]["shots_combined"]
        ideal_states_shots = processed_data_dict[q.uid]["ideal_states_shots"]
        clf = LinearDiscriminantAnalysis()
        try:
            clf.fit(shots_combined, ideal_states_shots)
        except Exception as err:  # noqa: BLE001
            workflow.log(logging.ERROR, "Fit failed for %s: %s.", q.uid, err)
        else:
            fit_results[q.uid] = clf

    return fit_results


@workflow.task
def calculate_assignment_matrices(
    qubits: Qubits,
    processed_data_dict: dict[str, dict[str, ArrayLike | dict]],
    fit_results: dict[str, None] | dict,
) -> dict[str, None]:
    """Calculate the correct assignment matrices from the result of the classification.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
        processed_data_dict: the processed data dictionary returned by collect_shots
        fit_results: the classification fit results returned by fit_data.

    Returns:
        dict with qubit UIDs as keys and the assignment matrix for each qubit as keys.
    """
    qubits = validate_and_convert_qubits_sweeps(qubits)
    assignment_matrices = {}
    for q in qubits:
        if q.uid not in fit_results:
            continue

        shots_combined = processed_data_dict[q.uid]["shots_combined"]
        ideal_states_shots = processed_data_dict[q.uid]["ideal_states_shots"]
        clf = fit_results[q.uid]
        assignment_matrices[q.uid] = confusion_matrix(
            ideal_states_shots, clf.predict(shots_combined), normalize="true"
        )

    return assignment_matrices


@workflow.task
def calculate_assignment_fidelities(
    qubits: Qubits,
    assignment_matrices: dict[str, None],
) -> dict[str, float]:
    """Calculate the correct assignment fidelity from the correct assignment matrices.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            assignment_matrices.
        assignment_matrices: the dictionary of assignment matrices returned by
            calculate_assignment_matrices.

    Returns:
        dict with qubit UIDs as keys and the assignment fidelity for each qubit as keys.
    """
    qubits = validate_and_convert_qubits_sweeps(qubits)
    assignment_fidelities = {}
    for q in qubits:
        if q.uid not in assignment_matrices:
            continue

        assigm_mtx = assignment_matrices[q.uid]
        assignment_fidelities[q.uid] = np.trace(assigm_mtx) / float(np.sum(assigm_mtx))

    return assignment_fidelities


@workflow.task
def plot_iq_blobs(
    qubits: Qubits,
    states: Sequence[str],
    processed_data_dict: dict[str, dict[str, ArrayLike | dict]],
    fit_results: dict[str, None] | None,
    options: IQBlobAnalysisOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the IQ-blobs plots.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in
            processed_data_dict and fit_results.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].
        processed_data_dict: the processed data dictionary returned by collect_shots.
        fit_results: the classification fit results returned by fit_data.
        options:
            The options for building the workflow as an instance of
            [IQBlobAnalysisOptions]. See the docstring of this class for more details.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.

        If a qubit uid is not found in fit_results, the fit is not plotted.
    """
    opts = IQBlobAnalysisOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}
    for q in qubits:
        shots_per_state = processed_data_dict[q.uid]["shots_per_state"]
        shots_combined = processed_data_dict[q.uid]["shots_combined"]

        fig, ax = plt.subplots()
        ax.set_title(timestamped_title(f"IQ Blobs {q.uid}"))
        ax.set_xlabel("Real Signal Component, $V_{\\mathrm{I}}$ (a.u.)")
        ax.set_ylabel("Imaginary Signal Component, $V_{\\mathrm{Q}}$ (a.u.)")

        for i, s in enumerate(states):
            state_shots = shots_per_state[s]
            # plot shots
            ax.scatter(
                np.real(state_shots),
                np.imag(state_shots),
                c=f"C{i}",
                alpha=0.25,
                label=s,
            )
            # plot mean point
            mean_state = np.mean(state_shots)
            ax.plot(np.real(mean_state), np.imag(mean_state), "o", mfc=f"C{i}", mec="k")

        if len(states) > 1 and fit_results is not None and q.uid in fit_results:
            clf = fit_results[q.uid]
            # plot discrimination lines
            levels = None if len(states) > 2 else [0.5]  # noqa: PLR2004
            DecisionBoundaryDisplay.from_estimator(
                clf,
                shots_combined,
                grid_resolution=500,
                plot_method="contour",
                ax=ax,
                eps=1e-1,
                levels=levels,
            )

        ax.legend(frameon=False)

        if opts.save_figures:
            workflow.save_artifact(f"IQ_blobs_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures


@workflow.task
def plot_assignment_matrices(
    qubits: Qubits,
    states: Sequence[str],
    assignment_matrices: dict[str, ArrayLike],
    assignment_fidelities: dict[str, float],
    options: IQBlobAnalysisOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the correct-assignment-matrices plots.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in
            assignment_matrices and assignment_fidelities.
        states:
            The basis states the qubits should be prepared in. May be either a string,
            e.g. "gef", or a list of letters, e.g. ["g","e","f"].
        assignment_matrices: the dictionary of assignment matrices returned by
            calculate_assignment_matrices.
        assignment_fidelities: the dictionary of assignment fidelities returned by
            calculate_assignment_matrices.
        options:
            The options for building the workflow as an instance of
            [IQBlobAnalysisOptions]. See the docstring of this class for more details.

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = IQBlobAnalysisOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)
    figures = {}
    for q in qubits:
        if q.uid not in assignment_matrices:
            figures[q.uid] = None
            continue

        assignm_mtx = assignment_matrices[q.uid]

        fig, ax = plt.subplots()
        ax.set_ylabel("Prepared State")
        ax.set_xlabel(
            "Assigned State"
            f"\n$F_{{avg}}$ = {assignment_fidelities[q.uid] * 100:0.2f}%"
            if q.uid in assignment_fidelities
            else ""
        )
        ax.set_title(timestamped_title(f"Assignment matrix {q.uid}"))

        cmap = plt.get_cmap("Reds")
        im = ax.imshow(
            assignm_mtx,
            interpolation="nearest",
            cmap=cmap,
            norm=mc.LogNorm(vmin=5e-3, vmax=1.0),
        )
        cb = fig.colorbar(im)
        cb.set_label("Assignment Probability, $P$")

        target_names = ["$|g\\rangle$", "$|e\\rangle$"]
        if "f" in states:
            target_names += ["$|f\\rangle$"]
        tick_marks = np.arange(len(target_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(target_names)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(target_names)

        thresh = assignm_mtx.max() / 1.5
        for i, j in product(range(assignm_mtx.shape[0]), range(assignm_mtx.shape[1])):
            ax.text(
                j,
                i,
                f"{assignm_mtx[i, j]:0.4f}",
                horizontalalignment="center",
                color="white" if assignm_mtx[i, j] > thresh else "black",
                fontsize=plt.rcParams["font.size"] + 2,
            )

        if opts.save_figures:
            workflow.save_artifact(f"Assignment_matrix_{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures
