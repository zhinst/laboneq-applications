# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the analysis for a qubit measurement QND experiment.

The experiment is defined in laboneq_applications.contrib.experiments.

In this analysis, we determine the QND fidelity from two consecutive measurements
by comparing the measurement outputs. This is plotted in a confusion matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from laboneq.workflow import if_, return_, task, workflow
from sklearn.metrics import confusion_matrix

from laboneq_applications.analysis.iq_blobs import (
    calculate_assignment_fidelities,
    plot_assignment_matrices,
)
from laboneq_applications.analysis.options import (
    TuneUpAnalysisWorkflowOptions,  # noqa: TCH001
)
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_result,
)

if TYPE_CHECKING:
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import QuantumElements

@workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The measurement qnd analysis workflow.

    The workflow consists of the following steps:
    - [collect_shots]()
    - [calculate_assignment_matrices]()
    - [calculate_assignment_fidelities]()
    - [plot_assignment_matrices]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        options:
            The options for building the workflow, passed as an instance of
            [TuneUpAnalysisWorkflowOptions]. See the docstring of this class for
            more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        options = analysis_workflow.options()
        result = analysis_workflow(
            results=results
            qubits=[q0, q1],
            options=options,
        ).run()
        ```
    """
    processed_data_dict = collect_shots(qubits, result)
    assignment_matrices = calculate_assignment_matrices(
        qubits, processed_data_dict
        )
    qnd_assignment_fidelities = calculate_assignment_fidelities(
        qubits, assignment_matrices
        )

    states = ["g","e"]
    with if_(options.do_plotting):
        plot_assignment_matrices(
                qubits, states, assignment_matrices, qnd_assignment_fidelities
                )

    return_(qnd_assignment_fidelities)


@task
def collect_shots(
    qubits: QuantumElements,
    result: RunExperimentResults,
) -> dict[str, dict[str, ArrayLike | dict]]:
    """Collect the single shots acquired for each preparation state.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in result.
        result:
            The experiment results returned by the run_experiment task.

    Returns:
        dict with qubit UIDs as keys and values as a dict with the following keys:
            first_measurement - list of integers containing the first measurement
                                outcomes.
            second_measurement - list of integers containing the second measurement
                                outcomes.
    """
    qubits = validate_and_convert_qubits_sweeps(qubits)
    validate_result(result)
    processed_data_dict = {q.uid: {} for q in qubits}
    for q in qubits:
        processed_data_dict[q.uid] = {
            "first_measurement": np.real(result[f"measure_1_{q.uid}"].data).astype(int),
            "second_measurement": np.real(result[f"measure_2_{q.uid}"].data).astype(int)
        }

    return processed_data_dict


@task
def calculate_assignment_matrices(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike | dict]],
) -> dict[str, None]:
    """Calculate the correct assignment matrices.

    Arguments:
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the
            processed_data_dict.
        processed_data_dict: the processed data dictionary returned by collect_shots

    Returns:
        dict with qubit UIDs as keys and the assignment matrix for each qubit as keys.
    """
    qubits = validate_and_convert_qubits_sweeps(qubits)
    assignment_matrices = {}
    for q in qubits:
        first_measurement = processed_data_dict[q.uid]["first_measurement"]
        second_measurement = processed_data_dict[q.uid]["second_measurement"]

        assignment_matrices[q.uid] = confusion_matrix(
            first_measurement, second_measurement, normalize="true"
        )

    return assignment_matrices
