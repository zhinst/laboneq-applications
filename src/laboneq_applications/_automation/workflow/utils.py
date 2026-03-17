# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from laboneq.workflow import WorkflowResult


def group_element_workflow_parameters(
    element_workflow_parameters: dict[str, dict[str, Any]],
    quantum_elements: str | list[str | tuple[str, ...]],
) -> dict[str, list[Any]]:
    """Group element workflow parameters into an experiment workflow list format.

    In the element workflow parameters dictionary, the primary key is the
    quantum element UID and the secondary key is the parameter name. However, for
    experiment workflows, the primary key is the parameter name and the parameter
    values are given as a list in quantum element order.

    !!! note
        This is a helper function for `run_executable`.

    Arguments:
        element_workflow_parameters: The element workflow parameters.
        quantum_elements: The target quantum elements.

    Returns:
        The grouped element workflow parameters.
    """
    if isinstance(quantum_elements, str | tuple):
        grouped_element_workflow_parameters = element_workflow_parameters[
            quantum_elements
        ]
    else:
        # Collect all unique parameter keys from all qubits
        secondary_keys = set()
        for k, qb_params in element_workflow_parameters.items():
            if k in quantum_elements:
                secondary_keys.update(qb_params.keys())
        grouped_element_workflow_parameters = {}
        for key in secondary_keys:
            grouped_element_workflow_parameters[key] = []
            for k, d in element_workflow_parameters.items():
                if k in quantum_elements:
                    if key in d:
                        grouped_element_workflow_parameters[key].append(d[key])
                    else:
                        grouped_element_workflow_parameters[key].append(None)

    return grouped_element_workflow_parameters


def get_eval_outputs(
    workflow_results: dict[tuple[str], WorkflowResult],
) -> dict[str, dict[str, bool]]:
    """Get the evaluation output dictionary from the workflow results.

    !!! note
        This is a helper function for `run_executable`.

    Arguments:
        workflow_results: The workflow results.

    Returns:
        The evaluation output.
    """
    eval_outputs = {}
    for workflow_result in workflow_results.values():
        task_list = [t.name for t in workflow_result.tasks]
        if "evaluate_experiment" in task_list:
            eval_output = workflow_result.tasks["evaluate_experiment"].output
            eval_outputs.update(eval_output)
    return eval_outputs
