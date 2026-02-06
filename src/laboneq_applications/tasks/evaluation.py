# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the task for updating setup parameters."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import uncertainties as unc
from laboneq import workflow
from laboneq.workflow.result import WorkflowResult

from laboneq_applications.core import validation

if TYPE_CHECKING:
    from laboneq_applications.typing import QuantumElements


@workflow.task
def evaluate_experiment(
    analysis_results: WorkflowResult,
    parameter: str,
    parameter_thresholds: dict[str, float],
    fit_r2_thresholds: dict[str, float],
) -> dict[str, dict[str, bool]]:
    """Evaluates the analysis workflow result.

    We evaluate the analysis workflow results for the Ramsey experiment. We output the
    evaluation flags, which is a dictionary of the form:

    ```python
    {"q0": {"success": False, "update": False}, ...}
    ```

    For each qubit:

    If the r2 value of the fit is above the `r2_threshold` (set in the
    `RamseyEvaluationOptions`), then the `success` flag is set to `True`.

    If the change in one of the `new_parameter_values` relative to the
    `old_parameter_values` is above a threshold (set in the `RamseyEvaluationOptions`),
    then the `update` flag is set to `True`.

    If both the `success` and `update` flags are `True`, then the QPU will be
    updated.

    !!! note
        In this evaluation task, we check only the r2 value of the fit to determine
        whether a Ramsey experiment is successful, and we check only the change in one
        qubit parameter to determine whether the qubit update is significant. In
        practice, many other indictors can be used. We encourage the user to expand
        this evaluation task to match their criteria for success and updates.

    Arguments:
        analysis_results:
            The analysis workflow results.
        parameter:
            Parameter for which to evaluate the difference.
        parameter_thresholds:
            Thresholds for the parameter difference for each experiment resource
            (qubits, pairs of qubits, etc.). Passed as a dictionary with qubit UIDs or
             tuples of two qubit UIDs as keys and the threshold values as values.
        fit_r2_thresholds:
            Threshold for the r2 value of the fit for each experiment resource
            (qubits, pairs of qubits, etc.). Passed as a dictionary with qubit UIDs or
             tuples of two qubit UIDs as keys and the threshold values as values.

    Returns:
        The dictionary of evaluation flags. The keys are the qubit UIDs and the values
        are dictionaries of booleans. The `success` flag tells us whether the Ramsey
        experiment is successful, and the `update` flag tells us whether the qubit
        parameter value updates are significant. If both the `success` and `update`
        flags are `True`, then the qubit in the QPU will be updated.
    """
    fit_results = analysis_results.tasks["fit_data"].output
    qubit_parameters = analysis_results.tasks["extract_qubit_parameters"].output
    old_params = qubit_parameters["old_parameter_values"]
    new_params = qubit_parameters["new_parameter_values"]

    eval_flags = {}
    for q, q_params in new_params.items():
        if q not in fit_r2_thresholds:
            raise ValueError(f"{q} not found in fit_r2_thresholds")
        if q not in parameter_thresholds:
            raise ValueError(f"{q} not found in parameter_thresholds")
        if parameter not in q_params:
            raise ValueError(
                f"{parameter} not found in the new parameter values, {q_params}."
            )

        eval_flags[q] = {"success": False, "update": False}

        # check R-squared success
        r2 = abs(fit_results[q].rsquared)
        if r2 > fit_r2_thresholds[q]:
            eval_flags[q]["success"] = True
        else:
            workflow.log(
                logging.WARNING,
                f"The R-squared value of the fit ({r2}) is below the "
                f"threshold ({fit_r2_thresholds[q]}) for {q}.",
            )

        # check parameter threshold success
        new_param_value = q_params[parameter]
        if isinstance(new_param_value, unc.core.Variable):
            new_param_value = new_param_value.nominal_value
        old_param_value = old_params[q][parameter]
        if isinstance(old_param_value, unc.core.Variable):
            old_param_value = old_param_value.nominal_value
        param_diff = abs(new_param_value - old_param_value)
        if param_diff < parameter_thresholds[q]:
            workflow.log(
                logging.WARNING,
                f"The difference ({param_diff}) between the new and old "
                f"values of the parameter {parameter} is below the threshold "
                f"({parameter_thresholds[q]}) for {q}.",
            )
        elif eval_flags[q]["success"]:
            eval_flags[q]["update"] = True

    return eval_flags


@workflow.task
def get_evaluation_parameter(
    default_evaluation_parameter: str, evaluation_parameter: str | None = None
) -> str:
    """Returns the evaluation parameter, falling back to a default value if needed."""
    return (
        default_evaluation_parameter
        if evaluation_parameter is None
        else evaluation_parameter
    )


@workflow.task
def get_evaluation_thresholds(
    qubits: QuantumElements,
    default_evaluation_threshold: float,
    evaluation_thresholds: float | Sequence[float | None] | None = None,
) -> dict[str, float]:
    """Returns the evaluation thresholds, falling back to default values if needed.

    !!! note
        This task converts from the list format (input) to the dict format (output).
    """
    # Validate qubits
    qubits = validation.validate_and_convert_qubits_sweeps(qubits)

    # Standardize inputs as lists
    if isinstance(evaluation_thresholds, (float, int)):
        evaluation_thresholds = [evaluation_thresholds]

    # Add default values
    if evaluation_thresholds is None:
        final_evaluation_thresholds = [default_evaluation_threshold for _ in qubits]
    else:
        final_evaluation_thresholds = []
        for param in evaluation_thresholds:
            if param is None:
                final_evaluation_thresholds.append(default_evaluation_threshold)
            else:
                final_evaluation_thresholds.append(param)

    final_evaluation_thresholds_dict = {}
    for q_idx, threshold in enumerate(final_evaluation_thresholds):
        final_evaluation_thresholds_dict[qubits[q_idx].uid] = threshold

    return final_evaluation_thresholds_dict
