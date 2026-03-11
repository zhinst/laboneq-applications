# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the task for updating setup parameters."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import uncertainties as unc
from laboneq import workflow

from laboneq_applications.core import validation

if TYPE_CHECKING:
    from lmfit.model import ModelResult

    from laboneq_applications.typing import QuantumElements


@workflow.task(save=False)
def evaluate_parameter_and_fit_r2_thresholds(
    old_parameter_values: dict[str, Any],
    new_parameter_values: dict[str, Any],
    fit_data: dict[str, ModelResult],
    quantum_elements: QuantumElements,
    *,
    parameter: str,
    default_parameter_threshold: float,
    default_fit_r2_threshold: float,
    evaluation_parameters: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, bool]]:
    """Evaluates the parameter and fit r2 thresholds.

    We evaluate the analysis workflow results for a generic experiment. We output the
    evaluation flags, which is a dictionary of the form:

    ```python
    {"q0": {"success": False, "update": False}, ...}
    ```

    For each quantum element:

    If the r2 value of the fit is above the `r2_threshold`, then the `success` flag is
    set to `True`.

    If the change in one of the `new_parameter_values` relative to the
    `old_parameter_values` is above a threshold, then the `update` flag is set to
    `True`.

    If both the `success` and `update` flags are `True`, then the QPU will be
    updated.

    !!! warning
        This evaluation task is not compatible with all experiment workflows. This task
        requires an evaluation parameter of type `float` and fit data that contains an
        r2 value at the location `fit_data[q].rsquared`.

    !!! note
        In this evaluation task, we check only the r2 value of the fit to determine
        whether a generic experiment is successful, and we check only the change in one
        qubit parameter to determine whether the qubit update is significant. In
        practice, many other indictors can be used. We encourage the user to write their
        own experiment evaluation task to match their criteria for success and updates.

    Arguments:
        old_parameter_values:
            The old parameter values.
        new_parameter_values:
            The new parameter values.
        fit_data:
            The fit data.
        quantum_elements:
            The quantum elements to run the experiments on.
        parameter:
            The name of the parameter for which to evaluate the difference.
        default_parameter_threshold:
            The default parameter threshold. If the change in the parameter value is
            above this threshold, then `update` is set to `True`.
        default_fit_r2_threshold:
            The default r2 threshold. If the r2 value of the fit is above this
            threshold, then `success` is set to `True`.
        evaluation_parameters:
            The evaluation parameters. The expected dictionary keys are
            "parameter_thresholds" and "fit_r2_thresholds". The
            "parameter_thresholds" is the dictionary of thresholds
            for the parameter differences, keyed by quantum element UID. The
            "fit_r2_thresholds" is the dictionary of thresholds for the r2 values of
            the fits, keyed by quantum element UID.

    Returns:
        The dictionary of evaluation flags. The keys are the quantum element UIDs and
        the values are dictionaries of booleans. The `success` flag tells us whether
        the experiment is successful, and the `update` flag tells us whether the
        parameter value updates are significant. If both the `success` and `update`
        flags are `True`, then the quantum element in the QPU will be updated.

    Raises:
        ValueError: If `parameter` is not found in `new_parameter_values`.
    """
    quantum_elements = validation.validate_and_convert_qubits_sweeps(quantum_elements)

    params = evaluation_parameters or {}

    # parameter_thresholds
    parameter_thresholds = {}
    for q in quantum_elements:
        parameter_thresholds[q.uid] = default_parameter_threshold
    parameter_thresholds.update(params.get("parameter_thresholds", {}))

    # fit_r2_thresholds
    fit_r2_thresholds = {}
    for q in quantum_elements:
        fit_r2_thresholds[q.uid] = default_fit_r2_threshold
    fit_r2_thresholds.update(params.get("fit_r2_thresholds", {}))

    eval_flags = {}
    for q, q_params in new_parameter_values.items():
        if parameter not in q_params:
            raise ValueError(
                f"{parameter} not found in the new parameter values, {q_params}."
            )

        eval_flags[q] = {"success": False, "update": False}

        # check R-squared success
        r2 = abs(fit_data[q].rsquared)
        if r2 > fit_r2_thresholds[q]:
            eval_flags[q]["success"] = True
        else:
            workflow.log(
                logging.WARNING,
                f"The R-squared value of the fit ({r2}) is below the "
                f"threshold ({fit_r2_thresholds[q]}) for {q}.",
            )

        # check parameter threshold update
        new_param_value = q_params[parameter]
        if isinstance(new_param_value, unc.core.Variable):
            new_param_value = new_param_value.nominal_value
        old_param_value = old_parameter_values[q][parameter]
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
