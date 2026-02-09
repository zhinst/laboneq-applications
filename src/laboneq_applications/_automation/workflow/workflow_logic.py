# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import bisect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from laboneq_applications._automation import WorkflowLayer


class WorkflowLogic(ABC):
    """Workflow decision logic."""

    def __init__(self, *, iterations: int | None = None):
        """Initialize the parameters.

        Arguments:
            iterations: If None, we only execute the decision logic until the layer
                has passed. If int, we execute the decision logic for that many
                iterations, regardless of the layer status.
        """
        self.iterations = iterations

    @abstractmethod
    def run_executable(self, layer: "WorkflowLayer") -> tuple[str, dict]:
        """Run the executable.

        Arguments:
            layer: The workflow layer.

        Returns:
            new_layer_key: The key of the new layer.
            new_params: The dictionary of new workflow parameters.
        """


class FixedParameterUpdate(WorkflowLogic):
    """Fixed workflow parameter update."""

    def __init__(
        self,
        new_layer_key: str,
        parameter_changes: dict,
        *,
        relative: bool = False,
        **kwargs,
    ):
        """Initialize the parameters.

        Arguments:
            new_layer_key: The key of the destination layer.
            parameter_changes: The dictionary of parameter changes. The values in
                the dictionary may be either relative or absolute differences.
            relative: Whether the parameter differences are absolute or relative.
            **kwargs: The keyword arguments for the `WorkflowLogic` parent class.
        """
        super().__init__(**kwargs)
        self.new_layer_key = new_layer_key
        self.parameter_changes = parameter_changes
        self.relative = relative

    def run_executable(self, layer: "WorkflowLayer") -> tuple[str, dict]:
        """Run fixed workflow parameters update."""
        new_params = {}
        for k1, v1 in self.parameter_changes.items():  # k1 == quantum element UID
            for k2, v2 in v1.items():  # k2 == parameter key
                if k1 not in new_params:
                    new_params[k1] = {}
                if self.relative:
                    new_params[k1][k2] = layer.workflow_parameters[k1][k2] * (1 + v2)
                else:
                    new_params[k1][k2] = layer.workflow_parameters[k1][k2] + v2

        return self.new_layer_key, new_params


class AdaptFrequencyRange(WorkflowLogic):
    """Adapt frequency range."""

    def __init__(
        self,
        new_layer_key: str,
        range_thresholds: dict[int, float],
        **kwargs,
    ):
        """Initialize the parameters.

        Arguments:
            new_layer_key: The key of the destination layer.
            range_thresholds: The dictionary of sample thresholds. For example,
                consider {0: 1.1, 200: 1.2, 400: 1.3}. A frequency range between 0 and
                199 will have a multiplier of 1.1, a frequency range between 200 and
                399 will have a multiplier of 1.2, and a frequency range greater than
                400 will have a multiplier of 1.3.
            **kwargs: The keyword arguments for the `WorkflowLogic` parent class.
        """
        super().__init__(**kwargs)
        self.new_layer_key = new_layer_key
        self.range_thresholds = range_thresholds

    @staticmethod
    def get_bucket_value(s: dict[int, float], x: int) -> float:
        """Get bucket value."""
        keys = sorted(s)
        idx = bisect.bisect_right(keys, x) - 1
        if idx < 0:
            raise ValueError(f"Value {x} is less than all bucket lower bounds!")
        return s[keys[idx]]

    def run_executable(self, layer: "WorkflowLayer") -> tuple[str, dict]:
        """Run adapt frequency range."""
        new_params = {}
        for q in layer.quantum_elements:
            new_params[q] = {}
            frequencies = layer.workflow_results[0].output.data[q].result.axis[0][0]

            freq_range = int(max(frequencies) - min(frequencies))
            multiplier = self.get_bucket_value(self.range_thresholds, freq_range)

            midpoint = (max(frequencies) + min(frequencies)) / 2
            new_params[q]["frequencies"] = (
                frequencies - midpoint
            ) * multiplier + midpoint

        return self.new_layer_key, new_params
