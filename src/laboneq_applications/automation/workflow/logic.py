# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""The workflow automation logic."""

import bisect
from typing import TYPE_CHECKING

import attrs
from laboneq.automation.logic import AutomationLogic
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter

if TYPE_CHECKING:
    from laboneq_applications.automation import WorkflowLayer


@classformatter
@attrs.define(kw_only=True)
class WorkflowLogic(AutomationLogic):
    """Workflow decision logic."""

    def run_executable_core(self, layer: "WorkflowLayer") -> tuple[str | None, dict]:
        """The core of the `run_executable` method.

        !!! note
            This is an internal method that is meant to be called via `run_executable`.

        !!! tip
            Use `WorkflowLayer.target_node_keys` and
            `WorkflowLayer.target_parameters` instead
            of `WorkflowLayer.node_keys` and `WorkflowLayer.parameters`, so that
            optional overrides in `WorkflowAutomation.run_layer` are respected.

        Arguments:
            layer: The workflow automation layer.

        Returns:
            new_layer_key: The key of the next layer to be executed.
            new_params: The dictionary of new automation parameters.
        """


@classformatter
@attrs.define
class AdaptFrequencyRange(WorkflowLogic):
    """Adapt frequency range."""

    new_layer_key: str
    range_thresholds: dict[int, float]

    @staticmethod
    def get_bucket_value(s: dict[int, float], x: int) -> float:
        """Get bucket value."""
        keys = sorted(s)
        idx = bisect.bisect_right(keys, x) - 1
        if idx < 0:
            raise ValueError(f"Value {x} is less than all bucket lower bounds!")
        return s[keys[idx]]

    def run_executable_core(self, layer: "WorkflowLayer") -> tuple[str, dict]:
        """Run adapt frequency range."""
        new_params = {}
        for q in layer.quantum_elements:
            new_params[q] = {}
            frequencies = (
                next(iter(layer.workflow_results.values()))
                .output.data[q]
                .result.axis[0]
            )

            freq_range = int(max(frequencies) - min(frequencies))
            multiplier = self.get_bucket_value(self.range_thresholds, freq_range)

            midpoint = (max(frequencies) + min(frequencies)) / 2
            new_params[q]["frequencies"] = (
                frequencies - midpoint
            ) * multiplier + midpoint

        return self.new_layer_key, {"workflow_parameters": new_params}
