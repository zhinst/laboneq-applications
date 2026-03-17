# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""The workflow automation."""

from __future__ import annotations

import attrs
from laboneq.automation import Automation
from laboneq.automation import AutomationStatus as Status
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl import Session
from laboneq.dsl.quantum import QPU


@classformatter
@attrs.define
class WorkflowAutomation(Automation):
    """The workflow automation framework.

    In LabOne Q, experiments are structured using workflows, which may serve different
    roles. For example, a workflow may calibrate quantum elements, apply a set of
    quantum operations, and/or analyze the experiment results. Moreover, some workflows
    are dependent on the outcome of their predecessors. To scale to large, complex
    experiment suites, it is crucial to structure and automate the execution of these
    experiment workflows. We achieve this using a workflow automation framework.

    Attributes:
        session: The session.
        qpu: The quantum processing unit (QPU) (optional). By default, all layers will
            use this QPU. This can be overridden on a per-layer basis.
    """

    session: Session
    qpu: QPU | None = None

    def reset(self) -> None:
        """Reset the automation framework."""
        super().reset()

        for layer in self.layers():
            layer.eval_outputs = {}

    def sync_auto_params_with_layer_params(self, layer_key: str) -> None:
        """Synchronize the automation parameters with the layer parameters.

        Arguments:
            layer_key: The layer key.
        """
        super().sync_auto_params_with_layer_params(layer_key)

        # Update auto QPU with layer QPU
        layer = self.get_layer(layer_key)
        if layer.qpu:
            self.qpu = layer.qpu

    def _run_layer(
        self,
        layer_key: str,
    ) -> tuple[str | None, dict]:
        """Run the automation layer.

        !!! note
            This is an internal method that is meant to be called via `run_layer`.

        !!! important
            When the end of the graph is reached, return the new layer key `None`.
            The `next_layer_key` method does this automatically.

        Arguments:
            layer_key: The layer key.

        Returns:
            new_layer_key: The key of the next layer to be executed.
            new_params: The dictionary of new automation parameters.
        """
        layer = self.get_layer(layer_key)

        if not layer.sequential:
            layer.run_executable(self)
        else:
            combined_results = {}
            combined_eval_outputs = {}

            for q in layer.quantum_elements:
                if layer[q].status in Status.active():
                    result = layer.run_executable(self, quantum_elements=[q])
                    combined_results |= result
                    combined_eval_outputs |= layer.eval_outputs
            layer.workflow_results = combined_results
            layer.eval_outputs = combined_eval_outputs

        return self.next_layer_key(layer_key), {}
