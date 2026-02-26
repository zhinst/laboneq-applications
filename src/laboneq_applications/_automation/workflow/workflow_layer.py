# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import attrs
from laboneq._automation import AutomationLayer
from laboneq._automation.element import AutomationElementStatus as Status
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.quantum import QPU, QuantumParameters
from laboneq.workflow import WorkflowBuilder, WorkflowResult

from laboneq_applications._automation.workflow.utils import (
    get_eval_outputs,
    group_element_workflow_parameters,
)
from laboneq_applications._automation.workflow.workflow_node import WorkflowNode

if TYPE_CHECKING:
    from laboneq_applications._automation.workflow.workflow_automation import (
        WorkflowAutomation,
    )


@classformatter
@attrs.define
class WorkflowLayer(AutomationLayer):
    """A workflow layer in the automation framework.

    Attributes:
        qpu: The QPU. By default, the QPU from the `Automation` instance is used.
        eval_outputs: The layer evaluation outputs.
    """

    qpu: QPU | None = None
    eval_outputs: dict[str, dict[str, bool]] = attrs.field(factory=dict, init=False)

    @property
    def nodes(self) -> dict[str, WorkflowNode]:
        """The node dictionary."""
        for node_key in self.node_keys:
            deps = {
                f"{layer_key}_{node_key}"
                for layer_key in self.depends_on
                if layer_key != "root"
            }
            if node_key not in self._node_lookup:
                self._node_lookup[node_key] = WorkflowNode(
                    key=node_key,
                    depends_on=deps,
                    layer_key=self.key,
                )
        return {
            k: self._node_lookup[k] for k in self.node_keys if k in self._node_lookup
        }

    @property
    def workflow_builder(self) -> WorkflowBuilder | None:
        return self.function

    @workflow_builder.setter
    def workflow_builder(self, value: WorkflowBuilder | None) -> None:
        self.function = value

    @property
    def quantum_elements(self) -> list[str]:
        return self.node_keys

    @quantum_elements.setter
    def quantum_elements(self, value: list[str]) -> None:
        self.node_keys = value

    @property
    def element_workflow_parameters(self) -> dict[str, dict[str, Any]]:
        if "element_workflow_parameters" in self.parameters:
            return self.parameters["element_workflow_parameters"]
        return {}

    @element_workflow_parameters.setter
    def element_workflow_parameters(self, value: dict[str, dict[str, Any]]) -> None:
        self.parameters["element_workflow_parameters"] = value

    @property
    def common_workflow_parameters(self) -> dict[str, Any]:
        if "common_workflow_parameters" in self.parameters:
            return self.parameters["common_workflow_parameters"]
        return {}

    @common_workflow_parameters.setter
    def common_workflow_parameters(self, value: dict[str, Any]) -> None:
        self.parameters["common_workflow_parameters"] = value

    @property
    def temporary_qpu_parameters(
        self,
    ) -> dict[str | tuple[str, str, str], dict | QuantumParameters]:
        if "temporary_qpu_parameters" in self.parameters:
            return self.parameters["temporary_qpu_parameters"]
        return {}

    @temporary_qpu_parameters.setter
    def temporary_qpu_parameters(
        self, value: dict[str | tuple[str, str, str], dict | QuantumParameters]
    ) -> None:
        self.parameters["temporary_qpu_parameters"] = value

    @property
    def workflow_options(self) -> dict[str, Any]:
        if "workflow_options" in self.parameters:
            return self.parameters["workflow_options"]
        return {}

    @workflow_options.setter
    def workflow_options(self, value: dict[str, Any]) -> None:
        self.parameters["workflow_options"] = value

    @property
    def workflow_results(self) -> dict:
        return self.results

    @workflow_results.setter
    def workflow_results(self, value: dict) -> None:
        self.results = value

    def run_executable(
        self, auto: WorkflowAutomation
    ) -> dict[tuple[str, ...], WorkflowResult]:
        """Run an experiment workflow.

        Arguments:
            auto: The workflow automation instance.

        Returns:
            A dictionary of workflow results, keyed by quantum elements.
        """
        # Prepare quantum elements
        quantum_elements = self.quantum_elements.copy()

        for node in self.nodes.values():
            if node.status in Status.inactive() and node.key in quantum_elements:
                quantum_elements.remove(node.key)

        quantum_elements_tuple = tuple(quantum_elements)

        if len(quantum_elements) == 1:  # the type needs to match workflow parameters
            quantum_elements = quantum_elements[0]

        # Prepare element workflow parameters
        grouped_element_workflow_parameters = group_element_workflow_parameters(
            self.element_workflow_parameters, quantum_elements
        )

        # Prepare workflow options
        built_workflow_options = self.workflow_builder.options()
        if self.workflow_options:
            for key, value in self.workflow_options.items():
                set_option_method = getattr(built_workflow_options, key)
                set_option_method(value)

        # Build experiment workflow
        workflow = self.workflow_builder(
            auto.session,
            auto.qpu,
            quantum_elements,
            temporary_parameters=self.temporary_qpu_parameters,
            options=built_workflow_options,
            **grouped_element_workflow_parameters,
            **self.common_workflow_parameters,
        )

        # Set node statuses (pre run)
        for q in quantum_elements_tuple:
            node = self.nodes[q]
            node.status = Status.RUNNING

        # Run experiment workflow
        workflow_result = workflow.run()
        self.workflow_results[quantum_elements_tuple] = workflow_result

        # Get evaluation output
        self.eval_outputs = get_eval_outputs(self.workflow_results)
        eval_successes = {k: v["success"] for k, v in self.eval_outputs.items()}

        # Set node statuses (post run)
        for node_key, node in self.nodes.items():
            if eval_successes and node_key in eval_successes:
                eval_success = eval_successes[node_key]
                node.status = Status.PASSED if eval_success else Status.FAILED
            elif node.status in Status.active():
                node.status = Status.PASSED

        return {quantum_elements_tuple: workflow_result}
