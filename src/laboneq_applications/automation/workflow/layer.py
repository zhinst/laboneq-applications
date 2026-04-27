# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""The workflow automation layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import attrs
from laboneq.automation import AutomationLayer, AutomationLayerResult, NodeKey
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.quantum import QPU, QuantumParameters
from laboneq.workflow import WorkflowBuilder

from laboneq_applications.automation.workflow.node import WorkflowNode
from laboneq_applications.automation.workflow.utils import (
    get_eval_successes,
    group_element_workflow_parameters,
)

if TYPE_CHECKING:
    from laboneq_applications.automation.workflow.automation import (
        WorkflowAutomation,
    )


@classformatter
@attrs.define
class WorkflowLayer(AutomationLayer):
    """A workflow layer in the automation framework.

    Attributes:
        qpu: The QPU. By default, the QPU from the `Automation` instance is used.
    """

    qpu: QPU | None = None

    @property
    def nodes(self) -> dict[NodeKey, WorkflowNode]:
        """The node dictionary."""
        for node_key in self.node_keys:
            if node_key not in self._node_lookup:
                if isinstance(node_key, str):
                    element_key = (node_key,)
                else:
                    element_key = tuple(node_key)
                deps = {
                    f"{layer_key}_{k}"
                    for layer_key in self.depends_on
                    if layer_key != "root"
                    for k in element_key
                }
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
        """The workflow builder."""
        return self.function

    @workflow_builder.setter
    def workflow_builder(self, value: WorkflowBuilder | None) -> None:
        """The workflow builder setter."""
        self.function = value

    @property
    def quantum_elements(self) -> list[NodeKey]:
        """The quantum elements, respecting any override and selection."""
        return self.target_node_keys

    @quantum_elements.setter
    def quantum_elements(self, value: list[NodeKey]) -> None:
        """The quantum elements setter."""
        self.node_keys = value

    @property
    def active_quantum_elements(self) -> list[NodeKey]:
        """The active quantum elements, respecting any override and selection."""
        return self.active_node_keys

    @property
    def workflow_parameters(self) -> dict[str, dict[str, Any]]:
        """The workflow parameters, respecting any override."""
        return self.target_parameters.get("workflow_parameters", {})

    @workflow_parameters.setter
    def workflow_parameters(self, value: dict[str, dict[str, Any]]) -> None:
        """The workflow parameters setter."""
        self.parameters["workflow_parameters"] = value

    @workflow_parameters.deleter
    def workflow_parameters(self) -> None:
        """The workflow parameters deleter."""
        del self.parameters["workflow_parameters"]

    @property
    def element_workflow_parameters(self) -> dict[str, dict[str, Any]]:
        """The element workflow parameters, respecting any override."""
        wf_params = self.target_parameters.get("workflow_parameters", {})
        return {k: v for k, v in wf_params.items() if k != "__common__"}

    @element_workflow_parameters.setter
    def element_workflow_parameters(self, value: dict[str, dict[str, Any]]) -> None:
        """The element workflow parameters setter."""
        common_wf_params = self.parameters.get("workflow_parameters", {}).get(
            "__common__", {}
        )
        wf_params = {k: v for k, v in value.items() if k != "__common__"}
        if common_wf_params:
            wf_params["__common__"] = common_wf_params
        self.parameters["workflow_parameters"] = wf_params

    @element_workflow_parameters.deleter
    def element_workflow_parameters(self) -> None:
        """The element workflow parameters deleter."""
        keys_to_delete = [
            k
            for k in self.parameters.get("workflow_parameters", {})
            if k != "__common__"
        ]
        for k in keys_to_delete:
            del self.parameters["workflow_parameters"][k]

    @property
    def common_workflow_parameters(self) -> dict[str, Any]:
        """The common workflow parameters, respecting any override."""
        return self.target_parameters.get("workflow_parameters", {}).get(
            "__common__", {}
        )

    @common_workflow_parameters.setter
    def common_workflow_parameters(self, value: dict[str, Any]) -> None:
        """The common workflow parameters setter."""
        self.parameters.setdefault("workflow_parameters", {})["__common__"] = value

    @common_workflow_parameters.deleter
    def common_workflow_parameters(self) -> None:
        """The common workflow parameters deleter."""
        del self.parameters["workflow_parameters"]["__common__"]

    @property
    def evaluation_parameters(self) -> dict[str, Any]:
        """The evaluation parameters, respecting any override."""
        return self.target_parameters.get("evaluation_parameters", {})

    @evaluation_parameters.setter
    def evaluation_parameters(self, value: dict[str, Any]) -> None:
        """The evaluation parameters setter."""
        self.parameters["evaluation_parameters"] = value

    @evaluation_parameters.deleter
    def evaluation_parameters(self) -> None:
        """The evaluation parameters deleter."""
        del self.parameters["evaluation_parameters"]

    @property
    def temporary_parameters(
        self,
    ) -> dict[str | tuple[str, str, str], dict | QuantumParameters]:
        """The temporary parameters, respecting any override."""
        return self.target_parameters.get("temporary_parameters", {})

    @temporary_parameters.setter
    def temporary_parameters(
        self, value: dict[str | tuple[str, str, str], dict | QuantumParameters]
    ) -> None:
        """The temporary parameters setter."""
        self.parameters["temporary_parameters"] = value

    @temporary_parameters.deleter
    def temporary_parameters(self) -> None:
        """The temporary parameters deleter."""
        del self.parameters["temporary_parameters"]

    @property
    def options(self) -> dict[str, Any]:
        """The options, respecting any override."""
        return self.target_parameters.get("options", {})

    @options.setter
    def options(self, value: dict[str, Any]) -> None:
        """The options setter."""
        self.parameters["options"] = value

    @options.deleter
    def options(self) -> None:
        """The options deleter."""
        del self.parameters["options"]

    @property
    def workflow_results(self) -> dict:
        """The workflow results."""
        return self.results

    @workflow_results.setter
    def workflow_results(self, value: dict) -> None:
        """The workflow results setter."""
        self.results = value

    def run_executable_core(
        self,
        auto: WorkflowAutomation,
    ) -> AutomationLayerResult:
        """Run an experiment workflow.

        Arguments:
            auto: The workflow automation instance.

        Returns:
            The automation layer result.
        """
        # Prepare storage key
        storage_key = (
            f"{auto.timestamp}-{auto.name}",
            self.key,
        )
        if self.target_node_keys != self.node_keys:
            quantum_elements_string = "_".join(
                q._key_str for q in self.target_nodes.values()
            )
            storage_key = (*storage_key, quantum_elements_string)

        # Prepare quantum elements
        wf_active_quantum_elements = self.active_quantum_elements
        if len(wf_active_quantum_elements) == 1 and not isinstance(
            wf_active_quantum_elements[0], tuple
        ):  # the active quantum elements type needs to match experiment workflows
            wf_active_quantum_elements = wf_active_quantum_elements[0]

        # Prepare workflow parameters
        grouped_element_workflow_parameters = group_element_workflow_parameters(
            self.element_workflow_parameters, wf_active_quantum_elements
        )

        # Prepare evaluation parameters
        evaluation_parameters = (
            {"evaluation_parameters": self.evaluation_parameters}
            if self.evaluation_parameters
            else {}
        )

        # Prepare options
        built_options = self.workflow_builder.options()
        for key, value in self.options.items():
            getattr(built_options, key)(value)

        # Build experiment workflow
        workflow = self.workflow_builder(
            auto.session,
            auto.qpu,
            wf_active_quantum_elements,
            **grouped_element_workflow_parameters,
            **self.common_workflow_parameters,
            **evaluation_parameters,
            temporary_parameters=self.temporary_parameters,
            options=built_options,
        )
        workflow.storage_key = storage_key

        # Run experiment workflow
        workflow_result = workflow.run()

        self.workflow_results[tuple(self.active_quantum_elements)] = workflow_result
        eval_successes = get_eval_successes(workflow_result) or {}

        return AutomationLayerResult(
            results={tuple(self.active_quantum_elements): workflow_result},
            successes=eval_successes,
        )
