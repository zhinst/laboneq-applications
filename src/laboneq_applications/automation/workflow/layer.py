# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""The workflow automation layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import attrs
from laboneq.automation import AutomationLayer
from laboneq.automation import AutomationStatus as Status
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.quantum import QPU, QuantumParameters
from laboneq.workflow import WorkflowBuilder, WorkflowResult

from laboneq_applications.automation.workflow.node import WorkflowNode
from laboneq_applications.automation.workflow.utils import (
    get_eval_outputs,
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
        eval_outputs: The layer evaluation outputs.
    """

    qpu: QPU | None = None
    eval_outputs: dict[str, dict[str, bool]] = attrs.field(factory=dict, init=False)

    @property
    def nodes(self) -> dict[str | tuple[str, ...], WorkflowNode]:
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
    def quantum_elements(self) -> list[str | tuple[str, ...]]:
        """The quantum elements."""
        return self.node_keys

    @quantum_elements.setter
    def quantum_elements(self, value: list[str | tuple[str, ...]]) -> None:
        """The quantum elements setter."""
        self.node_keys = value

    @property
    def workflow_parameters(self) -> dict[str, dict[str, Any]]:
        """The workflow parameters."""
        return self.parameters.get("workflow_parameters", {})

    @workflow_parameters.setter
    def workflow_parameters(self, value: dict[str, dict[str, Any]]) -> None:
        """The workflow parameters setter."""
        self.parameters["workflow_parameters"] = value

    @property
    def element_workflow_parameters(self) -> dict[str, dict[str, Any]]:
        """The element workflow parameters."""
        wf_params = self.parameters.get("workflow_parameters", {})
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

    @property
    def common_workflow_parameters(self) -> dict[str, Any]:
        """The common workflow parameters."""
        return self.parameters.get("workflow_parameters", {}).get("__common__", {})

    @common_workflow_parameters.setter
    def common_workflow_parameters(self, value: dict[str, Any]) -> None:
        """The common workflow parameters setter."""
        self.parameters["workflow_parameters"]["__common__"] = value

    @property
    def evaluation_parameters(self) -> dict[str, Any]:
        """The evaluation parameters."""
        return self.parameters.get("evaluation_parameters", {})

    @evaluation_parameters.setter
    def evaluation_parameters(self, value: dict[str, Any]) -> None:
        """The evaluation parameters setter."""
        self.parameters["evaluation_parameters"] = value

    @property
    def temporary_parameters(
        self,
    ) -> dict[str | tuple[str, str, str], dict | QuantumParameters]:
        """The temporary parameters."""
        return self.parameters.get("temporary_parameters", {})

    @temporary_parameters.setter
    def temporary_parameters(
        self, value: dict[str | tuple[str, str, str], dict | QuantumParameters]
    ) -> None:
        """The temporary parameters setter."""
        self.parameters["temporary_parameters"] = value

    @property
    def options(self) -> dict[str, Any]:
        """The options."""
        return self.parameters.get("options", {})

    @options.setter
    def options(self, value: dict[str, Any]) -> None:
        """The options setter."""
        self.parameters["options"] = value

    @property
    def workflow_results(self) -> dict:
        """The workflow results."""
        return self.results

    @workflow_results.setter
    def workflow_results(self, value: dict) -> None:
        """The workflow results setter."""
        self.results = value

    def run_executable(
        self,
        auto: WorkflowAutomation,
        quantum_elements: list[str | tuple[str, ...]] | None = None,
    ) -> dict[tuple[str, ...], WorkflowResult]:
        """Run an experiment workflow.

        Arguments:
            auto: The workflow automation instance.
            quantum_elements: A list of keys of quantum elements to use
                in the experiment workflow (optional). If no list is provided,
                then the workflow is run on all quantum elements in the layer.

        Returns:
            A dictionary of workflow results, keyed by quantum elements.
        """
        # Prepare quantum elements
        if quantum_elements is None:
            quantum_elements = self.quantum_elements
            storage_key = (
                f"{auto.timestamp}-{auto.name}",
                self.key,
            )
        else:
            quantum_elements_string = "_".join(
                "-".join(map(str, q)) if isinstance(q, tuple) else q
                for q in quantum_elements
            )
            storage_key = (
                f"{auto.timestamp}-{auto.name}",
                self.key,
                quantum_elements_string,
            )

        run_elements = [
            n.key
            for n in self.nodes.values()
            if n.status in Status.active() and n.key in quantum_elements
        ]

        quantum_elements_tuple = tuple(run_elements)

        if len(run_elements) == 1 and not isinstance(
            run_elements[0], tuple
        ):  # the type needs to match workflow parameters
            run_elements = run_elements[0]

        # Prepare workflow parameters
        grouped_element_workflow_parameters = group_element_workflow_parameters(
            self.element_workflow_parameters, run_elements
        )

        # Prepare evaluation parameters
        if self.evaluation_parameters:
            evaluation_parameters = {
                "evaluation_parameters": self.evaluation_parameters
            }
        else:
            evaluation_parameters = self.evaluation_parameters

        # Prepare options
        built_options = self.workflow_builder.options()
        if self.options:
            for key, value in self.options.items():
                set_option_method = getattr(built_options, key)
                set_option_method(value)

        # Build experiment workflow
        workflow = self.workflow_builder(
            auto.session,
            auto.qpu,
            run_elements,
            **grouped_element_workflow_parameters,
            **self.common_workflow_parameters,
            **evaluation_parameters,
            temporary_parameters=self.temporary_parameters,
            options=built_options,
        )
        workflow.storage_key = storage_key

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
        for q in quantum_elements_tuple:
            if q in eval_successes:
                self.nodes[q].status = (
                    Status.PASSED if eval_successes[q] else Status.FAILED
                )
            else:
                self.nodes[q].status = Status.PASSED

        return {quantum_elements_tuple: workflow_result}
