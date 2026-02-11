# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""LabOne Q workflow automation subclass."""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any

import networkx as nx
from laboneq._automation import Automation
from laboneq._automation import AutomationElementStatus as Status
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.quantum import QuantumParameters
from laboneq.workflow import WorkflowOptions
from laboneq.workflow.opts import OptionBuilder
from laboneq.workflow.timestamps import local_timestamp

from laboneq_applications._automation.utils import make_json_serializable, nested_update
from laboneq_applications._automation.workflow import workflow_logic
from laboneq_applications._automation.workflow.workflow_logic import WorkflowLogic

if TYPE_CHECKING:
    from laboneq_applications._automation.workflow.workflow_layer import WorkflowLayer


@classformatter
class WorkflowAutomation(Automation):
    """The LabOne Q workflow automation framework."""

    def run(self) -> None:
        """Run the automation framework (layer-by-layer).

        Run the automation framework in parallel (layer-by-layer), starting from the
        layer after `root`.

        !!! note
            We continuously re-try to run a layer until the maximum fail count is
            reached, or some other value error occurs.
        """
        new_layer_key = next(self.layer_keys(include_root=False))
        while new_layer_key != "__end__":
            try:
                _, new_layer_key, _ = self.run_layer(new_layer_key)
            except ValueError as err:  # noqa: PERF203
                # Must catch per-iteration: stop execution immediately on error
                warnings.warn(f"{err}", stacklevel=2)
                break

    def run_sequential(self) -> None:
        """Run the automation framework (node-by-node).

        Run the automation framework sequentially (node-by-node), sorted using a
        topological sort.

        !!! note
            We continuously re-try to run a node until the maximum fail count is
            reached, or some other value error occurs.
        """
        for key in nx.topological_sort(self._node_graph):
            node = self._node_lookup[key]
            if node and hasattr(node, "status") and node.status != Status.EMPTY:
                while node.status != Status.PASSED:
                    try:
                        self.run_node(key)
                    except ValueError as err:  # noqa: PERF203
                        warnings.warn(f"{err}", stacklevel=2)
                        continue

    def run_node(
        self, key: str, *, force: bool = False
    ) -> dict[str, dict[str, bool]] | None:
        """Run the automation node.

        Arguments:
            key: The node key.
            force: Whether to force run the node. All node checks are ignored, such as
                node status, node dependencies, and node fail count.

        Returns:
            The evaluation output.
        """
        # Get the node
        node = self.get_node(key)

        # Check the node
        if not force and not self.run_node_candidate(node):
            return None

        # Run the node
        workflow_result = node.run_executable(self)
        node.timestamp = local_timestamp()

        # Check for failures
        fail_flag = False
        task_list = [t.name for t in workflow_result.tasks]
        if "evaluate_experiment" in task_list:
            eval_output = workflow_result.tasks["evaluate_experiment"].output
            for eval_flags in eval_output.values():
                if eval_flags["success"] is False:
                    fail_flag = True
        else:
            warnings.warn(
                f"No `evaluate_experiment` task found in the experiment workflow "
                f"{node.workflow_builder._name}. "
                f"Setting node.status to 'passed'.",
                stacklevel=2,
            )

        # Set the node status
        if fail_flag:
            node.status = Status.FAILED
            node.fail_count += 1
            # Deactivate if max fail count reached
            if node.fail_count >= node.max_fail_count:
                self.deactivate_node(node.key)
        else:
            node.status = Status.PASSED

        return eval_output

    @staticmethod
    def _set_temp_parameters(
        layer: WorkflowLayer,
        quantum_elements: str | list[str] | None,
        workflow_parameters: dict[str, dict[str, Any]] | None = None,
        general_workflow_parameters: dict[str, Any] | None = None,
        temporary_qpu_parameters: dict[
            str | tuple[str, str, str], dict | QuantumParameters
        ]
        | None = None,
        workflow_options: WorkflowOptions | dict[str, Any] | None = None,
        logic: WorkflowLogic | None = None,
    ) -> str | list[str] | None:
        quantum_elements = (
            [quantum_elements]
            if isinstance(quantum_elements, str)
            else quantum_elements
        )
        if quantum_elements and not set(quantum_elements).issubset(
            set(layer.quantum_elements)
        ):
            raise ValueError(
                f"The set of quantum elements {set(quantum_elements)} "
                f"is not in the layer. "
                f"The set of quantum elements in the layer is "
                f"{set(layer.quantum_elements)}."
            )
        if quantum_elements is not None:
            layer.quantum_elements = quantum_elements

        if workflow_parameters is not None:
            # Cannot update properties in-place
            layer_workflow_parameters = layer.workflow_parameters
            layer_workflow_parameters.update(workflow_parameters)
            layer.workflow_parameters = layer_workflow_parameters
        if general_workflow_parameters is not None:
            # Cannot update properties in-place
            layer_general_workflow_parameters = layer.general_workflow_parameters
            layer_general_workflow_parameters.update(general_workflow_parameters)
            layer.general_workflow_parameters = layer_general_workflow_parameters
        if temporary_qpu_parameters is not None:
            layer.temporary_qpu_parameters = temporary_qpu_parameters
        if isinstance(workflow_options, dict):
            options = layer.workflow_builder.options()
            for key, value in workflow_options.items():
                set_option_method = getattr(options, key)
                set_option_method(value)
            layer.workflow_options = options
        elif isinstance(workflow_options, OptionBuilder):
            layer.workflow_options = workflow_options
        if logic is not None:
            layer.logic = logic

        return layer

    def run_layer(  # noqa: C901, PLR0912
        self,
        layer_key: str,
        *,
        quantum_elements: str | list[str] | None = None,
        workflow_parameters: dict[str, dict[str, Any]] | None = None,
        general_workflow_parameters: dict[str, Any] | None = None,
        temporary_qpu_parameters: dict[
            str | tuple[str, str, str], dict | QuantumParameters
        ]
        | None = None,
        workflow_options: WorkflowOptions | dict[str, Any] | None = None,
        force: bool = False,
        logic: WorkflowLogic | None = None,
    ) -> tuple[dict[str, dict[str, bool]] | None, str, dict | None]:
        """Run the automation layer.

        Arguments:
            layer_key: The layer key.
            quantum_elements: The subset of quantum elements to run the layer on
                (optional).
            workflow_parameters: The dictionary of workflow parameters, keyed by
                quantum element UID (optional).
            general_workflow_parameters: The dictionary of general workflow parameters
                (optional).
            temporary_qpu_parameters: The temporary QPU parameters (optional).
            workflow_options: The workflow options (optional).
            force: Whether to force run the layer (optional). All layer checks are
                ignored, such as layer status, layer dependencies, and layer fail count.
            logic: The workflow decision logic (optional).

        Returns:
            Tuple with the evaluation output dictionary, the next layer key and
            the new parameters.
        """
        # Get the layer
        layer = self.get_layer(layer_key)

        # Set temporary parameters
        layer_quantum_elements = layer.quantum_elements
        layer_workflow_parameters = layer.workflow_parameters
        layer_general_workflow_parameters = layer.general_workflow_parameters
        layer_temporary_qpu_parameters = layer.temporary_qpu_parameters
        layer_workflow_options = layer.workflow_options
        layer_logic = layer.logic
        if any(
            [
                quantum_elements,
                workflow_parameters,
                general_workflow_parameters,
                temporary_qpu_parameters,
                workflow_options,
                logic,
            ]
        ):
            layer = self._set_temp_parameters(
                layer,
                quantum_elements,
                workflow_parameters,
                general_workflow_parameters,
                temporary_qpu_parameters,
                workflow_options,
                logic,
            )

        # Check the layer
        if not force and not self.run_layer_candidate(layer):
            new_layer_key = self.next_layer_key(layer_key)
            return None, new_layer_key, None

        # Run the layer
        if not layer.sequential:
            layer.run_executable(self)
        else:
            for node in self.nodes(layer_key=layer_key):
                if node.quantum_elements in layer.quantum_elements and (
                    self.run_node_candidate(node) or force
                ):
                    node.run_executable(self)
            layer.workflow_results = [
                n.workflow_result for n in layer.nodes if n.workflow_result is not None
            ]
            layer.eval_outputs = {}
            for workflow_result in layer.workflow_results:
                eval_output = layer._layer_evaluation_output([workflow_result])
                layer.eval_outputs.update(eval_output)
            layer.timestamp = local_timestamp()

        # Increment the layer fail count (if needed)
        if layer.status != Status.PASSED:
            layer.fail_count += 1
        else:
            layer.success_count += 1

        # Deactivate failed nodes (if needed)
        if layer.fail_count == layer.max_fail_count:
            for node in layer.nodes:
                if node.status == Status.FAILED:
                    self.deactivate_node(node.key)

        # Execute workflow logic
        if layer.status in [Status.PASSED, Status.DEACTIVATED]:
            try:
                default_next_layer_key = self.next_layer_key(layer_key)
            except IndexError:
                default_next_layer_key = "__end__"
        else:
            default_next_layer_key = layer_key
        new_layer_key, new_params = default_next_layer_key, None
        if layer.logic and (
            layer.status not in [Status.PASSED, Status.DEACTIVATED]
            or (
                layer.logic.iterations is not None
                and (
                    layer.fail_count + layer.success_count - 1 < layer.logic.iterations
                )
            )
        ):
            new_layer_key, new_params = layer.logic.run_executable(layer)
            nested_update(layer_workflow_parameters, new_params)

        # Reset temporary parameters
        layer.quantum_elements = layer_quantum_elements
        layer.workflow_parameters = layer_workflow_parameters
        layer.general_workflow_parameters = layer_general_workflow_parameters
        layer.temporary_qpu_parameters = layer_temporary_qpu_parameters
        layer.workflow_options = layer_workflow_options
        layer.logic = layer_logic

        return layer.eval_outputs, new_layer_key, new_params

    def reset(self) -> None:
        """Reset the automation framework.

        Resets all quantum element status parameters to their default values.
        """
        for node in self.nodes():
            if node.status in [Status.PASSED, Status.FAILED, Status.DEACTIVATED]:
                node.status = Status.READY
            node.fail_count = 0
            node.success_count = 0
            node.timestamp = None
            node.workflow_result = None
        for layer in self.layers():
            layer.success_count = 0
            layer.fail_count = 0
            layer.timestamp = None
            layer.workflow_results = None

    def extract_automation_parameters(
        self,
        layer_key: str,
        quantum_element: str | None = None,
        *,
        cal_params: dict | None = None,
    ) -> tuple:
        """Safely extract automation parameters.

        Arguments:
            layer_key: The key of the workflow layer.
            quantum_element: The quantum element UID (optional).
            cal_params: The calibration parameters dictionary (optional). By default,
                we take the calibration parameters from the automation instance.

        Returns:
            A tuple of workflow parameters, general workflow parameters,
            temporary QPU parameters, workflow options, and logic.
        """
        if cal_params is None:
            cal_params = self.automation_parameters

        if cal_params is None:
            workflow_parameters = None
            general_workflow_parameters = None
            temporary_qpu_parameters = None
            workflow_options = None
            logic = None
        else:
            if quantum_element:
                workflow_parameters = cal_params[layer_key][quantum_element]
            else:
                workflow_parameters = {
                    k: v
                    for k, v in cal_params[layer_key].items()
                    if k in [q.uid for q in self.qpu.quantum_elements]
                }
            general_workflow_parameters = {
                k: v
                for k, v in cal_params[layer_key].items()
                if k
                not in {q.uid for q in self.qpu.quantum_elements}
                | {"temporary_parameters", "options", "logic"}
            }
            if "temporary_parameters" in cal_params[layer_key]:
                temporary_qpu_parameters = cal_params[layer_key]["temporary_parameters"]
            else:
                temporary_qpu_parameters = None
            if "options" in cal_params[layer_key]:
                workflow_options = cal_params[layer_key]["options"]
            else:
                workflow_options = None
            if "logic" in cal_params[layer_key]:
                logic_class = getattr(
                    workflow_logic, cal_params[layer_key]["logic"]["class"]
                )
                logic_args = cal_params[layer_key]["logic"]["arguments"]
                logic = logic_class(**logic_args)
            else:
                logic = None

        return (
            workflow_parameters,
            general_workflow_parameters,
            temporary_qpu_parameters,
            workflow_options,
            logic,
        )

    def export_automation_parameters(self) -> None:
        """Export the automation parameters to a file.

        TODO: Expand this method to collect all parameters from automation elements.
        """
        timestamp = local_timestamp()
        name = self.__class__.__name__

        cal_params_serializable = make_json_serializable(self.automation_parameters)
        with open(f"{timestamp}_{name}_parameters.json", "w") as f:
            json.dump(cal_params_serializable, f, indent=4)
