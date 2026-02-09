# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""LabOne Q workflow layer subclass."""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Any

from laboneq._automation import AutomationLayer
from laboneq._automation.element import AutomationElementStatus as Status
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.quantum import QuantumParameters
from laboneq.workflow import WorkflowBuilder, WorkflowOptions, WorkflowResult
from laboneq.workflow.timestamps import local_timestamp

from laboneq_applications._automation.workflow.workflow_logic import WorkflowLogic
from laboneq_applications._automation.workflow.workflow_node import WorkflowNode

if TYPE_CHECKING:
    from laboneq_applications._automation.workflow.workflow_automation import (
        WorkflowAutomation,
    )


@classformatter
class WorkflowLayer(AutomationLayer):
    """A LabOne Q workflow layer in the automation framework.

    Attributes:
        key: The automation element key.
        depends_on: A list of automation element dependencies.
        qpu: The QPU to use (optional). If not specified, the QPU from the
            `Automation` instance is used.
        max_fail_count: The maximum number of allowed failures.
        time_valid: The time for which the automation element is reliably valid.
        time_until_invalid: The time until the automation element is invalid.
        status: The status of the automation element.
        fail_count: The number of failed runs.
        success_count: The number of successful runs.
        timestamp: The time the automation element was last run.
        sequential: Whether to execute the layer sequentially.
        workflow_builder: The workflow builder instance.
        quantum_elements: The list of quantum element UIDs. For multiple-qubit
            experiments, this may be a list of lists.
        workflow_parameters: The dictionary of workflow parameters, keyed by
            quantum element UID.
        general_workflow_parameters: The dictionary of general workflow parameters.
        temporary_qpu_parameters: The temporary QPU parameters,
        workflow_options: The workflow options.
        logic: The layer decision logic.
        node_builder: The node builder instance.
        empty_args: The arguments to build an empty node.
        nodes: The list of child workflow nodes.
        workflow_results: The list of workflow results.
    """

    def __init__(
        self,
        # compulsory workflow parameters
        workflow_builder: WorkflowBuilder,
        quantum_elements: list[str] | list[list[str]] | None,
        *,
        # optional workflow parameters
        workflow_parameters: dict[str, dict[str, Any]] | None = None,
        general_workflow_parameters: dict[str, Any] | None = None,
        temporary_qpu_parameters: dict[
            str | tuple[str, str, str], dict | QuantumParameters
        ]
        | None = None,
        workflow_options: WorkflowOptions | None = None,
        logic: WorkflowLogic | None = None,
        # automation layer parameters
        **kwargs,
    ) -> None:
        """Initialize LabOne Q workflow layer attributes.

        Arguments:
            workflow_builder: The workflow builder instance.
            quantum_elements: The list of quantum element UIDs. For multiple-qubit
                experiments, this may be a list of lists.
            workflow_parameters: The dictionary of workflow parameters, keyed by
                quantum element UID.
            general_workflow_parameters: The dictionary of general workflow parameters.
            temporary_qpu_parameters: The temporary QPU parameters.
            workflow_options: The workflow options.
            logic: The workflow decision logic.
            **kwargs: Automation layer parameters.

        This constructor also accepts the arguments of
        [`AutomationElement`][laboneq._automation.framework.element.AutomationElement]
        and [`AutomationLayer`][laboneq._automation.framework.layer.AutomationLayer].
        The arguments `key` and `depends_on` are compulsory.
        """
        # compulsory workflow parameters
        self.workflow_builder = workflow_builder
        self.quantum_elements = quantum_elements
        # optional workflow parameters
        self._workflow_parameters = workflow_parameters
        self.general_workflow_parameters = general_workflow_parameters
        self._temporary_qpu_parameters = temporary_qpu_parameters
        self._workflow_options = workflow_options
        self.logic = logic
        # automation layer parameters
        super().__init__(**kwargs)

        # empty node
        self.node_builder = WorkflowNode
        self.empty_args = {
            "workflow_builder": None,
            "quantum_elements": None,
            "key": "empty",
            "depends_on": [],
            "layer": self.key,
            "status": Status.EMPTY,
        }

        # node list
        node_list = []
        general_kwargs = {
            "qpu": self.qpu,
            "temporary_qpu_parameters": self._temporary_qpu_parameters,
        }
        if self.quantum_elements is None:
            general_kwargs["workflow_options"] = copy.deepcopy(self._workflow_options)
            node = WorkflowNode(
                self.workflow_builder,
                self.quantum_elements,
                key=f"{self.key}_node",
                depends_on=self.depends_on,
                layer=self.key,
                workflow_parameters=self._workflow_parameters,
                general_workflow_parameters=self.general_workflow_parameters,
                **general_kwargs,
            )
            node_list.append(node)
        elif isinstance(self.quantum_elements, list):
            for q in self.quantum_elements:
                if isinstance(q, list):
                    raise NotImplementedError("Multi-qubit experiment workflows.")

                if self._workflow_parameters and q in self._workflow_parameters:
                    workflow_params = self._workflow_parameters[q]
                else:
                    workflow_params = None

                deps = []
                for dep in self.depends_on:
                    if dep != self._ROOT:
                        deps.append(f"{dep}_{q}")
                    else:
                        deps.append(self._ROOT)

                general_kwargs["workflow_options"] = copy.deepcopy(
                    self._workflow_options
                )
                node = WorkflowNode(
                    self.workflow_builder,
                    q,
                    key=f"{self.key}_{q}",
                    depends_on=deps,
                    layer=self.key,
                    workflow_parameters=workflow_params,
                    general_workflow_parameters=self.general_workflow_parameters,
                    **general_kwargs,
                )
                node_list.append(node)
        self.nodes: list[WorkflowNode] = node_list

        # workflow results
        self.workflow_results: list[WorkflowResult] | None = None

        # evaluation outputs
        self.eval_outputs: list[dict[str, dict[str, bool]]] | None = None

    @property
    def workflow_parameters(self) -> dict[str, dict[str, Any]] | None:
        """Workflow parameters property.

        The workflow parameters of a layer are a dictionary of references to the
        workflow parameters of the child nodes. If any of the child node workflow
        parameters get updated, then the layer workflow parameters will be updated.

        Returns:
            The workflow parameters.
        """
        combined_dict = {}
        for node in self.nodes:
            if node.quantum_elements in self.quantum_elements:
                combined_dict[node.quantum_elements] = node.workflow_parameters
        return combined_dict

    @workflow_parameters.setter
    def workflow_parameters(self, value: dict[str, dict[str, Any]] | None) -> None:
        """Workflow parameters setter.

        If the layer workflow parameters are set, then this will automatically
        propagate to all the child nodes.

        !!! note
            If items in the layer dictionary are modified, this will not propagate to
            the child nodes.

        Raises:
            ValueError: If any of the quantum elements are not present in the layer.
        """
        for key in value:
            if key not in self.quantum_elements:
                raise ValueError(f"Quantum element {key} not in layer.")

        self._workflow_parameters = value
        self.update_node_parameters()

    @property
    def temporary_qpu_parameters(
        self,
    ) -> dict[str | tuple[str, str, str], dict | QuantumParameters] | None:
        """Temporary QPU parameters property.

        The temporary QPU parameters of a layer are a dictionary of references to the
        temporary QPU parameters of the child nodes. If any of the child node temporary
        QPU parameters get updated, then the layer temporary QPU parameters will be
        updated.

        Returns:
            The temporary QPU parameters.
        """
        combined_dict = {}
        for node in self.nodes:
            if node.temporary_qpu_parameters is not None:
                combined_dict[node.quantum_elements] = node.temporary_qpu_parameters[
                    node.quantum_elements
                ]
        if combined_dict:
            return combined_dict
        return None

    @temporary_qpu_parameters.setter
    def temporary_qpu_parameters(
        self, value: dict[str | tuple[str, str, str], dict | QuantumParameters] | None
    ) -> None:
        """Temporary QPU parameters setter.

        If the layer temporary QPU parameters are set, then this will automatically
        propagate to all the child nodes.

        !!! note
            If items in the layer dictionary are modified, this will not propagate to
            the child nodes.
        """
        self._temporary_qpu_parameters = value
        self.update_node_parameters()

    @staticmethod
    def _combine_active_reset(
        options: WorkflowOptions, node_options: list[WorkflowOptions]
    ) -> None:
        """Combines the active reset options of the nodes ("any" logic).

        !!! note
            This is a helper function for `workflow_options`.

        Arguments:
            options: The workflow layer options.
            node_options: The list of workflow node options.
        """
        combined_active_reset = True
        for opt in node_options:
            if opt is not None:
                for item in opt.active_reset:
                    if item.option.active_reset is False:
                        combined_active_reset = False
                        break
            if not combined_active_reset:
                break
        set_option_method = options.active_reset
        set_option_method(combined_active_reset)

    @staticmethod
    def _combine_active_reset_repetitions(
        options: WorkflowOptions, node_options: list[WorkflowOptions]
    ) -> None:
        """Combines the active reset repetitions options of the nodes ("max" logic).

        !!! note
            This is a helper function for `workflow_options`.

        Arguments:
            options: The workflow layer options.
            node_options: The list of workflow node options.
        """
        active_reset_repetitions_list = []
        for opt in node_options:
            if opt is not None:
                for item in opt.active_reset_repetitions:
                    active_reset_repetitions_list.append(  # noqa: PERF401
                        item.option.active_reset_repetitions
                    )
        combined_active_reset_repetitions = max(active_reset_repetitions_list)
        set_option_method = options.active_reset_repetitions
        set_option_method(combined_active_reset_repetitions)

    @staticmethod
    def _combine_count(
        options: WorkflowOptions, node_options: list[WorkflowOptions]
    ) -> None:
        """Combines the count options of the nodes ("max" logic).

        !!! note
            This is a helper function for `workflow_options`.

        Arguments:
            options: The workflow layer options.
            node_options: The list of workflow node options.
        """
        count_list = []
        for opt in node_options:
            if opt is not None:
                for item in opt.count:
                    count_list.append(item.option.count)  # noqa: PERF401
        combined_count = max(count_list)
        set_option_method = options.count
        set_option_method(combined_count)

    @property
    def workflow_options(self) -> WorkflowOptions | None:
        """Workflow options property.

        The workflow options of a layer is a property of the workflow options of the
        child nodes. If any of the child node workflow options get updated, then the
        workflow options of the layer may be updated.

        Currently, the following rules are implemented:
        - If the `active_reset` options for any of the child nodes is `False`, then the
        `active_reset` option for the layer will be `False`.
        - The `active_reset_repetitions` option for the layer is the maximum of the
        `active_reset_repetitions` options for the child nodes.
        - The `count` option for the layer is the maximum of the `count` options for the
        child nodes.

        Returns:
            The workflow options.
        """
        node_options = []
        for node in self.nodes:
            node_options.append(node.workflow_options)  # noqa: PERF401

        if self._workflow_options is None:
            if any(item is not None for item in node_options):
                options = self.workflow_builder.options()
            else:
                return None
        else:
            options = self._workflow_options

        # Combine selected options of nodes
        self._combine_active_reset(options, node_options)
        self._combine_active_reset_repetitions(options, node_options)
        self._combine_count(options, node_options)

        return options

    @workflow_options.setter
    def workflow_options(self, value: WorkflowOptions | None) -> None:
        """Workflow options setter.

        If the layer workflow options are set, then this will automatically
        propagate to all the child nodes.

        !!! note
            If values in the layer workflow options are modified, this will not
            propagate to the child nodes.
        """
        self._workflow_options = value
        self.update_node_parameters()

    def __repr__(self) -> str:
        return (
            f"<{type(self).__qualname__} "
            f"key={self.key} "
            f"depends_on={self.depends_on} "
            f"qpu={self.qpu!r} "
            f"max_fail_count = {self.max_fail_count} "
            f"time_valid = {self.time_valid} "
            f"time_until_invalid = {self.time_until_invalid} "
            f"status = {self.status} "
            f"fail_count = {self.fail_count} "
            f"success_count = {self.success_count} "
            f"timestamp = {self.timestamp} "
            f"sequential = {self.sequential} "
            f"workflow={self.workflow_builder._name} "
            f"quantum_elements={self.quantum_elements} "
            f"workflow_parameters={self.workflow_parameters!r} "
            f"general_workflow_parameters={self.general_workflow_parameters!r} "
            f"temporary_qpu_parameters={self.temporary_qpu_parameters} "
            f"workflow_options={self.workflow_options} "
            f"logic={self.logic} "
            f"node_builder={self.node_builder} "
            f"empty_args={self.empty_args} "
            f"nodes={self.nodes} "
            f"workflow_results={self.workflow_results}"
            f">"
        )

    def __rich_repr__(self):
        yield "key", self.key
        yield "depends_on", self.depends_on
        yield "qpu", self.qpu
        yield "max_fail_count", self.max_fail_count
        yield "time_valid", self.time_valid
        yield "time_until_invalid", self.time_until_invalid
        yield "status", self.status
        yield "fail_count", self.fail_count
        yield "success_count", self.success_count
        yield "timestamp", self.timestamp
        yield "sequential", self.sequential
        yield "workflow", self.workflow_builder._name
        yield "quantum_elements", self.quantum_elements
        yield "workflow_parameters", self.workflow_parameters
        yield "general_workflow_parameters", self.general_workflow_parameters
        yield "temporary_qpu_parameters", self.temporary_qpu_parameters
        yield "workflow_options", self.workflow_options
        yield "logic", self.logic
        yield "node_builder", self.node_builder
        yield "empty_args", self.empty_args
        yield "nodes", self.nodes
        yield "workflow_results", self.workflow_results

    def add_automation_parameters(self, auto: WorkflowAutomation) -> None:  # noqa: C901
        """Add automation parameters.

        Arguments:
            auto: The workflow automation instance.

        !!! note
            Parameters defined directly on the workflow layer take priority.
        """
        # Extract automation parameters
        (
            workflow_parameters,
            general_workflow_parameters,
            temporary_qpu_parameters,
            workflow_options,
            logic,
        ) = auto.extract_automation_parameters(self.key)

        # Define the QPU
        if self.qpu is None:
            self.qpu = auto.qpu

        # Define the workflow parameters
        if workflow_parameters is not None:
            if self._workflow_parameters is not None:
                workflow_parameters.update(self._workflow_parameters)
            self._workflow_parameters = workflow_parameters

        # Define the general workflow parameters
        if general_workflow_parameters is not None:
            if self.general_workflow_parameters is not None:
                general_workflow_parameters.update(self.general_workflow_parameters)
            self.general_workflow_parameters = general_workflow_parameters

        # Define the temporary parameters
        if temporary_qpu_parameters is not None:
            if self._temporary_qpu_parameters is not None:
                temporary_qpu_parameters.update(self._temporary_qpu_parameters)
            self._temporary_qpu_parameters = temporary_qpu_parameters

        # Define the options
        if self._workflow_options is None:
            options = self.workflow_builder.options()
            if workflow_options is not None:
                for key, value in workflow_options.items():
                    set_option_method = getattr(options, key)
                    set_option_method(value)
            self._workflow_options = options

        # Define the logic
        if logic is not None:
            self.logic = logic

        # Update node parameters
        self.update_node_parameters()

    def update_node_parameters(self) -> None:
        """Update node parameters.

        After the parameters for a layer have been set, propagate these parameters to
        the child nodes.
        """
        for node in self.nodes:
            node.qpu = self.qpu
            if (
                self._workflow_parameters is not None
                and node.quantum_elements in self._workflow_parameters
            ):
                node.workflow_parameters = self._workflow_parameters[
                    node.quantum_elements
                ]
            else:
                node.workflow_parameters = None
            node.general_workflow_parameters = self.general_workflow_parameters
            node.temporary_qpu_parameters = self._temporary_qpu_parameters
            node.workflow_options = copy.deepcopy(self._workflow_options)
            node.logic = self.logic

    @staticmethod
    def _convert_workflow_parameters(
        workflow_parameters: dict[str, dict[str, Any]] | None,
        quantum_elements: str | list[str],
    ) -> dict[str, list[Any]]:
        """Convert workflow parameters to LabOne Q experiment workflow list format.

        In the workflow parameters dictionary, the primary key is the quantum element
        UID. However, for LabOne Q experiment workflows, we need the parameter values
        as a list if there are multiple quantum elements involved.

        !!! note
            This is a helper function for `run_executable`.

        Arguments:
            workflow_parameters: The workflow parameters dictionary, keyed by quantum
                element UID.
            quantum_elements: The target quantum elements.

        Returns:
            The workflow parameters dictionary, in a list format, compatible with
            LabOne Q experiment workflows.
        """
        if workflow_parameters is not None:
            if isinstance(quantum_elements, str):
                grouped_workflow_parameters = workflow_parameters[quantum_elements]
            else:
                # Collect all unique parameter keys from all qubits
                secondary_keys = set()
                for k, qb_params in workflow_parameters.items():
                    if k in quantum_elements:
                        secondary_keys.update(qb_params.keys())
                grouped_workflow_parameters = {}
                for key in secondary_keys:
                    grouped_workflow_parameters[key] = []
                    for k, d in workflow_parameters.items():
                        if k in quantum_elements:
                            if key in d:
                                grouped_workflow_parameters[key].append(d[key])
                            else:
                                grouped_workflow_parameters[key].append(None)
        else:
            grouped_workflow_parameters = {}

        return grouped_workflow_parameters

    def run_executable(self, auto: WorkflowAutomation) -> list[WorkflowResult]:
        """Run a LabOne Q experiment workflow.

        Arguments:
            auto: The workflow automation instance.

        Returns:
            A list of workflow results.

        !!! note
            The session, QPU, and calibration parameters may be read from the
            automation framework.
        """
        # Check QPU
        if self.qpu is None:
            raise ValueError("No QPU specified.")

        # Check quantum elements
        if self.quantum_elements is None:
            raise ValueError("No quantum elements specified.")
        if isinstance(self.quantum_elements, list) and any(
            isinstance(item, list) for item in self.quantum_elements
        ):
            raise NotImplementedError("Multi-qubit experiment workflows.")
        quantum_elements = (
            self.quantum_elements.copy()
            if isinstance(self.quantum_elements, list)
            else self.quantum_elements
        )

        # Remove failed quantum elements
        for node in self.nodes:
            if (
                node.status == Status.DEACTIVATED
                and node.quantum_elements in quantum_elements
            ):
                quantum_elements.remove(node.quantum_elements)

        # Unpack lists of unit length
        if len(quantum_elements) == 1:  # type needs to match sweep parameters
            quantum_elements = quantum_elements[0]

        # Convert workflow parameters
        grouped_workflow_parameters = self._convert_workflow_parameters(
            self.workflow_parameters, quantum_elements
        )

        general_workflow_parameters = self.general_workflow_parameters

        # Build experiment workflow
        workflow = self.workflow_builder(
            auto.session,
            self.qpu,
            quantum_elements,
            temporary_parameters=self.temporary_qpu_parameters,
            options=self.workflow_options,
            **grouped_workflow_parameters,
            **general_workflow_parameters,
        )

        # Run experiment workflow
        workflow_result = workflow.run()

        self.workflow_results = [workflow_result]
        self.timestamp = local_timestamp()

        # Get the evaluation output
        eval_outputs = self._layer_evaluation_output(self.workflow_results)
        self.eval_outputs = eval_outputs
        eval_successes = {
            f"{self.key}_{k}": v["success"] for k, v in eval_outputs.items()
        }

        for node in self.nodes:
            # Set the node statuses and increment node fail count (if needed)
            if eval_successes and node.key in eval_successes:
                eval_success = eval_successes[node.key]
                node.status = Status.PASSED if eval_success else Status.FAILED
                node.fail_count += 1 if not eval_success else 0
                node.success_count += 1 if eval_success else 0
                node.timestamp = self.timestamp
                node.workflow_result = workflow_result
            # If there is no `evaluate_experiment` task, set the active nodes to PASSED
            elif node.status not in [Status.EMPTY, Status.DEACTIVATED]:
                node.status = Status.PASSED
                node.timestamp = self.timestamp
                node.workflow_result = workflow_result
                node.success_count += 1

        return [workflow_result]

    @staticmethod
    def _layer_evaluation_output(
        workflow_results: list[WorkflowResult],
    ) -> dict[str, dict[str, bool]]:
        """Process the workflow results into an evaluation output dictionary.

        !!! note
            This is a helper function for `run_executable`.

        Arguments:
            workflow_results: The workflow results.

        Returns:
            The evaluation output.
        """
        eval_outputs = {}
        for workflow_result in workflow_results:
            if workflow_result is None:
                continue
            task_list = [t.name for t in workflow_result.tasks]
            if "evaluate_experiment" in task_list:
                eval_output = workflow_result.tasks["evaluate_experiment"].output
                eval_outputs.update(eval_output)
            else:
                warnings.warn(
                    f"No `evaluate_experiment` task found in the experiment workflow "
                    f"{workflow_result._name}. "
                    f"Setting layer.status to 'passed'.",
                    stacklevel=2,
                )
        return eval_outputs
