# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""LabOne Q workflow node subclass."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from laboneq._automation import AutomationNode
from laboneq._automation.element import AutomationElementStatus as Status
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.quantum import QuantumParameters
from laboneq.workflow import WorkflowBuilder, WorkflowOptions, WorkflowResult
from laboneq.workflow.timestamps import local_timestamp

from laboneq_applications._automation.workflow.workflow_logic import WorkflowLogic

if TYPE_CHECKING:
    from laboneq_applications._automation.workflow.workflow_automation import (
        WorkflowAutomation,
    )


@classformatter
class WorkflowNode(AutomationNode):
    """A LabOne Q workflow node in the automation framework.

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
        layer: The key of the parent layer.
        workflow_builder: The workflow builder instance.
        quantum_elements: The quantum element UID. For multiple-qubit
            experiments, this may be a list.
        workflow_parameters: The workflow parameters.
        general_workflow_parameters: The general workflow parameters.
        temporary_qpu_parameters: The temporary QPU parameters.
        workflow_options: The workflow options.
        logic: The node decision logic.
        workflow_result: The workflow result.
    """

    def __init__(
        self,
        # compulsory workflow parameters
        workflow_builder: WorkflowBuilder | None,
        quantum_elements: str | list[str] | None,
        *,
        # optional workflow parameters
        workflow_parameters: dict[str, Any] | None = None,
        general_workflow_parameters: dict[str, Any] | None = None,
        temporary_qpu_parameters: dict[
            str | tuple[str, str, str], dict | QuantumParameters
        ]
        | None = None,
        workflow_options: WorkflowOptions | None = None,
        logic: WorkflowLogic | None = None,
        # automation node parameters
        **kwargs,
    ) -> None:
        """Initialize LabOne Q workflow node attributes.

        Arguments:
            workflow_builder: The workflow builder instance.
            quantum_elements: The quantum element UID. For multiple-qubit
                experiments, this may be a list.
            workflow_parameters: The workflow parameters.
            general_workflow_parameters: The general workflow parameters.
            temporary_qpu_parameters: The temporary QPU parameters.
            workflow_options: The workflow options.
            logic: The node decision logic.
            **kwargs: Automation layer parameters.

        This constructor also accepts the arguments of
        [`AutomationElement`][laboneq._automation.framework.element.AutomationElement]
        and [`AutomationNode`][laboneq._automation.framework.layer.AutomationNode].
        The arguments `key`, `depends_on`, and `layer` are compulsory.
        """
        # compulsory workflow parameters
        self.workflow_builder = workflow_builder
        self.quantum_elements = quantum_elements
        # optional workflow parameters
        self.workflow_parameters = workflow_parameters
        self.general_workflow_parameters = general_workflow_parameters
        self.temporary_qpu_parameters = temporary_qpu_parameters
        self.workflow_options = workflow_options
        self.logic = logic
        # automation node parameters
        super().__init__(**kwargs)

        # workflow results
        self.workflow_result: WorkflowResult | None = None

    def __repr__(self) -> str:
        if isinstance(self.workflow_builder, WorkflowBuilder):
            workflow_builder_name = self.workflow_builder._name
        else:
            workflow_builder_name = None

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
            f"workflow={workflow_builder_name} "
            f"quantum_elements={self.quantum_elements} "
            f"workflow_parameters={self.workflow_parameters} "
            f"general_workflow_parameters={self.general_workflow_parameters} "
            f"temporary_qpu_parameters={self.temporary_qpu_parameters} "
            f"workflow_options={self.workflow_options} "
            f"logic={self.logic} "
            f"workflow_result={self.workflow_result}"
            f">"
        )

    def __rich_repr__(self):
        if isinstance(self.workflow_builder, WorkflowBuilder):
            workflow_builder_name = self.workflow_builder._name
        else:
            workflow_builder_name = None

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
        yield "workflow", workflow_builder_name
        yield "quantum_elements", self.quantum_elements
        yield "workflow_parameters", self.workflow_parameters
        yield "general_workflow_parameters", self.general_workflow_parameters
        yield "temporary_qpu_parameters", self.temporary_qpu_parameters
        yield "workflow_options", self.workflow_options
        yield "logic", self.logic
        yield "workflow_result", self.workflow_result

    def _set_workflow_parameters(
        self, workflow_parameters: dict[str, Any] | None
    ) -> None:
        """Set the workflow parameters of the node.

        !!! note
            This is a helper function for `add_automation_parameters`.

        Arguments:
            workflow_parameters: The workflow parameters.
        """
        if isinstance(self.quantum_elements, str):
            q = self.quantum_elements
        else:
            raise NotImplementedError("Multi-qubit experiment workflows.")
        if workflow_parameters is not None:
            if self.workflow_parameters is not None:
                workflow_parameters[q].update(self.workflow_parameters)
            self.workflow_parameters = workflow_parameters[q]

    def _set_general_workflow_parameters(
        self, general_workflow_parameters: dict[str, Any] | None
    ) -> None:
        """Set the general workflow parameters of the node.

        !!! note
            This is a helper function for `add_automation_parameters`.

        Arguments:
            general_workflow_parameters: The general workflow parameters.
        """
        if general_workflow_parameters is not None:
            if self.general_workflow_parameters is not None:
                general_workflow_parameters.update(self.general_workflow_parameters)
            self.general_workflow_parameters = general_workflow_parameters

    def _set_temporary_qpu_parameters(
        self,
        temporary_qpu_parameters: dict[
            str | tuple[str, str, str], dict | QuantumParameters
        ]
        | None,
    ) -> None:
        """Set the temporary qpu parameters of the node.

        !!! note
            This is a helper function for `add_automation_parameters`.

        Arguments:
            temporary_qpu_parameters: The temporary QPU parameters.
        """
        if temporary_qpu_parameters is not None:
            if self.temporary_qpu_parameters is not None:
                temporary_qpu_parameters.update(self.temporary_qpu_parameters)
            self.temporary_qpu_parameters = temporary_qpu_parameters

    def _set_workflow_options(self, workflow_options: WorkflowOptions | None) -> None:
        """Set the workflow options of the node.

        !!! note
            This is a helper function for `add_automation_parameters`.

        Arguments:
            workflow_options: The workflow options.
        """
        if self.workflow_options is None:
            options = self.workflow_builder.options()
            if workflow_options is not None:
                for key, value in workflow_options.items():
                    set_option_method = getattr(options, key)
                    set_option_method(value)
            self.workflow_options = options

    def add_automation_parameters(self, auto: WorkflowAutomation) -> None:
        """Add automation parameters.

        Arguments:
            auto: The workflow automation instance.

        !!! note
            Parameters defined directly on the workflow node take priority.
        """
        # Skip for empty workflow nodes
        if self.workflow_builder is None:
            return

        # Extract automation parameters
        (
            workflow_parameters,
            general_workflow_parameters,
            temporary_qpu_parameters,
            workflow_options,
            logic,
        ) = auto.extract_automation_parameters(self.layer)

        # Define the node parameters
        if self.qpu is None:
            self.qpu = auto.qpu
        self._set_workflow_parameters(workflow_parameters)
        self._set_general_workflow_parameters(general_workflow_parameters)
        self._set_temporary_qpu_parameters(temporary_qpu_parameters)
        self._set_workflow_options(workflow_options)
        if logic is not None:
            self.logic = logic

    def run_executable(self, auto: WorkflowAutomation) -> WorkflowResult:
        """Runs a LabOne Q experiment workflow.

        Arguments:
            auto: The workflow automation instance.

        Returns:
            The workflow result.

        !!! note
            The session, QPU, and calibration parameters may be read from the
            automation framework.
        """
        # Check QPU
        if self.qpu is None:
            raise ValueError("No QPU specified.")

        # Check workflow builder
        if self.workflow_builder is None:
            raise ValueError("No workflow builder specified.")

        # Check quantum elements
        if self.quantum_elements is None:
            raise ValueError("No quantum elements specified.")
        if not isinstance(self.quantum_elements, str):
            raise NotImplementedError("Multi-qubit experiment workflows.")

        # Check workflow parameters
        if self.workflow_parameters is None:
            workflow_parameters = {}
        else:
            workflow_parameters = self.workflow_parameters

        # Check general workflow parameters
        if self.general_workflow_parameters is None:
            general_workflow_parameters = {}
        else:
            general_workflow_parameters = self.general_workflow_parameters

        # Build experiment workflow
        workflow = self.workflow_builder(
            auto.session,
            self.qpu,
            self.quantum_elements,
            temporary_parameters=self.temporary_qpu_parameters,
            options=self.workflow_options,
            **workflow_parameters,
            **general_workflow_parameters,
        )

        # Run experiment workflow
        workflow_result = workflow.run()

        self.workflow_result = workflow_result
        self.timestamp = local_timestamp()

        task_list = [t.name for t in workflow_result.tasks]
        if "evaluate_experiment" in task_list:
            eval_output = workflow_result.tasks["evaluate_experiment"].output
            eval_success = eval_output[self.quantum_elements]["success"]
            self.status = Status.PASSED if eval_success else Status.FAILED
            self.fail_count += 1 if not eval_output else 0
            self.success_count += 1 if eval_output else 0
        else:
            self.status = Status.PASSED
            self.success_count += 1

        return workflow_result
