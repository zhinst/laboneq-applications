# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""The workflow automation node."""

import attrs
from laboneq.automation import AutomationNode
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter


@classformatter
@attrs.define
class WorkflowNode(AutomationNode):
    """A workflow node in the automation framework."""

    @property
    def quantum_element(self) -> str:
        """The quantum element."""
        return self.key

    @quantum_element.setter
    def quantum_element(self, value: str) -> None:
        """The quantum element setter."""
        self.key = value
