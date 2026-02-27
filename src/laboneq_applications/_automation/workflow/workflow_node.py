# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import attrs
from laboneq._automation import AutomationNode
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter


@classformatter
@attrs.define
class WorkflowNode(AutomationNode):
    """A workflow node in the automation framework."""

    @property
    def quantum_element(self) -> str:
        return self.key

    @quantum_element.setter
    def quantum_element(self, value: str) -> None:
        self.key = value
