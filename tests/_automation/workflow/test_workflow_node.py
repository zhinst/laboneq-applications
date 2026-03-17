# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import pytest

from laboneq_applications._automation import WorkflowNode


@pytest.fixture
def node() -> WorkflowNode:
    return WorkflowNode(key="node1", depends_on=set(), layer_key="layer1")


class TestWorkflowNode:
    def test_quantum_element(self, node):
        assert node.quantum_element is node.key
        node.quantum_element = "new_node1"
        assert node.quantum_element == "new_node1"
