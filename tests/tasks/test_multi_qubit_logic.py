# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import re

import pytest
from laboneq.dsl.quantum.qpu_topology import QPUTopology
from laboneq.dsl.quantum.quantum_element import QuantumParameters

from laboneq_applications.tasks.multi_qubit_logic import extract_nodes_from_edges


@pytest.fixture
def two_tunable_transmon_platform_with_topology(two_tunable_transmon_platform):
    platform = two_tunable_transmon_platform
    qpu = platform.qpu
    qubits = qpu.quantum_elements

    topo = QPUTopology(qubits)
    topo.add_edge("coupler_1", "q0", "q1", quantum_element=qubits[0])
    topo.add_edge("coupler_2", "q0", "q1", quantum_element=qubits[1])
    topo.add_edge("coupler_3", "q0", "q1", quantum_element=qubits[0])
    topo.add_edge("coupler_1", "q1", "q0", quantum_element=qubits[1])
    topo.add_edge("coupler_2", "q1", "q0", parameters=QuantumParameters())
    topo.add_edge("coupler_3", "q1", "q0")
    qpu.topology = topo

    return platform


class TestMultiQubitLogic:
    def test_extract_nodes_from_edges(
        self, two_tunable_transmon_platform_with_topology
    ):
        topo = two_tunable_transmon_platform_with_topology.qpu.topology

        assert extract_nodes_from_edges(list(topo.edges())) == (
            [e.source_node for e in topo.edges()],
            [e.target_node for e in topo.edges()],
        )
        assert extract_nodes_from_edges(list(topo.edges()), node_type="source") == [
            e.source_node for e in topo.edges()
        ]
        assert extract_nodes_from_edges(list(topo.edges()), node_type="target") == [
            e.target_node for e in topo.edges()
        ]

        with pytest.raises(
            TypeError,
            match=re.escape(
                "The `edges` argument has "
                "invalid type: "
                "<class 'generator'>. Expected "
                "type: list[TopologyEdge]."
            ),
        ):
            extract_nodes_from_edges(topo.edges())

        with pytest.raises(
            ValueError,
            match="The `node_type` value is not supported: "
            "'destination'. Expected values: 'all', "
            "'source', 'target'.",
        ):
            extract_nodes_from_edges(list(topo.edges()), node_type="destination")
