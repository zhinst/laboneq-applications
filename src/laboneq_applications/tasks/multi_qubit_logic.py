# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines tasks handling multi-qubit logic and manipulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from laboneq.dsl.quantum import QuantumElement
from laboneq.workflow import task

if TYPE_CHECKING:
    from laboneq.dsl.quantum.qpu_topology import TopologyEdge


@task(save=False)
def extract_nodes_from_edges(
    edges: list[TopologyEdge],
    node_type: Literal["all", "source", "target"] = "all",
) -> list[QuantumElement] | tuple[list[QuantumElement], list[QuantumElement]]:
    """Extract the nodes of a list of edges.

    Args:
        edges: A list of TopologyEdge instances from which to extract the nodes.
        node_type (str): The type of nodes to extract:
            "all": both the source and the target nodes are returned;
            "source": only the source nodes are returned;
            "target": only the target nodes are returned.

    Returns:
        The list of extracted nodes. If "all" is selected as a `node_type`, we return a
        tuple of lists: the first element is the list of source nodes, and the second
        element is the list of target nodes.

    Raises:
        TypeError: If `edges` has an invalid type.
        ValueError: If the `node_type` is not "all", "source", or "target".
    """
    if not isinstance(edges, list):
        raise TypeError(
            f"The `edges` argument has invalid type: {type(edges)}. "
            f"Expected type: list[TopologyEdge]."
        )

    source_qubits = [e.source_node for e in edges]
    target_qubits = [e.target_node for e in edges]

    if node_type == "all":
        return source_qubits, target_qubits
    if node_type == "source":
        return source_qubits
    if node_type == "target":
        return target_qubits

    raise ValueError(
        f"The `node_type` value is not supported: '{node_type}'. "
        f"Expected values: 'all', 'source', 'target'."
    )
