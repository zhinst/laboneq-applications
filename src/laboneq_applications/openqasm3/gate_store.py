# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines functions to leverage the LabOne Q GateStore."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from laboneq.openqasm3.gate_store import GateStore

if TYPE_CHECKING:
    from laboneq.dsl.experiment.builtins_dsl import QuantumOperations
    from laboneq.dsl.quantum.quantum_operations import Operation
    from laboneq.simple import Section

    from laboneq_applications.qpu_types.tunable_transmon.qubit_types import (
        TunableTransmonQubit,
    )


def _gate_from_operation(
    qubit: TunableTransmonQubit, operation: Operation
) -> Callable[..., Section]:
    """Return a GateStore gate for a quantum operation."""

    def gate(*args) -> Section:
        return operation(qubit, *args)

    gate.__name__ = operation.op.__name__
    gate.__doc__ = operation.op.__doc__

    return gate


def create_gate_store(
    quantum_operations: QuantumOperations,
    qubit_map: dict[str, TunableTransmonQubit],
    gate_map: dict[str, str] | None = None,
) -> GateStore:
    """Creates a GateStore to convert QASM gates to L1Q qops.

    Arguments:
        quantum_operations:
            The QuantumOperations to support in the GateStore.
        qubit_map:
            A dictionary that translates QASM qubits to L1Q qubits.
        gate_map:
            A dictionary that translates the native gates from QASM to L1Q, e.g.
            {"id":None, "sx":"x90", "x":"x180", "rz":"rz"}
            If not provided, the gate_map will assumed as an identity map of qop.
    """
    gate_store = GateStore()
    if gate_map is None:
        gate_map = {gate: gate for gate in quantum_operations.keys()}  # noqa: SIM118
    for oq3_qubit, l1q_qubit in qubit_map.items():
        for qasm_gate, l1q_gate in gate_map.items():
            try:
                operation = quantum_operations[l1q_gate]
            except KeyError as e:
                msg = f"Gate '{l1q_gate}' is not supported by QuantumOperations."
                raise ValueError(msg) from e

            gate_store.register_gate_section(
                qasm_gate,
                [oq3_qubit],
                _gate_from_operation(l1q_qubit, operation),
            )

    return gate_store
