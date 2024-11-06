# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for laboneq_applications.opeqnqasm3.gate_store"""

from __future__ import annotations

import pytest
from laboneq.openqasm3.openqasm3_importer import exp_from_qasm
from laboneq.simple import Session

from laboneq_applications.openqasm3 import create_gate_store


@pytest.fixture()
def platform(two_tunable_transmon_platform):
    return two_tunable_transmon_platform


@pytest.fixture()
def qop(platform):
    return platform.qpu.quantum_operations


@pytest.fixture()
def qubits(platform):
    return platform.qpu.qubits


@pytest.fixture()
def qubit_map(qubits):
    return {qubit.uid: qubit for qubit in qubits}


class TestCreateGateStore:
    def check_exp_compiles(self, exp, platform):
        session = Session(platform.setup)
        session.connect(do_emulation=True)
        session.compile(exp)

    def test_create_gate_store(self, qop, qubit_map):
        gate_store = create_gate_store(qop, qubit_map)

        assert set(gate_store.gates) == {
            (gate, (qubit,))
            for gate in qop.keys()  # noqa: SIM118
            for qubit in qubit_map
        }

    def test_custom_qubit_map(self, qop, qubits, platform):
        qubit_map = {"my_qubit0": qubits[0], "my_qubit1": qubits[1]}
        gate_store = create_gate_store(qop, qubit_map)

        qasm_text = """
        OPENQASM 3;
        qubit my_qubit0;
        qubit my_qubit1;
        x90 my_qubit0;
        x90 my_qubit1;
        """
        exp = exp_from_qasm(qasm_text, qubits=qubit_map, gate_store=gate_store)

        assert "/logical_signal_groups/q0/drive" in exp.signals
        assert "/logical_signal_groups/q1/drive" in exp.signals

        self.check_exp_compiles(exp, platform)

    def test_custom_gate_map(self, qop, qubit_map, platform):
        gate_map = {"swirl": "x180"}
        gate_store = create_gate_store(qop, qubit_map, gate_map)

        qasm_text = """
        OPENQASM 3;
        qubit q0;
        qubit q1;
        swirl q0;
        swirl q1;
        """

        exp = exp_from_qasm(qasm_text, qubits=qubit_map, gate_store=gate_store)

        root_sect = exp.sections[0].children[0]
        sect_q0 = root_sect.children[0]
        sect_q1 = root_sect.children[1]
        assert "x180" in sect_q0.name
        assert "x180" in sect_q1.name

        self.check_exp_compiles(exp, platform)

    def test_bad_gate_map(self, qop, qubit_map):
        gate_map = {"x90": "unknown_gate"}
        with pytest.raises(ValueError) as err:
            create_gate_store(qop, qubit_map, gate_map)
        assert (
            str(err.value)
            == "Gate 'unknown_gate' is not supported by QuantumOperations."
        )
