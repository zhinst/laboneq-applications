# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import pytest
from laboneq.dsl.quantum import QPU
from laboneq.workflow import task, workflow

from laboneq_applications.qpu_types.twpa import TWPA, TWPAOperations, TWPAParameters
from laboneq_applications.tasks import temporary_qpu


@pytest.fixture
def temp_parameters():
    """Temporary parameters to update."""

    return {
        "q0": {
            "ge_drive_amplitude_pi": 0.55,
            "ge_drive_amplitude_pi2": 0.255,
        },
        "q1": {
            "resonance_frequency_ef": 5.58e9,
            "readout_resonator_frequency": 7e9,
        },
        ("test", "q0", "q1"): {
            "readout_lo_frequency": 1.23,
            "ge_drive_length": 4.56,
        },
    }


class TestTemporaryQPU:
    def test_run_standalone(self, two_tunable_transmon_platform, temp_parameters):
        qpu = two_tunable_transmon_platform.qpu
        q0 = qpu["q0"]
        q1 = qpu["q1"]
        qpu.topology.add_edge(
            "test", "q0", "q1", parameters=q0.parameters.copy(), quantum_element=q0
        )
        edge = qpu.topology["test", "q0", "q1"]

        # check parameter update
        new_qpu = temporary_qpu(qpu, temp_parameters)
        new_q0 = new_qpu["q0"]
        new_q1 = new_qpu["q1"]
        new_edge = new_qpu.topology["test", "q0", "q1"]
        assert q0.parameters.ge_drive_amplitude_pi == 0.8
        assert q0.parameters.ge_drive_amplitude_pi2 == 0.4
        assert q1.parameters.resonance_frequency_ef == 6.31e9
        assert q1.parameters.readout_resonator_frequency == 7.109999999999999e9
        assert edge.parameters.readout_lo_frequency == 7e9
        assert edge.parameters.ge_drive_length == 51e-9
        assert new_q0.parameters.ge_drive_amplitude_pi == 0.55
        assert new_q0.parameters.ge_drive_amplitude_pi2 == 0.255
        assert new_q1.parameters.resonance_frequency_ef == 5.58e9
        assert new_q1.parameters.readout_resonator_frequency == 7e9
        assert new_edge.parameters.readout_lo_frequency == 1.23
        assert new_edge.parameters.ge_drive_length == 4.56

        # check custom qubit parameters class
        q0_params = q0.parameters.copy()
        q1_params = q1.parameters.copy()
        edge_params = edge.parameters.copy()
        q0_params.ge_drive_amplitude_pi = 0.55
        q1_params.resonance_frequency_ef = 5.58e9
        edge_params.readout_lo_frequency = 1.23
        temp_parameters_class = {
            "q0": q0_params,
            "q1": q1_params,
            ("test", "q0", "q1"): edge_params,
        }
        new_qpu = temporary_qpu(qpu, temp_parameters_class)
        new_q0 = new_qpu["q0"]
        new_q1 = new_qpu["q1"]
        new_edge = new_qpu.topology["test", "q0", "q1"]
        assert q0.parameters.ge_drive_amplitude_pi == 0.8
        assert q1.parameters.resonance_frequency_ef == 6.31e9
        assert edge.parameters.readout_lo_frequency == 7e9
        assert new_q0.parameters.ge_drive_amplitude_pi == 0.55
        assert new_q1.parameters.resonance_frequency_ef == 5.58e9
        assert new_edge.parameters.readout_lo_frequency == 1.23

        # check type hints
        with pytest.raises(TypeError) as err:
            temporary_qpu(qpu, 1)
        assert (
            str(err.value)
            == "The temporary parameters have invalid type: <class 'int'>. "
            "Expected type: "
            "dict[str | tuple[str, str, str], dict | QuantumParameters] | None."
        )

    def test_run_standalone_without_edge_quantum_element(
        self, two_tunable_transmon_platform, temp_parameters
    ):
        qpu = two_tunable_transmon_platform.qpu
        q0 = qpu["q0"]
        qpu.topology.add_edge(
            "test",
            "q0",
            "q1",
            parameters=q0.parameters.copy(),
        )
        edge = qpu.topology["test", "q0", "q1"]
        new_qpu = temporary_qpu(qpu, temp_parameters)
        new_edge = new_qpu.topology["test", "q0", "q1"]
        assert edge.parameters.readout_lo_frequency == 7e9
        assert edge.parameters.ge_drive_length == 51e-9
        assert edge.quantum_element is None
        assert new_edge.parameters.readout_lo_frequency == 1.23
        assert new_edge.parameters.ge_drive_length == 4.56
        assert new_edge.quantum_element is None

    def test_twpa(self):
        twpa0 = TWPA(
            "twpa0",
            {"measure": "q0/measure", "acquire": "q0/acquire"},
            TWPAParameters(),
        )
        qpu = QPU(twpa0, TWPAOperations)

        new_qpu = temporary_qpu(
            qpu, {"twpa0": {"readout_range_out": -10, "readout_range_in": -10}}
        )
        new_twpa0 = new_qpu["twpa0"]
        assert twpa0.parameters.readout_range_out == 5
        assert twpa0.parameters.readout_range_in == 10
        assert new_twpa0.parameters.readout_range_out == -10
        assert new_twpa0.parameters.readout_range_in == -10

    def test_partial_modify(self, two_tunable_transmon_platform, temp_parameters):
        qpu = two_tunable_transmon_platform.qpu
        q0 = qpu["q0"]
        q1 = qpu["q1"]
        qpu.topology.add_edge(
            "test", "q0", "q1", parameters=q0.parameters.copy(), quantum_element=q0
        )
        qpu.topology.add_edge(
            "test2", "q0", "q1", parameters=q0.parameters.copy(), quantum_element=q0
        )
        edge = qpu.topology["test", "q0", "q1"]
        edge2 = qpu.topology["test2", "q0", "q1"]

        # check with dict
        partial_parameters = {
            ("test", "q0", "q1"): temp_parameters[("test", "q0", "q1")]
        }
        new_qpu = temporary_qpu(qpu, partial_parameters)
        new_q0 = new_qpu["q0"]
        new_q1 = new_qpu["q1"]
        new_edge = new_qpu.topology["test", "q0", "q1"]
        new_edge2 = new_qpu.topology["test2", "q0", "q1"]
        assert new_q0 == q0
        assert new_q1 == q1
        assert edge.parameters.readout_lo_frequency == 7e9
        assert edge.parameters.ge_drive_length == 51e-9
        assert new_edge.parameters.readout_lo_frequency == 1.23
        assert new_edge.parameters.ge_drive_length == 4.56
        assert edge2.parameters.readout_lo_frequency == 7e9
        assert edge2.parameters.ge_drive_length == 51e-9
        assert new_edge2.parameters.readout_lo_frequency == 7e9
        assert new_edge2.parameters.ge_drive_length == 51e-9

        # check with None
        new_qpu = temporary_qpu(qpu, None)
        assert new_qpu is not qpu
        assert new_qpu["q0"] == q0
        assert new_qpu["q1"] == q1
        assert new_qpu.topology["test", "q0", "q1"].tag == edge.tag
        assert new_qpu.topology["test", "q0", "q1"].source_node == edge.source_node
        assert new_qpu.topology["test", "q0", "q1"].target_node == edge.target_node
        assert new_qpu.topology["test", "q0", "q1"].parameters == edge.parameters
        assert (
            new_qpu.topology["test", "q0", "q1"].quantum_element == edge.quantum_element
        )

        # check with single argument
        new_qpu = temporary_qpu(qpu)
        assert new_qpu["q0"] == qpu["q0"]
        assert new_qpu["q1"] == qpu["q1"]
        assert new_qpu.topology["test", "q0", "q1"].tag == edge.tag
        assert new_qpu.topology["test", "q0", "q1"].source_node == edge.source_node
        assert new_qpu.topology["test", "q0", "q1"].target_node == edge.target_node
        assert new_qpu.topology["test", "q0", "q1"].parameters == edge.parameters
        assert (
            new_qpu.topology["test", "q0", "q1"].quantum_element == edge.quantum_element
        )

    def test_partial_modify_without_edge_quantum_elements(
        self, two_tunable_transmon_platform, temp_parameters
    ):
        qpu = two_tunable_transmon_platform.qpu
        q0 = qpu["q0"]
        qpu.topology.add_edge(
            "test",
            "q0",
            "q1",
            parameters=q0.parameters.copy(),
        )
        qpu.topology.add_edge(
            "test2",
            "q0",
            "q1",
            parameters=q0.parameters.copy(),
        )
        edge = qpu.topology["test", "q0", "q1"]
        edge2 = qpu.topology["test2", "q0", "q1"]
        partial_parameters = {
            ("test", "q0", "q1"): temp_parameters[("test", "q0", "q1")]
        }
        new_qpu = temporary_qpu(qpu, partial_parameters)
        new_edge = new_qpu.topology["test", "q0", "q1"]
        new_edge2 = new_qpu.topology["test2", "q0", "q1"]
        assert edge.parameters.readout_lo_frequency == 7e9
        assert edge.parameters.ge_drive_length == 51e-9
        assert edge.quantum_element is None
        assert new_edge.parameters.readout_lo_frequency == 1.23
        assert new_edge.parameters.ge_drive_length == 4.56
        assert new_edge.quantum_element is None
        assert edge2.parameters.readout_lo_frequency == 7e9
        assert edge2.parameters.ge_drive_length == 51e-9
        assert edge2.quantum_element is None
        assert new_edge2.parameters.readout_lo_frequency == 7e9
        assert new_edge2.parameters.ge_drive_length == 51e-9
        assert new_edge2.quantum_element is None

    def test_run_in_workflow(self, two_tunable_transmon_platform, temp_parameters):
        qpu = two_tunable_transmon_platform.qpu
        qpu.topology.add_edge(
            "test",
            "q0",
            "q1",
            parameters=qpu["q0"].parameters.copy(),
            quantum_element="q0",
        )

        @task
        def dumb_task(_qpu):
            return _qpu

        @workflow
        def test_workflow(qpu, temp_params):
            temp_qpu = temporary_qpu(qpu, temp_params)
            dumb_task(temp_qpu)

        res = test_workflow(qpu, temp_parameters).run()
        assert len(res.tasks) == 2
        qpu = res.tasks["dumb_task"].output
        q0 = qpu["q0"]
        q1 = qpu["q1"]
        edge = qpu.topology["test", "q0", "q1"]
        assert q0.parameters.ge_drive_amplitude_pi == 0.55
        assert q0.parameters.ge_drive_amplitude_pi2 == 0.255
        assert q1.parameters.resonance_frequency_ef == 5.58e9
        assert q1.parameters.readout_resonator_frequency == 7e9
        assert edge.parameters.readout_lo_frequency == 1.23
        assert edge.parameters.ge_drive_length == 4.56
