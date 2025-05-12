# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
from laboneq.workflow import task, workflow

from laboneq_applications.qpu_types.twpa import TWPA, TWPAParameters
from laboneq_applications.tasks import temporary_modify


@pytest.fixture
def qubit_parameters():
    """Qubit parameters to update."""

    return {
        "q0": {
            "ge_drive_amplitude_pi": 0.55,
            "ge_drive_amplitude_pi2": 0.255,
        },
        "q1": {
            "resonance_frequency_ef": 5.58e9,
            "readout_resonator_frequency": 7e9,
        },
    }


class TestTemporaryModify:
    def test_run_standalone(self, two_tunable_transmon_platform, qubit_parameters):
        [q0, q1] = two_tunable_transmon_platform.qpu.quantum_elements

        [new_q0, new_q1] = temporary_modify([q0, q1], qubit_parameters)
        assert new_q0.parameters.ge_drive_amplitude_pi == 0.55
        assert new_q0.parameters.ge_drive_amplitude_pi2 == 0.255
        assert new_q1.parameters.resonance_frequency_ef == 5.58e9
        assert new_q1.parameters.readout_resonator_frequency == 7e9

        q0_params = deepcopy(q0.parameters)
        q1_params = deepcopy(q1.parameters)
        q0_params.ge_drive_amplitude_pi = 0.55
        q1_params.resonance_frequency_ef = 5.58e9
        qubit_parameters_class = {
            "q0": q0_params,
            "q1": q1_params,
        }
        [new_q0, new_q1] = temporary_modify([q0, q1], qubit_parameters_class)
        assert new_q0.parameters.ge_drive_amplitude_pi == 0.55
        assert new_q1.parameters.resonance_frequency_ef == 5.58e9

        # single qubit
        new_q0 = temporary_modify(q0, qubit_parameters)
        assert new_q0.parameters.ge_drive_amplitude_pi == 0.55
        assert new_q0.parameters.ge_drive_amplitude_pi2 == 0.255

        new_q0 = temporary_modify([q0], qubit_parameters)
        assert len(new_q0) == 1
        assert new_q0[0].parameters.ge_drive_amplitude_pi == 0.55
        assert new_q0[0].parameters.ge_drive_amplitude_pi2 == 0.255

    def test_twpa(self):
        twpa0 = TWPA(
            "twpa0",
            {"measure": "q0/measure", "acquire": "q0/acquire"},
            TWPAParameters(),
        )
        new_twpa0 = temporary_modify(
            twpa0, {"twpa0": {"readout_range_out": -10, "readout_range_in": -10}}
        )

        assert new_twpa0.parameters.readout_range_out == -10
        assert new_twpa0.parameters.readout_range_in == -10

    def test_partial_modify(self, two_tunable_transmon_platform, qubit_parameters):
        [q0, q1] = two_tunable_transmon_platform.qpu.quantum_elements

        partial_parameters = {"q0": qubit_parameters["q0"]}
        [new_q0, new_q1] = temporary_modify([q0, q1], partial_parameters)
        assert new_q0.parameters.ge_drive_amplitude_pi == 0.55
        assert new_q0.parameters.ge_drive_amplitude_pi2 == 0.255

        assert new_q1 == q1

        [new_q0, new_q1] = temporary_modify([q0, q1], None)
        assert new_q0 == q0
        assert new_q1 == q1

    def test_run_in_workflow(self, two_tunable_transmon_platform, qubit_parameters):
        qubits = two_tunable_transmon_platform.qpu.quantum_elements

        @task
        def dumb_task(qubits):
            return qubits

        @workflow
        def test_workflow(qubits, qubit_parameters):
            qubits = temporary_modify(qubits, qubit_parameters)
            dumb_task(qubits)

        res = test_workflow(qubits, qubit_parameters).run()
        assert len(res.tasks) == 2
        qubits = res.tasks["dumb_task"].output
        assert qubits[0].parameters.ge_drive_amplitude_pi == 0.55
        assert qubits[0].parameters.ge_drive_amplitude_pi2 == 0.255
        assert qubits[1].parameters.resonance_frequency_ef == 5.58e9
        assert qubits[1].parameters.readout_resonator_frequency == 7e9
