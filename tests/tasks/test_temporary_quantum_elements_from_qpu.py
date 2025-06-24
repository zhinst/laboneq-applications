# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
from laboneq.dsl.quantum import QPU
from laboneq.workflow import task, workflow

from laboneq_applications.qpu_types.twpa import TWPA, TWPAOperations, TWPAParameters
from laboneq_applications.tasks import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
)


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


class TestTemporaryQuantumElementsFromQPU:
    def test_run_standalone(self, two_tunable_transmon_platform, qubit_parameters):
        qpu = two_tunable_transmon_platform.qpu
        qubits = qpu.quantum_elements

        # check parameter update
        new_qpu = temporary_qpu(qpu, qubit_parameters)
        new_qubits = temporary_quantum_elements_from_qpu(new_qpu, qubits)
        assert qubits[0].parameters.ge_drive_amplitude_pi == 0.8
        assert qubits[0].parameters.ge_drive_amplitude_pi2 == 0.4
        assert qubits[1].parameters.resonance_frequency_ef == 6.31e9
        assert qubits[1].parameters.readout_resonator_frequency == 7.109999999999999e9
        assert new_qubits[0].parameters.ge_drive_amplitude_pi == 0.55
        assert new_qubits[0].parameters.ge_drive_amplitude_pi2 == 0.255
        assert new_qubits[1].parameters.resonance_frequency_ef == 5.58e9
        assert new_qubits[1].parameters.readout_resonator_frequency == 7e9

        # check custom qubit parameters class
        q0_params = deepcopy(qubits[0].parameters)
        q1_params = deepcopy(qubits[1].parameters)
        q0_params.ge_drive_amplitude_pi = 0.55
        q1_params.resonance_frequency_ef = 5.58e9
        qubit_parameters_class = {
            "q0": q0_params,
            "q1": q1_params,
        }
        new_qpu = temporary_qpu(qpu, qubit_parameters_class)
        new_qubits = temporary_quantum_elements_from_qpu(new_qpu, qubits)
        assert qubits[0].parameters.ge_drive_amplitude_pi == 0.8
        assert qubits[1].parameters.resonance_frequency_ef == 6.31e9
        assert new_qubits[0].parameters.ge_drive_amplitude_pi == 0.55
        assert new_qubits[1].parameters.resonance_frequency_ef == 5.58e9

        # check single qubit update (QuantumElement)
        new_qpu = temporary_qpu(qpu, qubit_parameters)
        new_q0 = temporary_quantum_elements_from_qpu(new_qpu, qubits[0])
        assert qubits[0].parameters.ge_drive_amplitude_pi == 0.8
        assert qubits[0].parameters.ge_drive_amplitude_pi2 == 0.4
        assert new_q0.parameters.ge_drive_amplitude_pi == 0.55
        assert new_q0.parameters.ge_drive_amplitude_pi2 == 0.255

        # check single qubit update (str)
        new_qpu = temporary_qpu(qpu, qubit_parameters)
        new_q0 = temporary_quantum_elements_from_qpu(new_qpu, "q0")
        assert qubits[0].parameters.ge_drive_amplitude_pi == 0.8
        assert qubits[0].parameters.ge_drive_amplitude_pi2 == 0.4
        assert new_q0.parameters.ge_drive_amplitude_pi == 0.55
        assert new_q0.parameters.ge_drive_amplitude_pi2 == 0.255

        # check single qubit update (list[str])
        new_qpu = temporary_qpu(qpu, qubit_parameters)
        new_q0 = temporary_quantum_elements_from_qpu(new_qpu, ["q0"])
        assert len(new_q0) == 1
        assert qubits[0].parameters.ge_drive_amplitude_pi == 0.8
        assert qubits[0].parameters.ge_drive_amplitude_pi2 == 0.4
        assert new_q0[0].parameters.ge_drive_amplitude_pi == 0.55
        assert new_q0[0].parameters.ge_drive_amplitude_pi2 == 0.255

        # check type errors
        with pytest.raises(TypeError) as err:
            new_qubits = temporary_quantum_elements_from_qpu(new_qpu, {"q0": qubits[0]})
        assert (
            str(err.value) == "The quantum elements have invalid type: <class 'dict'>. "
            "Expected type: QuantumElements | list[str] | str | None."
        )
        with pytest.raises(TypeError) as err:
            new_qubits = temporary_quantum_elements_from_qpu(new_qpu, [1, 2, 3])
        assert (
            str(err.value)
            == "The quantum elements list items have invalid type: <class 'int'>. "
            "Expected type: QuantumElement | str."
        )

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
        new_twpa0 = temporary_quantum_elements_from_qpu(new_qpu, "twpa0")
        assert twpa0.parameters.readout_range_out == 5
        assert twpa0.parameters.readout_range_in == 10
        assert new_twpa0.parameters.readout_range_out == -10
        assert new_twpa0.parameters.readout_range_in == -10

    def test_partial_modify(self, two_tunable_transmon_platform, qubit_parameters):
        qpu = two_tunable_transmon_platform.qpu
        qubits = two_tunable_transmon_platform.qpu.quantum_elements

        # check with dict
        partial_parameters = {"q0": qubit_parameters["q0"]}
        new_qpu = temporary_qpu(qpu, partial_parameters)
        new_qubits = temporary_quantum_elements_from_qpu(new_qpu, qubits)
        assert qubits[0].parameters.ge_drive_amplitude_pi == 0.8
        assert qubits[0].parameters.ge_drive_amplitude_pi2 == 0.4
        assert new_qubits[0].parameters.ge_drive_amplitude_pi == 0.55
        assert new_qubits[0].parameters.ge_drive_amplitude_pi2 == 0.255
        assert new_qubits[1] == qubits[1]

        # check with None
        new_qpu = temporary_qpu(qpu, None)
        new_qubits = temporary_quantum_elements_from_qpu(new_qpu, qubits)
        assert new_qubits[0] == qubits[0]
        assert new_qubits[1] == qubits[1]

        # check with single argument
        new_qubits = temporary_quantum_elements_from_qpu(new_qpu)
        assert new_qubits[0] == qubits[0]
        assert new_qubits[1] == qubits[1]

    def test_run_in_workflow(self, two_tunable_transmon_platform, qubit_parameters):
        qpu = two_tunable_transmon_platform.qpu
        qubits = two_tunable_transmon_platform.qpu.quantum_elements

        @task
        def dumb_task(qubits):
            return qubits

        @workflow
        def test_workflow(qpu, qubits, qubit_parameters):
            new_qpu = temporary_qpu(qpu, qubit_parameters)
            qubits = temporary_quantum_elements_from_qpu(new_qpu, qubits)
            dumb_task(qubits)

        res = test_workflow(qpu, qubits, qubit_parameters).run()
        assert len(res.tasks) == 3
        qubits = res.tasks["dumb_task"].output
        assert qubits[0].parameters.ge_drive_amplitude_pi == 0.55
        assert qubits[0].parameters.ge_drive_amplitude_pi2 == 0.255
        assert qubits[1].parameters.resonance_frequency_ef == 5.58e9
        assert qubits[1].parameters.readout_resonator_frequency == 7e9
