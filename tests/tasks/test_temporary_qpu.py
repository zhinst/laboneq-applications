# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
from laboneq.dsl.quantum import QPU
from laboneq.workflow import task, workflow

from laboneq_applications.qpu_types.twpa import TWPA, TWPAOperations, TWPAParameters
from laboneq_applications.tasks import temporary_qpu


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


class TestTemporaryQPU:
    def test_run_standalone(self, two_tunable_transmon_platform, qubit_parameters):
        qpu = two_tunable_transmon_platform.qpu
        q0 = qpu.quantum_element_by_uid("q0")
        q1 = qpu.quantum_element_by_uid("q1")

        # check parameter update
        new_qpu = temporary_qpu(qpu, qubit_parameters)
        new_q0 = new_qpu.quantum_element_by_uid("q0")
        new_q1 = new_qpu.quantum_element_by_uid("q1")
        assert q0.parameters.ge_drive_amplitude_pi == 0.8
        assert q0.parameters.ge_drive_amplitude_pi2 == 0.4
        assert q1.parameters.resonance_frequency_ef == 6.31e9
        assert q1.parameters.readout_resonator_frequency == 7.109999999999999e9
        assert new_q0.parameters.ge_drive_amplitude_pi == 0.55
        assert new_q0.parameters.ge_drive_amplitude_pi2 == 0.255
        assert new_q1.parameters.resonance_frequency_ef == 5.58e9
        assert new_q1.parameters.readout_resonator_frequency == 7e9

        # check custom qubit parameters class
        q0_params = deepcopy(q0.parameters)
        q1_params = deepcopy(q1.parameters)
        q0_params.ge_drive_amplitude_pi = 0.55
        q1_params.resonance_frequency_ef = 5.58e9
        qubit_parameters_class = {
            "q0": q0_params,
            "q1": q1_params,
        }
        new_qpu = temporary_qpu(qpu, qubit_parameters_class)
        new_q0 = new_qpu.quantum_element_by_uid("q0")
        new_q1 = new_qpu.quantum_element_by_uid("q1")
        assert q0.parameters.ge_drive_amplitude_pi == 0.8
        assert q1.parameters.resonance_frequency_ef == 6.31e9
        assert new_q0.parameters.ge_drive_amplitude_pi == 0.55
        assert new_q1.parameters.resonance_frequency_ef == 5.58e9

        # check type hints
        with pytest.raises(TypeError) as err:
            new_qpu = temporary_qpu(qpu, 1)
        assert (
            str(err.value)
            == "The temporary parameters have invalid type: <class 'int'>. "
            "Expected type: dict[str, dict | QuantumParameters] | None."
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
        new_twpa0 = new_qpu.quantum_element_by_uid("twpa0")
        assert twpa0.parameters.readout_range_out == 5
        assert twpa0.parameters.readout_range_in == 10
        assert new_twpa0.parameters.readout_range_out == -10
        assert new_twpa0.parameters.readout_range_in == -10

    def test_partial_modify(self, two_tunable_transmon_platform, qubit_parameters):
        qpu = two_tunable_transmon_platform.qpu
        q0 = qpu.quantum_element_by_uid("q0")
        q1 = qpu.quantum_element_by_uid("q1")

        # check with dict
        partial_parameters = {"q0": qubit_parameters["q0"]}
        new_qpu = temporary_qpu(qpu, partial_parameters)
        new_q0 = new_qpu.quantum_element_by_uid("q0")
        new_q1 = new_qpu.quantum_element_by_uid("q1")
        assert q0.parameters.ge_drive_amplitude_pi == 0.8
        assert q0.parameters.ge_drive_amplitude_pi2 == 0.4
        assert new_q0.parameters.ge_drive_amplitude_pi == 0.55
        assert new_q0.parameters.ge_drive_amplitude_pi2 == 0.255
        assert new_q1 == q1

        # check with None
        new_qpu = temporary_qpu(qpu, None)
        assert new_qpu is not qpu
        assert new_qpu.quantum_element_by_uid("q0") == q0
        assert new_qpu.quantum_element_by_uid("q1") == q1

        # check with single argument
        new_qpu = temporary_qpu(qpu)
        assert new_qpu.quantum_element_by_uid("q0") == qpu.quantum_element_by_uid("q0")
        assert new_qpu.quantum_element_by_uid("q1") == qpu.quantum_element_by_uid("q1")

    def test_run_in_workflow(self, two_tunable_transmon_platform, qubit_parameters):
        qpu = two_tunable_transmon_platform.qpu

        @task
        def dumb_task(_qpu):
            return _qpu

        @workflow
        def test_workflow(qpu, qubit_parameters):
            temp_qpu = temporary_qpu(qpu, qubit_parameters)
            dumb_task(temp_qpu)

        res = test_workflow(qpu, qubit_parameters).run()
        assert len(res.tasks) == 2
        qpu = res.tasks["dumb_task"].output
        q0 = qpu.quantum_element_by_uid("q0")
        q1 = qpu.quantum_element_by_uid("q1")
        assert q0.parameters.ge_drive_amplitude_pi == 0.55
        assert q0.parameters.ge_drive_amplitude_pi2 == 0.255
        assert q1.parameters.resonance_frequency_ef == 5.58e9
        assert q1.parameters.readout_resonator_frequency == 7e9
