import pytest
from laboneq.simple import Session

from laboneq_applications.qpu_types import QPU, QuantumPlatform
from laboneq_applications.qpu_types.tunable_transmon.demo_qpus import (
    tunable_transmon_qubits,
    tunable_transmon_setup,
)
from laboneq_applications.qpu_types.tunable_transmon.operations import (
    TunableTransmonOperations,
)


class TestQuantumPlatofrm:
    def test_create(self):
        setup = tunable_transmon_setup(2)
        qubits = tunable_transmon_qubits(2, setup)
        qop = TunableTransmonOperations()
        qpu = QPU(qubits, qop)
        platform = QuantumPlatform(setup, qpu)

        assert platform.setup == setup
        assert platform.qpu == qpu

        session = platform.session(do_emulation=True)
        assert isinstance(session, Session)
        assert session.connection_state.emulated


class TestQPU:
    def test_create(self):
        setup = tunable_transmon_setup(2)
        qubits = tunable_transmon_qubits(2, setup)
        qop = TunableTransmonOperations()
        qpu = QPU(qubits, qop)

        assert qpu.qubits == qubits
        assert qpu._qubit_map == {q.uid: q for q in qubits}
        assert qpu.quantum_operations == qop

        copied_qubits = qpu.copy_qubits()
        assert copied_qubits == qubits
        assert id(copied_qubits) != id(qubits)

    def test_update_qubits(self):
        setup = tunable_transmon_setup(2)
        qubits = tunable_transmon_qubits(2, setup)
        qop = TunableTransmonOperations()
        qpu = QPU(qubits, qop)

        assert qpu.qubits[0].parameters.ge_drive_amplitude_pi != 1
        assert qpu.qubits[0].parameters.resonance_frequency_ef != 5.8e9
        assert qpu.qubits[1].parameters.ef_drive_amplitude_pi != 1
        assert qpu.qubits[1].parameters.ge_drive_pulse != {
            "function": "const",
        }

        qubit_parameters = {
            "q0": {
                "ge_drive_amplitude_pi": 1,
                "resonance_frequency_ef": 5.8e9,
            },
            "q1": {
                "ef_drive_amplitude_pi": 1,
                "ge_drive_pulse": {
                    "function": "const",
                },
            },
        }

        qpu.update_qubits(qubit_parameters)
        assert qpu.qubits[0].parameters.ge_drive_amplitude_pi == 1
        assert qpu.qubits[0].parameters.resonance_frequency_ef == 5.8e9
        assert qpu.qubits[1].parameters.ef_drive_amplitude_pi == 1
        assert qpu.qubits[1].parameters.ge_drive_pulse == {
            "function": "const",
        }

    def test_update_qubits_fail(self):
        setup = tunable_transmon_setup(2)
        qubits = tunable_transmon_qubits(2, setup)
        qop = TunableTransmonOperations()
        qpu = QPU(qubits, qop)

        # Test raises error qubit not found
        qubit_parameters = {
            "q10": {
                "ge_drive_amplitude_pi": 1,
            },
        }

        with pytest.raises(ValueError) as err:
            qpu.update_qubits(qubit_parameters)
        assert str(err.value) == "Qubit q10 was not found in the QPU."

        # Test raises error qubit parameter not found
        assert qpu.qubits[0].parameters.resonance_frequency_ge == 6.5e9
        assert qpu.qubits[1].parameters.resonance_frequency_ge == 6.51e9
        qubit_parameters = {
            "q0": {
                "resonance_frequency_ge": 6.0e9,
            },
            "q1": {
                "resonance_frequency_ge": 6.1e9,
                "non-existing": 1,
            },
        }
        with pytest.raises(ValueError) as err:
            qpu.update_qubits(qubit_parameters)
        assert str(err.value) == (
            "Update parameters do not match the qubit parameters: ['non-existing']."
        )
        assert qpu.qubits[0].parameters.resonance_frequency_ge == 6.5e9
        assert qpu.qubits[1].parameters.resonance_frequency_ge == 6.51e9

    def test_measure_section_length(self):
        setup = tunable_transmon_setup(4)
        qubits = tunable_transmon_qubits(4, setup)
        for i, q in enumerate(qubits):
            q.parameters.readout_integration_length = 2e-6 - i * 0.1
        assert QPU.measure_section_length(qubits) == 2e-6
