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
        assert qpu.qop == qop

        copied_qubits = qpu.copy_qubits()
        assert copied_qubits == qubits
        assert id(copied_qubits) != id(qubits)
