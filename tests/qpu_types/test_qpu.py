from laboneq.simple import Session

from laboneq_applications.qpu_types import QPU
from laboneq_applications.qpu_types.tunable_transmon.demo_qpus import (
    tunable_transmon_qubits,
    tunable_transmon_setup,
)
from laboneq_applications.qpu_types.tunable_transmon.operations import (
    TunableTransmonOperations,
)


class TestQPU:
    def test_create_qpu(self):
        setup = tunable_transmon_setup(2)
        qubits = tunable_transmon_qubits(2, setup)
        qop = TunableTransmonOperations()
        qpu = QPU(setup, qubits, qop)
        assert qpu.setup == setup
        assert qpu.qubits == qubits
        assert qpu.qop == qop

        session = qpu.session(do_emulation=True)
        assert isinstance(session, Session)
        assert session.connection_state.emulated

        copied_qubits = qpu.copy_qubits()
        assert copied_qubits == qubits
        assert id(copied_qubits) != id(qubits)
