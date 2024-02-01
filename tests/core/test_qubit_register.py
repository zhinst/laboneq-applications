import pytest
from laboneq_library.core.qubit_register import QubitRegister
from laboneq_library.qpu_types.tunable_transmon import TunableTransmonQubit
from laboneq.dsl.device import create_connection, DeviceSetup
from laboneq.dsl.device.instruments import HDAWG


class TestQubitRegister:
    @pytest.fixture
    def qubits(self) -> None:
        return [
            TunableTransmonQubit("q0"),
            TunableTransmonQubit("q1")
        ]

    @pytest.fixture
    def register(self, qubits):
        return QubitRegister(qubits)

    def test_sequence(self, register, qubits):
        assert register == qubits

    def test_str(self, register):
        assert str(register) == "['TunableTransmonQubit(q0)', 'TunableTransmonQubit(q1)']"

    def test_save_load(self, register, tmp_path):
        p = tmp_path / "test.json"
        register.save(p)
        reg = QubitRegister.load(p)
        assert reg == register

    def test_link_signals(self, register):
        setup = DeviceSetup("test")
        setup.add_dataserver(host="localhost", port="8004")
        setup.add_instruments(HDAWG(uid="device_hdawg", address="dev123"))
        setup.add_connections(
            "device_hdawg",
            create_connection(to_signal="q0/flux", ports=f"SIGOUTS/0"),
        )
        register.link_signals(setup)
        assert register[0].signals == {
            "flux": setup.logical_signal_groups[register[0].uid].logical_signals["flux"].path
        }
        setup.add_connections(
            "device_hdawg",
            create_connection(to_signal="q1/flux", ports=f"SIGOUTS/1"),
        )
        register.link_signals(setup)
        assert register[1].signals == {
            "flux": setup.logical_signal_groups[register[1].uid].logical_signals["flux"].path
        }
