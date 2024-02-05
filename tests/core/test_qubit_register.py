import pytest
from laboneq.dsl.device import DeviceSetup, create_connection
from laboneq.dsl.device.instruments import HDAWG

from laboneq_library.core.qubit_register import QubitRegister
from laboneq_library.qpu_types.tunable_transmon import TunableTransmonQubit


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

    def test_sequence_like(self, register, qubits):
        assert register == qubits
        assert register[1:] == qubits[1:]
        assert [q for q in register] == qubits

    def test_str(self, register):
        assert str(register) == "['TunableTransmonQubit(q0)', 'TunableTransmonQubit(q1)']"

    def test_save_load(self, register, tmp_path):
        p = tmp_path / "test.json"
        register.save(p)
        assert QubitRegister.load(p) == register

        register = QubitRegister([])
        register.save(p)
        assert QubitRegister.load(p) == register

    def test_link_signals(self, register):
        setup = DeviceSetup("test")
        setup.add_dataserver(host="localhost", port="8004")
        setup.add_instruments(HDAWG(uid="device_hdawg", address="dev123"))
        setup.add_connections(
            "device_hdawg",
            create_connection(to_signal="q0/flux", ports="SIGOUTS/0"),
            create_connection(to_signal="q1/drive", ports="SIGOUTS/0"),
        )
        register.link_signals(setup)
        assert register[0].signals == {
            "flux": setup.logical_signal_groups[register[0].uid].logical_signals["flux"].path
        }
        assert register[1].signals == {
            "drive": setup.logical_signal_groups[register[1].uid].logical_signals["drive"].path
        }

    def test_link_signals_missing_qubit(self):
        setup = DeviceSetup("test")
        setup.add_dataserver(host="localhost", port="8004")
        setup.add_instruments(HDAWG(uid="device_hdawg", address="dev123"))
        setup.add_connections(
            "device_hdawg",
            create_connection(to_signal="q1/drive", ports="SIGOUTS/0"),
        )

        register = QubitRegister([TunableTransmonQubit("q0")])
        with pytest.raises(KeyError, match="Qubit q0 not in device setup"):
            register.link_signals(setup)
