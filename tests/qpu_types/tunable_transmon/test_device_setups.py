"""Tests for laboneq_applications.qpu_types.tunable_transmon.device_setups."""

import pytest

from laboneq_applications.qpu_types.tunable_transmon import demo_qpu


class TestDemoQPU:
    def test_single_transmon_qpu(self):
        qpu = demo_qpu(1)
        [q0] = qpu.qubits

        assert qpu.setup.uid == "tunable_transmons_1"
        assert qpu.setup.qubits == {}
        assert [inst.uid for inst in qpu.setup.instruments] == [
            "device_shfqc",
            "device_hdawg",
            "device_pqsc",
        ]
        assert list(qpu.setup.logical_signal_groups) == ["q0"]

        assert q0.uid == "q0"
        assert q0.parameters.drive_lo_frequency == 1.50e9

    def test_two_transmon_qpu(self):
        qpu = demo_qpu(2)
        [q0, q1] = qpu.qubits

        assert qpu.setup.uid == "tunable_transmons_2"
        assert qpu.setup.qubits == {}
        assert [inst.uid for inst in qpu.setup.instruments] == [
            "device_shfqc",
            "device_hdawg",
            "device_pqsc",
        ]
        assert list(qpu.setup.logical_signal_groups) == ["q0", "q1"]

        assert q0.uid == "q0"
        assert q0.parameters.drive_lo_frequency == 1.50e9

        assert q1.uid == "q1"
        assert q1.parameters.drive_lo_frequency == 1.60e9

    def test_too_few_qubits(self):
        with pytest.raises(ValueError) as err:
            demo_qpu(0)
        assert (
            str(err.value)
            == "This testing and demonstration setup requires at least one qubit."
        )

    def test_too_many_qubits(self):
        with pytest.raises(ValueError) as err:
            demo_qpu(9)
        assert (
            str(err.value)
            == "This testing and demonstration setup requires 8 or fewer qubits."
        )
