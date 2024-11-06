# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for laboneq_applications.qpu_types.tunable_transmon.device_setups."""

import pytest

from laboneq_applications.qpu_types.tunable_transmon import demo_platform


class TestDemoPlatform:
    def test_single_transmon_qpu(self):
        qp = demo_platform(1)
        [q0] = qp.qpu.qubits

        assert qp.setup.uid == "tunable_transmons_1"
        assert qp.setup.qubits == {}
        assert [inst.uid for inst in qp.setup.instruments] == [
            "device_shfqc",
            "device_hdawg",
            "device_pqsc",
        ]
        assert list(qp.setup.logical_signal_groups) == ["q0"]

        assert q0.uid == "q0"
        assert q0.parameters.drive_lo_frequency == 6.40e9

    def test_two_transmon_qpu(self):
        qp = demo_platform(2)
        [q0, q1] = qp.qpu.qubits

        assert qp.setup.uid == "tunable_transmons_2"
        assert qp.setup.qubits == {}
        assert [inst.uid for inst in qp.setup.instruments] == [
            "device_shfqc",
            "device_hdawg",
            "device_pqsc",
        ]
        assert list(qp.setup.logical_signal_groups) == ["q0", "q1"]

        assert q0.uid == "q0"
        assert q0.parameters.drive_lo_frequency == 6.40e9

        assert q1.uid == "q1"
        assert q1.parameters.drive_lo_frequency == 6.40e9

    def test_too_few_qubits(self):
        with pytest.raises(ValueError) as err:
            demo_platform(0)
        assert (
            str(err.value)
            == "This testing and demonstration setup requires at least one qubit."
        )

    def test_too_many_qubits(self):
        with pytest.raises(ValueError) as err:
            demo_platform(9)
        assert (
            str(err.value)
            == "This testing and demonstration setup requires 8 or fewer qubits."
        )
