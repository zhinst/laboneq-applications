# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for laboneq_applications.qpu_types.twpa.demo_qpus."""

import pytest

from laboneq_applications.qpu_types.twpa import demo_platform


class TestDemoPlatform:
    def test_single_twpa_qpu(self):
        qp = demo_platform(1)
        [twpa0] = qp.qpu.quantum_elements

        assert qp.setup.uid == "TravellingWaveParametericAmplifiers_1"
        assert qp.setup.qubits == {}
        assert [inst.uid for inst in qp.setup.instruments] == [
            "device_shfqc",
            "device_shfppc",
        ]
        assert list(qp.setup.logical_signal_groups) == ["twpa0"]

        assert twpa0.uid == "twpa0"
        assert twpa0.parameters.readout_lo_frequency == 6e9

    def test_two_twpas_qpu(self):
        qp = demo_platform(2)
        [twpa0, twpa1] = qp.qpu.quantum_elements

        assert qp.setup.uid == "TravellingWaveParametericAmplifiers_2"
        assert qp.setup.qubits == {}
        assert [inst.uid for inst in qp.setup.instruments] == [
            "device_shfqc",
            "device_shfppc",
        ]
        assert list(qp.setup.logical_signal_groups) == ["twpa0", "twpa1"]

        assert twpa0.uid == "twpa0"
        assert twpa0.parameters.readout_lo_frequency == 6e9

        assert twpa1.uid == "twpa1"
        assert twpa1.parameters.readout_lo_frequency == 6e9

    def test_too_few_twpas(self):
        with pytest.raises(ValueError) as err:
            demo_platform(0)
        assert (
            str(err.value)
            == "This testing and demonstration setup requires at least one TWPA."
        )

    def test_too_many_twpas(self):
        with pytest.raises(ValueError) as err:
            demo_platform(5)
        assert (
            str(err.value)
            == "This testing and demonstration setup requires 4 or fewer TWPAs."
        )
