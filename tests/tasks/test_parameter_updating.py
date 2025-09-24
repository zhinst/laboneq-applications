# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import pytest
import uncertainties as unc

from laboneq_applications.qpu_types.tunable_transmon.qubit_types import (
    TunableTransmonQubitParameters,
)
from laboneq_applications.tasks import update_qpu, update_qubits


@pytest.fixture
def parameters():
    """Parameters to update."""

    return {
        "q0": {
            "ge_drive_amplitude_pi": unc.ufloat(0.55, 0.02),
            "ge_drive_amplitude_pi2": 0.255,
        },
        "q1": {
            "resonance_frequency_ef": unc.ufloat(5.58e9, 1.1e6),
            "readout_resonator_frequency": 7e9,
        },
        ("test", "q0", "q1"): {
            "resonance_frequency_ef": unc.ufloat(5.59e9, 1.2e6),
            "readout_resonator_frequency": 8e9,
        },
    }


@pytest.fixture
def qubit_parameters():
    """Qubit parameters to update."""

    return {
        "q0": {
            "ge_drive_amplitude_pi": unc.ufloat(0.55, 0.02),
            "ge_drive_amplitude_pi2": 0.255,
        },
        "q1": {
            "resonance_frequency_ef": unc.ufloat(5.58e9, 1.1e6),
            "readout_resonator_frequency": 7e9,
        },
    }


class TestUpdateQPU:
    def test_run(self, two_tunable_transmon_platform, parameters):
        [q0, q1] = two_tunable_transmon_platform.qpu.quantum_elements
        qpu_topology = two_tunable_transmon_platform.qpu.topology
        qpu_topology.add_edge(
            "test", "q0", "q1", parameters=TunableTransmonQubitParameters()
        )

        assert q0.parameters.ge_drive_amplitude_pi != 0.55
        assert q0.parameters.ge_drive_amplitude_pi2 != 0.255
        assert q1.parameters.resonance_frequency_ef != 5.58e9
        assert q1.parameters.readout_resonator_frequency != 7e9
        assert (
            qpu_topology["test", "q0", "q1"].parameters.resonance_frequency_ef != 5.59e9
        )
        assert (
            qpu_topology["test", "q0", "q1"].parameters.readout_resonator_frequency
            != 8e9
        )

        update_qpu(two_tunable_transmon_platform.qpu, parameters)

        assert q0.parameters.ge_drive_amplitude_pi == 0.55
        assert q0.parameters.ge_drive_amplitude_pi2 == 0.255
        assert q1.parameters.resonance_frequency_ef == 5.58e9
        assert q1.parameters.readout_resonator_frequency == 7e9
        assert (
            qpu_topology["test", "q0", "q1"].parameters.resonance_frequency_ef == 5.59e9
        )
        assert (
            qpu_topology["test", "q0", "q1"].parameters.readout_resonator_frequency
            == 8e9
        )


class TestUpdateQubits:
    def test_run(self, two_tunable_transmon_platform, qubit_parameters):
        [q0, q1] = two_tunable_transmon_platform.qpu.quantum_elements

        assert q0.parameters.ge_drive_amplitude_pi != 0.55
        assert q0.parameters.ge_drive_amplitude_pi2 != 0.255
        assert q1.parameters.resonance_frequency_ef != 5.58e9
        assert q1.parameters.readout_resonator_frequency != 7e9

        update_qubits(two_tunable_transmon_platform.qpu, qubit_parameters)

        assert q0.parameters.ge_drive_amplitude_pi == 0.55
        assert q0.parameters.ge_drive_amplitude_pi2 == 0.255
        assert q1.parameters.resonance_frequency_ef == 5.58e9
        assert q1.parameters.readout_resonator_frequency == 7e9
