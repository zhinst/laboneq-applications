# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Device setups for tests."""

import pytest
from laboneq.contrib.example_helpers.generate_device_setup import generate_device_setup
from laboneq.dsl.device.connection import create_connection
from laboneq.dsl.quantum import QPU, QuantumPlatform

from laboneq_applications.contrib.qpu_types.bosonic_qubits import (
    demo_platform as demo_platform_bosonic,
)
from laboneq_applications.qpu_types.tunable_coupler import (
    CzParameters,
    TunableCoupler,
    TunableCouplerOperations,
)
from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonQubit,
)
from laboneq_applications.qpu_types.tunable_transmon import (
    demo_platform as demo_platform_transmons,
)
from laboneq_applications.qpu_types.twpa import demo_platform as demo_platform_twpas


@pytest.fixture
def single_tunable_transmon_platform() -> QuantumPlatform:
    """Return a single tunable transmon device setup and its qubits."""
    return demo_platform_transmons(1)


@pytest.fixture
def two_tunable_transmon_platform() -> QuantumPlatform:
    """Return a single tunable transmon device setup and its qubits."""
    return demo_platform_transmons(2)


@pytest.fixture
def single_twpa_platform() -> QuantumPlatform:
    """Return a single-TWPA device setup and its TWPA."""
    return demo_platform_twpas(1)


@pytest.fixture
def single_bosonic_qubit_platform() -> QuantumPlatform:
    """Return a single bosonic qubit device setup and its qubit."""
    return demo_platform_bosonic(1)


@pytest.fixture
def two_bosonic_qubit_platform() -> QuantumPlatform:
    """Return a two-bosonic-qubit device setup."""
    return demo_platform_bosonic(2)


@pytest.fixture
def four_tunable_transmon_cz_platform() -> QuantumPlatform:
    """Four tunable transmons with tunable couplers and "cz" topology edges.

    The QPU uses `TunableCouplerOperations` so callers can use `qop.cz(...)`
    and `qop.iswap(...)`. Each pair (q0-q1), (q1-q2), (q2-q3), (q3-q0) has a
    bidirectional ``"cz"`` edge with a default `CzParameters` instance.
    """
    setup = generate_device_setup(
        number_qubits=4,
        pqsc=[{"serial": "DEV10001"}],
        shfqc=[
            {
                "serial": "DEV12001",
                "number_of_channels": 6,
                "readou_multiplex": 6,
                "options": "SHFQC/PLUS/QC6CH/RTR",
            }
        ],
        hdawg=[
            {
                "serial": "DEV8800",
                "number_of_channels": 8,
                "options": "HDAWG8/CNT/ME/PC",
            }
        ],
        include_flux_lines=True,
        server_host="localhost",
    )

    qubits = TunableTransmonQubit.from_device_setup(setup)

    for q in qubits:
        q.parameters.ge_drive_pulse["sigma"] = 0.25
        q.parameters.readout_amplitude = 0.5
        q.parameters.reset_delay_length = 1e-6
        q.parameters.readout_range_out = -25
        q.parameters.readout_lo_frequency = 7.4e9

    qubits[0].parameters.drive_lo_frequency = 6.4e9
    qubits[0].parameters.resonance_frequency_ge = 6.3e9
    qubits[0].parameters.resonance_frequency_ef = 6.0e9
    qubits[0].parameters.readout_resonator_frequency = 7.0e9

    qubits[1].parameters.drive_lo_frequency = 6.4e9
    qubits[1].parameters.resonance_frequency_ge = 6.5e9
    qubits[1].parameters.resonance_frequency_ef = 6.3e9
    qubits[1].parameters.readout_resonator_frequency = 7.3e9

    qubits[2].parameters.drive_lo_frequency = 6.0e9
    qubits[2].parameters.resonance_frequency_ge = 5.8e9
    qubits[2].parameters.resonance_frequency_ef = 5.6e9
    qubits[2].parameters.readout_resonator_frequency = 7.2e9

    qubits[3].parameters.drive_lo_frequency = 6.0e9
    qubits[3].parameters.resonance_frequency_ge = 5.5e9
    qubits[3].parameters.resonance_frequency_ef = 5.3e9
    qubits[3].parameters.readout_resonator_frequency = 7.5e9

    couplings = {
        "c_q0q1": ("q0", "q1"),
        "c_q1q2": ("q1", "q2"),
        "c_q2q3": ("q2", "q3"),
        "c_q3q0": ("q3", "q0"),
    }

    for n, key in enumerate(couplings):
        # first 4 HDAWG channels are taken by the transmon flux lines
        channel_id = 4 + n
        if key not in setup.logical_signal_groups:
            setup.add_connections(
                "hdawg_0",
                create_connection(
                    to_signal=f"{key}/flux", ports=f"SIGOUTS/{channel_id}"
                ),
            )

    couplers = TunableCoupler.from_device_setup(setup, qubit_uids=couplings.keys())

    qpu = QPU(qubits + couplers, quantum_operations=TunableCouplerOperations())

    for coupler, (q0, q1) in couplings.items():
        qpu.topology.add_edge(
            tag="cz",
            source_node=q0,
            target_node=q1,
            quantum_element=coupler,
            parameters=CzParameters(),
        )
        qpu.topology.add_edge(
            tag="cz",
            source_node=q1,
            target_node=q0,
            quantum_element=coupler,
            parameters=CzParameters(),
        )

    return QuantumPlatform(setup, qpu)
