# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Smoke-tests for the zz_coupling_strength experiments."""

import numpy as np
import pytest
from laboneq.contrib.example_helpers.generate_device_setup import generate_device_setup
from laboneq.dsl.device.connection import create_connection
from laboneq.dsl.quantum import QPU, QuantumPlatform

from laboneq_applications.contrib.experiments import zz_coupling_strength
from laboneq_applications.qpu_types.tunable_coupler import TunableCoupler
from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonOperations,
    TunableTransmonQubit,
)


@pytest.fixture
def four_tunable_transmon_square_topology_platform() -> QuantumPlatform:
    """Return a single tunable transmon device setup and its qubits."""

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

    # define desired couplings
    couplings = {
        "c_q0q1": ("q0", "q1"),
        "c_q1q2": ("q1", "q2"),
        "c_q2q3": ("q2", "q3"),
        "c_q3q0": ("q3", "q0"),
    }

    # add signal lines for tunable couplers
    for n, key in enumerate(couplings):
        # first channels are already occupied by the flux line of TunableTransmon
        channel_id = 4 + n
        if key in setup.logical_signal_groups:
            pass
        else:
            setup.add_connections(
                "hdawg_0",
                create_connection(
                    to_signal=f"{key}/flux", ports=f"SIGOUTS/{channel_id}"
                ),
            )

    couplers = TunableCoupler.from_device_setup(setup, qubit_uids=couplings.keys())

    qops = TunableTransmonOperations()

    qpu = QPU(qubits + couplers, quantum_operations=qops)

    for coupler, (q0, q1) in couplings.items():
        qpu.topology.add_edge(
            source_node=q0, target_node=q1, quantum_element=coupler, tag="coupler"
        )
        qpu.topology.add_edge(
            source_node=q1, target_node=q0, quantum_element=coupler, tag="coupler"
        )

    return QuantumPlatform(setup, qpu)


class TestZZCouplingStrength:
    def test_zz_coupling_strength(self, four_tunable_transmon_square_topology_platform):
        platform = four_tunable_transmon_square_topology_platform
        qpu = platform.qpu
        qubit_pairs = [["q0", "q1"], ["q2", "q3"]]
        options = zz_coupling_strength.experiment_workflow.options()
        options.count(2**13)
        options.do_analysis(True)
        session = platform.session(do_emulation=True)
        wf = zz_coupling_strength.experiment_workflow(
            session=session,
            qpu=qpu,
            qubit_pairs=qubit_pairs,
            biases=[np.linspace(-0.06, 0.06, 11) for _ in range(len(qubit_pairs))],
            delays=[
                np.linspace(0, 10e-6, 11) * (_ + 1) for _ in range(len(qubit_pairs))
            ],
            options=options,
        )
        wf.run()
