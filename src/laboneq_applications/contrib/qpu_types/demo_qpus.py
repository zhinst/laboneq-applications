# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Coupled tunable-transmon device setups for testing and demonstration.

This module provides a demo quantum platform consisting of tunable transmon
qubits connected in a chain by tunable couplers, including the QPU topology
edges required by the coupler-mediated two-qubit gates and experiments
(see the `cz`, `iswap` and `discharge_coupler` quantum operations of
`TunableCouplerOperations`, and the chevron, jazz and phase-correction
experiment workflows in `laboneq_applications.contrib.experiments.two_qubit`).
"""

from __future__ import annotations

from laboneq.contrib.example_helpers.generate_device_setup import (
    generate_device_setup,
)
from laboneq.dsl.device.connection import create_connection
from laboneq.dsl.quantum import QPU, QuantumPlatform

from laboneq_applications.qpu_types.tunable_coupler import (
    CzParameters,
    TunableCoupler,
    TunableCouplerOperations,
)
from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonQubit,
)

# demo qubit parameters for up to four qubits
_DRIVE_LO_FREQUENCIES = [6.4e9, 6.4e9, 6.0e9, 6.0e9]
_RESONANCE_FREQUENCIES_GE = [6.3e9, 6.5e9, 5.8e9, 5.5e9]
_RESONANCE_FREQUENCIES_EF = [6.0e9, 6.3e9, 5.6e9, 5.3e9]
_READOUT_RESONATOR_FREQUENCIES = [7.0e9, 7.3e9, 7.2e9, 7.5e9]

_MIN_QUBITS = 2
_MAX_QUBITS = 4


def demo_platform_coupled_transmons(n_qubits: int = 2) -> QuantumPlatform:
    """Return a demo QPU with transmons connected in a chain by couplers.

    The returned setup consists of:

    - 1 PQSC
    - 1 SHFQC+/QC6CH (for the qubit drive and measurement lines)
    - 1 HDAWG8 (for the qubit and coupler flux lines)

    The qubits are connected in a chain by tunable couplers, i.e. coupler
    ``c_q0q1`` connects qubits ``q0`` and ``q1``, and so on. For each
    coupler, the QPU topology contains, in both directions:

    - an edge tagged ``"coupler"`` with no parameters, describing the
      physical connectivity;
    - an edge tagged ``"cz"`` with `CzParameters`, used by the
      coupler-mediated CZ gate and its calibration experiments.

    The maximum number of qubits is 4, at which point all eight flux
    channels of the HDAWG are in use (4 qubits + 3 couplers).

    Arguments:
        n_qubits:
            Number of qubits to include in the QPU (between 2 and 4).

    Returns:
        The quantum platform with the device setup and the QPU.
    """
    if not _MIN_QUBITS <= n_qubits <= _MAX_QUBITS:
        error_message = (
            f"The number of qubits must be between {_MIN_QUBITS} and "
            f"{_MAX_QUBITS}, got {n_qubits}."
        )
        raise ValueError(error_message)

    setup = generate_device_setup(
        number_qubits=n_qubits,
        pqsc=[{"serial": "DEV10001"}],
        shfqc=[
            {
                "serial": "DEV12001",
                "number_of_channels": 6,
                "readout_multiplex": 6,
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

    for i, q in enumerate(qubits):
        q.parameters.ge_drive_pulse["sigma"] = 0.25
        q.parameters.readout_amplitude = 0.5
        q.parameters.reset_delay_length = 1e-6
        q.parameters.readout_range_out = -25
        q.parameters.readout_lo_frequency = 7.4e9
        q.parameters.drive_lo_frequency = _DRIVE_LO_FREQUENCIES[i]
        q.parameters.resonance_frequency_ge = _RESONANCE_FREQUENCIES_GE[i]
        q.parameters.resonance_frequency_ef = _RESONANCE_FREQUENCIES_EF[i]
        q.parameters.readout_resonator_frequency = _READOUT_RESONATOR_FREQUENCIES[i]

    # couple the qubits in a chain
    couplings = {f"c_q{i}q{i + 1}": (f"q{i}", f"q{i + 1}") for i in range(n_qubits - 1)}

    # add flux lines for the tunable couplers
    for n, key in enumerate(couplings):
        # the first channels are already occupied by the qubit flux lines
        channel_id = n_qubits + n
        if key not in setup.logical_signal_groups:
            setup.add_connections(
                "hdawg_0",
                create_connection(
                    to_signal=f"{key}/flux", ports=f"SIGOUTS/{channel_id}"
                ),
            )

    couplers = TunableCoupler.from_device_setup(setup, qubit_uids=couplings.keys())

    quantum_operations = TunableCouplerOperations()

    qpu = QPU(qubits + couplers, quantum_operations=quantum_operations)

    for coupler, (q0, q1) in couplings.items():
        for source, target in ((q0, q1), (q1, q0)):
            qpu.topology.add_edge(
                source_node=source,
                target_node=target,
                quantum_element=coupler,
                tag="coupler",
            )
            # qubit-driven CZ flavor: parametric drive on the qubit flux
            # line (control_pulse), baseband bias pulse on the coupler
            qpu.topology.add_edge(
                source_node=source,
                target_node=target,
                quantum_element=coupler,
                tag="cz",
                parameters=CzParameters(
                    control_pulse={
                        "function": "modulated_flux",
                        "amplitude": 0.0,
                        "length": 100e-9,
                        "frequency": 300e6,
                    },
                    coupler_pulse={
                        "function": "const",
                        "amplitude": 0.0,
                        "length": 100e-9,
                    },
                ),
            )

    return QuantumPlatform(setup=setup, qpu=qpu)
