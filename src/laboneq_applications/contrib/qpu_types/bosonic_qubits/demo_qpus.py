# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Bosonic qubit device setups for testing and demonstration."""

from typing import Literal

import numpy as np
from laboneq.dsl.device import DeviceSetup, create_connection
from laboneq.dsl.device.instruments import HDAWG, PQSC, QHUB, SHFQC
from laboneq.dsl.quantum.qpu import QPU, QuantumPlatform

from .operations import BosonicQubitOperations
from .qubit_types import (
    BosonicQubit,
    BosonicQubitParameters,
)

# Default max_photon_number for demo qubits.
_DEMO_MAX_PHOTON_NUMBER = 5

# On the SHFQC, SG channels are paired into synthesizers:
#   synthesizer 0 → channels 0, 1
#   synthesizer 1 → channels 2, 3
#   synthesizer 2 → channels 4, 5
#
# Two channels that share a synthesizer MUST use the same LO frequency.
# Layout per SHFQC (2 qubits per unit):
#   ch0  → qubit 0 / drive_memory          (synthesizer 0)
#   ch1  → qubit 1 / drive_memory          (synthesizer 0)
#   ch2  → qubit 0 / drive_transmon_at_n*  (synthesizer 1, all share this port)
#   ch3  → qubit 1 / drive_transmon_at_n*  (synthesizer 1, all share this port)
#   QA0  → all qubits / measure + acquire
#
# Both memory drives on the same SHFQC use the same LO (synthesizer 0 constraint).
# Both transmon drives on the same SHFQC use the same LO (synthesizer 1 constraint).
# The memory LO (synth 0) and transmon LO (synth 1) are independent.
_QUBITS_PER_SHFQC = 2
_HDAWG_SIGOUTS = 8


def demo_platform(
    n_qubits: int,
    leader_instrument: Literal["PQSC", "QHub"] = "PQSC",
) -> QuantumPlatform:
    """Return a demo bosonic qubit QPU with the specified number of qubits.

    The returned setup consists of:

    - 1 leader instrument (PQSC or QHub)
    - N SHFQC (for memory and transmon drive lines and readout)
    - M HDAWG (for the SWAP flux lines)

    N and M are determined automatically from the number of qubits.
    Two qubits share each SHFQC.

    The following device options are used:

    - PQSC (default)
    - SHFQC/QC6CH
    - HDAWG8/MF/ME/SKW/PC

    Arguments:
        n_qubits:
            Number of qubits to include in the QPU.
        leader_instrument:
            Overrides the default PQSC device for the setup.

    Returns:
        The quantum platform (device setup + QPU).
    """
    setup = bosonic_qubit_setup(n_qubits, leader_instrument)
    qubits = bosonic_qubits(n_qubits, setup)
    quantum_operations = BosonicQubitOperations()
    qpu = QPU(qubits, quantum_operations=quantum_operations)
    return QuantumPlatform(setup=setup, qpu=qpu)


def _qubit_ids(n_qubits: int) -> list[str]:
    return [f"q{i}" for i in range(n_qubits)]


def _create_leader_instrument(
    leader_instrument_type: Literal["PQSC", "QHub"],
) -> PQSC | QHUB:
    if leader_instrument_type == "PQSC":
        return PQSC(uid="device_pqsc", address="dev125", device_options="PQSC")

    if leader_instrument_type == "QHub":
        return QHUB(uid="device_qhub", address="dev125", device_options="QHub")

    raise ValueError(
        f"Invalid choice for leader instrument {leader_instrument_type}, "
        "expected one of PQSC, QHub"
    )


def bosonic_qubit_setup(
    n_qubits: int,
    leader_instrument: Literal["PQSC", "QHub"] = "PQSC",
) -> DeviceSetup:
    """Return a demo bosonic qubit device setup.

    Two qubits share each SHFQC. Their memory drives are on
    SHFQC SG channels 0 and 1 (synthesizer 0), their transmon
    drives share channels 2 and 3 (synthesizer 1), and their
    readout multiplexes onto the single QA channel. Each qubit's
    SWAP drive occupies one HDAWG sigout.

    Arguments:
        n_qubits:
            Number of qubits to include in the QPU.
        leader_instrument:
            Overrides the default PQSC device for the setup.

    Returns:
        The device setup.
    """
    if n_qubits < 1:
        raise ValueError(
            "This testing and demonstration setup requires at least one qubit.",
        )

    qubit_ids = _qubit_ids(n_qubits)
    n_shfqc = n_qubits // _QUBITS_PER_SHFQC + (
        1 if n_qubits % _QUBITS_PER_SHFQC > 0 else 0
    )
    n_hdawg = n_qubits // _HDAWG_SIGOUTS + (1 if n_qubits % _HDAWG_SIGOUTS > 0 else 0)

    setup = DeviceSetup(f"bosonic_qubits_{n_qubits}")
    setup.add_dataserver(host="localhost", port="8004")

    for i in range(n_shfqc):
        setup.add_instruments(
            SHFQC(
                uid=f"device_shfqc_{i}",
                address=f"dev123{i}",
                device_options="SHFQC/QC6CH",
            ),
        )

    for i in range(n_hdawg):
        setup.add_instruments(
            HDAWG(
                uid=f"device_hdawg_{i}",
                address=f"dev124{i}",
                device_options="HDAWG8/MF/ME/SKW/PC",
            ),
        )

    setup.add_instruments(_create_leader_instrument(leader_instrument))

    for i, qubit in enumerate(qubit_ids):
        shfqc_idx = i // _QUBITS_PER_SHFQC
        qubit_slot = i % _QUBITS_PER_SHFQC  # 0 or 1 within the SHFQC

        # Memory drive: channels 0, 1 (synthesizer 0 — same LO required)
        sg_memory = qubit_slot
        # Transmon drives: channels 2, 3 (synthesizer 1 — same LO required)
        sg_transmon = qubit_slot + 2

        hdawg_idx = i // _HDAWG_SIGOUTS
        hdawg_sigout = i % _HDAWG_SIGOUTS

        setup.add_connections(
            f"device_shfqc_{shfqc_idx}",
            # Memory cavity drive (dedicated SG channel on synthesizer 0)
            create_connection(
                to_signal=f"{qubit}/drive_memory",
                ports=f"SGCHANNELS/{sg_memory}/OUTPUT",
            ),
        )

        # All drive_transmon_at_nX signals share one SG channel on synthesizer 1.
        for n in range(_DEMO_MAX_PHOTON_NUMBER + 1):
            setup.add_connections(
                f"device_shfqc_{shfqc_idx}",
                create_connection(
                    to_signal=f"{qubit}/drive_transmon_at_n{n}",
                    ports=f"SGCHANNELS/{sg_transmon}/OUTPUT",
                ),
            )

        # Readout: all qubits on the same SHFQC share the QA channel.
        setup.add_connections(
            f"device_shfqc_{shfqc_idx}",
            create_connection(
                to_signal=f"{qubit}/measure",
                ports="QACHANNELS/0/OUTPUT",
            ),
            create_connection(
                to_signal=f"{qubit}/acquire",
                ports="QACHANNELS/0/INPUT",
            ),
        )

        # SWAP drive: dedicated HDAWG sigout (LF mode, DC flux pulses).
        setup.add_connections(
            f"device_hdawg_{hdawg_idx}",
            create_connection(
                to_signal=f"{qubit}/swap_drive",
                ports=f"SIGOUTS/{hdawg_sigout}",
            ),
        )

    return setup


def bosonic_qubits(
    n_qubits: int,
    setup: DeviceSetup,
) -> list[BosonicQubit]:
    """Return demo bosonic qubits for the given device setup.

    Qubit parameters are set to slightly different values per qubit so
    they can be distinguished in demonstrations and tests.

    Pairs of qubits on the same SHFQC share the same memory LO
    (synthesizer 0 constraint) and the same transmon LO (synthesizer 1
    constraint).

    Arguments:
        n_qubits:
            Number of qubits to include in the QPU.
        setup:
            The device setup returned by [bosonic_qubit_setup]()
            with the same number of qubits.

    Returns:
        The list of qubits.
    """

    def q_param(
        i: int, base: float, unit: float = 1.0, dq: float = 0.01, max_: float = np.inf
    ) -> float:
        """Tweak a parameter a tiny amount to distinguish qubits."""
        return min((base + i * dq) * unit, max_)

    # Qubit pairs on the same SHFQC share LO frequencies.
    # memory LO on synthesizer 0, transmon LO on synthesizer 1.
    # Resonance frequencies are chosen so that all IFs are within
    # the ±1 GHz SHFQC SG bandwidth.
    #
    #   memory LO  ≈ 5.0 GHz, memory resonance ≈ 5.1 GHz → IF ≈ 100 MHz
    #   transmon LO ≈ 6.0 GHz, transmon resonance ≈ 6.1 GHz → IF ≈ 100 MHz
    #   readout LO  = 7.0 GHz (shared)

    qubits = []
    for i in range(n_qubits):
        # Pairs of qubits on the same SHFQC use the same LO.
        pair = i // _QUBITS_PER_SHFQC

        q = BosonicQubit.from_logical_signal_group(
            f"q{i}",
            setup.logical_signal_groups[f"q{i}"],
            parameters=BosonicQubitParameters(
                # Memory cavity (synthesizer 0 — shared LO within pair)
                memory_lo_frequency=int(q_param(pair, 5.0, 1e9, dq=0.2)),
                memory_resonance_frequency=q_param(i, 5.1, 1e9),
                memory_drive_amplitude=0.2,
                memory_drive_length=100e-9,
                # Ancilla transmon (synthesizer 1 — shared LO within pair)
                transmon_lo_frequency=int(q_param(pair, 6.0, 1e9, dq=0.2)),
                transmon_resonance_frequency_at_n0=q_param(i, 6.1, 1e9),
                # Dispersive shift
                chi=q_param(i, -1.0, 1e6, dq=0.01),
                # Selective drive amplitudes (n = 0 .. max_photon_number)
                selective_transmon_X180_amplitudes=[0.1]
                * (_DEMO_MAX_PHOTON_NUMBER + 1),
                selective_transmon_X90_amplitudes=[0.05]
                * (_DEMO_MAX_PHOTON_NUMBER + 1),
                selective_transmon_Rx_length=2e-6,
                # Readout
                readout_lo_frequency=7e9,
                readout_resonator_frequency=q_param(i, 7.1, 1e9),
                readout_amplitude=1.0,
                readout_length=2e-6,
                readout_integration_length=2e-6,
                # SWAP gate
                swap_length=500e-9,
                swap_amplitude=0.1,
                # Reset
                reset_delay_length=1e-6,
                # Displacement calibration result (demo value)
                displacement_amp_per_unit_beta=0.5,
            ),
        )
        qubits.append(q)

    return qubits
