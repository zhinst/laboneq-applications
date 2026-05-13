# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tunable transmon qubit device setups for testing and demonstration."""

from typing import Literal

import numpy as np
from laboneq.dsl.calibration import Oscillator, SignalCalibration
from laboneq.dsl.device import DeviceSetup, create_connection
from laboneq.dsl.device.instruments import HDAWG, PQSC, QHUB, SHFQC
from laboneq.dsl.enums import ModulationType
from laboneq.dsl.quantum.qpu import QPU, QuantumPlatform

from .operations import TunableTransmonOperations
from .qubit_types import (
    TunableTransmonQubit,
    TunableTransmonQubitParameters,
)


def demo_platform(
    n_qubits: int,
    leader_instrument: Literal["PQSC", "QHub"] = "PQSC",
) -> QuantumPlatform:
    """Return a demo tunable transmon QPU with the specified number of qubits.

    The returned setup consists of:

    - 1 leader instrument (PQSC or QHub)
    - N SHFQC (for the qubit drive and measurement lines)
    - M HDAWG (for the qubit flux lines)

    N and M are determined automatically from the number of qubits.

    The following device options are used:

    - PQSC (default)
    - SHFQC/QC6CH
    - HDAWG8/MF/ME/SKW/PC

    The qubits share a single multiplexed readout line.

    Arguments:
        n_qubits:
            Number of qubits to include in the QPU.
        leader_instrument:
            Overrides the default PQSC device for the setup.

    Returns:
        The QPU.

    !!! version-changed "Changed in version 26.4.0."
        The number of qubits, `n_qubits`, may be arbitrarily large.
        In earlier versions, it was allowed to be at most six.
    """
    setup = tunable_transmon_setup(n_qubits, leader_instrument)
    qubits = tunable_transmon_qubits(n_qubits, setup)
    quantum_operations = TunableTransmonOperations()
    qpu = QPU(qubits, quantum_operations=quantum_operations)
    return QuantumPlatform(setup=setup, qpu=qpu)


class _DeviceAndPortSet:
    """A set of ports on a set of devices.

    Arguments:
        ports:
            Number of ports per device.
        qubits:
            List of qubit uids.

    Attributes:
        num_devices:
            Total number of devices needed.
        qubit_to_port:
            A map from qubit uid to a tuple of the device and port number.
    """

    def __init__(self, *, ports: int, qubits: list[str]):
        n_qubits = len(qubits)

        self.num_devices = n_qubits // ports + (1 if n_qubits % ports > 0 else 0)
        self.qubit_to_port = {
            qubit_id: (i // ports, i % ports) for i, qubit_id in enumerate(qubits)
        }


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


def tunable_transmon_setup(
    n_qubits: int,
    leader_instrument: Literal["PQSC", "QHub"] = "PQSC",
) -> DeviceSetup:
    """Return a demo tunable transmon device setup.

    The returned setup consists of:

    - 1 leader instrument (PQSC or QHub)
    - N SHFQC (for the qubit drive and measurement lines)
    - M HDAWG (for the qubit flux lines)

    N and M are determined automatically from the number of qubits.

    The following device options are used:

    - PQSC (default)
    - SHFQC/QC6CH
    - HDAWG8/MF/ME/SKW/PC

    The qubits share a single multiplexed readout line.

    Arguments:
        n_qubits:
            Number of qubits to include in the QPU.
        leader_instrument:
            Overrides the default PQSC device for the setup.

    Returns:
        The device setup.

    !!! version-changed "Changed in version 26.4.0."
        The number of qubits, `n_qubits`, may be arbitrarily large.
        In earlier versions, it was allowed to be at most six.
    """
    if n_qubits < 1:
        raise ValueError(
            "This testing and demonstration setup requires at least one qubit.",
        )

    qubit_ids = [f"q{i}" for i in range(n_qubits)]

    shfqc_drive_lines = _DeviceAndPortSet(ports=6, qubits=qubit_ids)
    hdawg_sigouts = _DeviceAndPortSet(ports=8, qubits=qubit_ids)

    setup = DeviceSetup(f"tunable_transmons_{n_qubits}")
    setup.add_dataserver(host="localhost", port="8004")

    for i in range(shfqc_drive_lines.num_devices):
        setup.add_instruments(
            SHFQC(
                uid=f"device_shfqc_{i}",
                address=f"dev123{i}",
                device_options="SHFQC/QC6CH",
            ),
        )

    for i in range(hdawg_sigouts.num_devices):
        setup.add_instruments(
            HDAWG(
                uid=f"device_hdawg_{i}",
                address=f"dev124{i}",
                device_options="HDAWG8/MF/ME/SKW/PC",
            ),
        )

    setup.add_instruments(_create_leader_instrument(leader_instrument))

    for qubit in qubit_ids:
        shfqc, shfqc_port = shfqc_drive_lines.qubit_to_port[qubit]
        hdawg, hdawg_port = hdawg_sigouts.qubit_to_port[qubit]
        setup.add_connections(
            f"device_shfqc_{shfqc}",
            # each qubit uses their own drive line:
            create_connection(
                to_signal=f"{qubit}/drive",
                ports=f"SGCHANNELS/{shfqc_port}/OUTPUT",
            ),
            create_connection(
                to_signal=f"{qubit}/drive_ef",
                ports=f"SGCHANNELS/{shfqc_port}/OUTPUT",
            ),
            # all qubits multiplex on the measure and acquire lines:
            create_connection(
                to_signal=f"{qubit}/measure",
                ports="QACHANNELS/0/OUTPUT",
            ),
            create_connection(to_signal=f"{qubit}/acquire", ports="QACHANNELS/0/INPUT"),
        )
        setup.add_connections(
            f"device_hdawg_{hdawg}",
            # each qubit has its own flux line:
            create_connection(to_signal=f"{qubit}/flux", ports=f"SIGOUTS/{hdawg_port}"),
        )

    for qubit in qubit_ids:
        for line, frequency, mod_type in [
            ("drive", 5e9, ModulationType.HARDWARE),
            ("drive_ef", 6e9, ModulationType.HARDWARE),
            ("measure", 4e9, ModulationType.SOFTWARE),
        ]:
            logical_signal = setup.logical_signal_by_uid(f"{qubit}/{line}")
            oscillator = Oscillator(modulation_type=mod_type)
            logical_signal.calibration = SignalCalibration(
                local_oscillator=Oscillator(frequency=frequency),
                oscillator=oscillator,
            )
            if line == "measure":
                # acquire and measure lines must share the same oscillator
                acquire_signal = setup.logical_signal_by_uid(f"{qubit}/acquire")
                acquire_signal.calibration = SignalCalibration(
                    local_oscillator=Oscillator(frequency=frequency),
                    oscillator=oscillator,
                )

    return setup


def tunable_transmon_qubits(
    n_qubits: int,
    setup: DeviceSetup,
) -> list[TunableTransmonQubit]:
    """Return demo tunable transmon device qubits.

    The qubits are constructed for the device setup returned by
    [tunable_transmon_setup]().

    The qubits share a single readout local oscillator frequency
    because they share a single multiplexed readout line in the
    device setup.

    Other qubit parameters (e.g. `drive_lo_frequency`) are set to
    slightly different values to allow them to be distinguished in
    demonstrations and tests.

    Arguments:
        n_qubits:
            Number of qubits to include in the QPU.
        setup:
            The device setup. It is assumed that the setup
            is the setup returned by [tunable_transmon_setup]()
            called with the same number of qubits.

    Returns:
        The list of qubits.
    """

    def q_param(
        i: int, base: float, unit: float = 1.0, dq: float = 0.01, max_: float = np.inf
    ) -> float:
        """Tweak qubit parameter a tiny amount to distinguish them."""
        return min((base + i * dq) * unit, max_)

    qubits = []
    for i in range(n_qubits):
        q = TunableTransmonQubit.from_logical_signal_group(
            f"q{i}",
            setup.logical_signal_groups[f"q{i}"],
            parameters=TunableTransmonQubitParameters(
                # Groups of qubits share the same LO frequency.
                # The LO advances in 200 MHz steps (SHF hardware constraint).
                # Convert to integer, otherwise some configurations
                #   may encounter issues with LabOne Q compiler
                drive_lo_frequency=int(q_param(i // 20, 6.4, 1e9, dq=0.2, max_=8.4e9)),
                resonance_frequency_ge=q_param(i, 6.5, 1e9),
                resonance_frequency_ef=q_param(i, 6.3, 1e9),
                readout_lo_frequency=7e9,
                readout_resonator_frequency=q_param(i, 7.1, 1e9),
                ge_drive_amplitude_pi=q_param(i, 0.8, max_=1.0),
                ge_drive_amplitude_pi2=q_param(i, 0.4, max_=1.0),
                ge_drive_length=51e-9,
                ge_drive_pulse={
                    "function": "drag",
                    "beta": 0.01,
                    "sigma": 0.21,
                },
                ef_drive_amplitude_pi=q_param(i, 0.7, max_=1.0),
                ef_drive_amplitude_pi2=q_param(i, 0.3, max_=1.0),
                ef_drive_length=52e-9,
                ef_drive_pulse={
                    "function": "drag",
                    "beta": 0.01,
                    "sigma": 0.21,
                },
            ),
        )
        qubits.append(q)

    return qubits
