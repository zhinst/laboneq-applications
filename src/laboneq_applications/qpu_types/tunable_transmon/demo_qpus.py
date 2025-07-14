# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tunable transmon qubit device setups for testing and demonstration."""

from laboneq.dsl.calibration import Oscillator, SignalCalibration
from laboneq.dsl.device import DeviceSetup, create_connection
from laboneq.dsl.device.instruments import HDAWG, PQSC, SHFQC
from laboneq.dsl.enums import ModulationType
from laboneq.dsl.quantum.qpu import QPU, QuantumPlatform

from .operations import TunableTransmonOperations
from .qubit_types import (
    TunableTransmonQubit,
    TunableTransmonQubitParameters,
)


def demo_platform(n_qubits: int) -> QuantumPlatform:
    """Return a demo tunable transmon QPU with the specified number of qubits.

    The returned setup consists of:

    - 1 PQSC
    - 1 SHFQC (for the qubit drive and measurement lines)
    - 1 HDAWG (for the qubit flux lines)

    with device options:

    - 1 PQSC
    - 1 SHFQC/QC6CH
    - 1 HDAWG8/MF/ME/SKW/PC

    The maximum number of qubits is 6 (which is the number of drive lines on
    the SHFQC).

    The qubits share a single multiplexed readout line.

    Arguments:
        n_qubits:
            Number of qubits to include in the QPU.

    Returns:
        The QPU.
    """
    setup = tunable_transmon_setup(n_qubits)
    qubits = tunable_transmon_qubits(n_qubits, setup)
    quantum_operations = TunableTransmonOperations()
    qpu = QPU(qubits, quantum_operations=quantum_operations)
    return QuantumPlatform(setup=setup, qpu=qpu)


def tunable_transmon_setup(n_qubits: int) -> DeviceSetup:
    """Return a demo tunable transmon device setup.

    The returned setup consists of:

    - 1 PQSC
    - 1 SHFQC (for the qubit drive and measurement lines)
    - 1 HDAWG (for the qubit flux lines)

    with device options:

    - 1 PQSC
    - 1 SHFQC/QC6CH
    - 1 HDAWG8/MF/ME/SKW/PC

    The maximum number of qubits is 6 (which is the number of drive lines on
    the SHFQC).

    The qubits share a single multiplexed readout line.

    Arguments:
        n_qubits:
            Number of qubits to include in the QPU.

    Returns:
        The device setup.
    """
    if n_qubits < 1:
        raise ValueError(
            "This testing and demonstration setup requires at least one qubit.",
        )
    SHFQC_DRIVE_LINES: int = 6  # noqa: N806
    if n_qubits > SHFQC_DRIVE_LINES:
        raise ValueError(
            "This testing and demonstration setup requires 8 or fewer qubits.",
        )

    qubit_ids = [f"q{i}" for i in range(n_qubits)]

    setup = DeviceSetup(f"tunable_transmons_{n_qubits}")
    setup.add_dataserver(host="localhost", port="8004")

    setup.add_instruments(
        SHFQC(uid="device_shfqc", address="dev123", device_options="SHFQC/QC6CH"),
    )
    setup.add_instruments(
        HDAWG(
            uid="device_hdawg",
            address="dev124",
            device_options="HDAWG8/MF/ME/SKW/PC",
        ),
    )
    setup.add_instruments(
        PQSC(uid="device_pqsc", address="dev125", device_options="PQSC")
    )

    for i, qubit in enumerate(qubit_ids):
        setup.add_connections(
            "device_shfqc",
            # each qubit uses their own drive line:
            create_connection(
                to_signal=f"{qubit}/drive",
                ports=f"SGCHANNELS/{i}/OUTPUT",
            ),
            create_connection(
                to_signal=f"{qubit}/drive_ef",
                ports=f"SGCHANNELS/{i}/OUTPUT",
            ),
            # all qubits multiplex on the measure and acquire lines:
            create_connection(
                to_signal=f"{qubit}/measure",
                ports="QACHANNELS/0/OUTPUT",
            ),
            create_connection(to_signal=f"{qubit}/acquire", ports="QACHANNELS/0/INPUT"),
        )

    for i, qubit in enumerate(qubit_ids):
        setup.add_connections(
            "device_hdawg",
            # each qubit has its own flux line:
            create_connection(to_signal=f"{qubit}/flux", ports=f"SIGOUTS/{i}"),
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
        i: int,
        base: float,
        unit: float = 1.0,
        dq: float = 0.01,
    ) -> float:
        """Tweak qubit parameter a tiny amount to distinguish them."""
        return (base + i * dq) * unit

    qubits = []
    for i in range(n_qubits):
        q = TunableTransmonQubit.from_logical_signal_group(
            f"q{i}",
            setup.logical_signal_groups[f"q{i}"],
            parameters=TunableTransmonQubitParameters(
                # A pair of neighbor qubits share the same LO frequency
                # Convert to integer, otherwise some configurations
                #   may encounter issues with LabOne Q compiler
                drive_lo_frequency=int(q_param(i // 2, 6.4, 1e9, dq=0.2)),
                resonance_frequency_ge=q_param(i, 6.5, 1e9),
                resonance_frequency_ef=q_param(i, 6.3, 1e9),
                readout_lo_frequency=7e9,
                readout_resonator_frequency=q_param(i, 7.1, 1e9),
                ge_drive_amplitude_pi=q_param(i, 0.8),
                ge_drive_amplitude_pi2=q_param(i, 0.4),
                ge_drive_length=51e-9,
                ge_drive_pulse={
                    "function": "drag",
                    "beta": 0.01,
                    "sigma": 0.21,
                },
                ef_drive_amplitude_pi=q_param(i, 0.7),
                ef_drive_amplitude_pi2=q_param(i, 0.3),
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
