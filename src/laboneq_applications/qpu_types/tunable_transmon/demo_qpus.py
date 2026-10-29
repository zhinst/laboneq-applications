# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tunable transmon qubit device setups for testing and demonstration."""

from typing import TYPE_CHECKING, Literal

import numpy as np
from laboneq.dsl.calibration import Oscillator, SignalCalibration
from laboneq.dsl.device import DeviceSetup, create_connection
from laboneq.dsl.device.instruments import HDAWG, PQSC, QHUB, SHFQC, ZQCS
from laboneq.dsl.device.instruments.shfqa import SHFQA
from laboneq.dsl.enums import ModulationType
from laboneq.dsl.quantum.qpu import QPU, QuantumPlatform

from .operations import TunableTransmonOperations
from .qubit_types import (
    TunableTransmonQubit,
    TunableTransmonQubitParameters,
)

if TYPE_CHECKING:
    from zhinst_zqcs.dsl.device.setup_descriptions_zqcs import SetupDescriptionZQCS

# ZQCS hardware constants
_GLOBAL_DELAY = 5e-6
_QA_START_DELAY = 40e-9
_RO_INT_DELAY = 456.0e-9


def demo_platform(
    n_qubits: int,
    leader_instrument: Literal["PQSC", "QHub", "ZQCS"] = "PQSC",
    *,
    readout_multiplex: bool = True,
) -> QuantumPlatform:
    """Return a demo tunable transmon QPU with the specified number of qubits.

    With a PQSC or QHub leader, the returned setup consists of:

    - 1 leader instrument (PQSC or QHub)
    - N SHFQC (for the qubit drive and measurement lines)
    - M HDAWG (for the qubit flux lines)
    - K SHFQA (for the qubit measure and acquire lines)

    N, M and K are determined automatically from the number of qubits.

    The following device options are used:

    - PQSC (default)
    - SHFQC/QC6CH
    - HDAWG8/MF/ME/SKW/PC

    If `readout_multiplex` is `True` (default), groups of up to 16 qubits
    share a multiplexed readout line. Otherwise every qubit is given its
    own measure and acquire channel.

    With a ZQCS leader, the setup consists of a single ZQCS instrument and
    requires the `zhinst_zqcs` package. If `readout_multiplex` is `True`,
    all qubits share one readout line.

    Arguments:
        n_qubits:
            Number of qubits to include in the QPU.
        leader_instrument:
            Overrides the default PQSC device for the setup.
        readout_multiplex:
            Whether to multiplex the measure and acquire lines.

    Returns:
        The QPU.

    !!! version-changed "Changed in version 26.4.0"
        The number of qubits, `n_qubits`, may be arbitrarily large.
        In earlier versions, it was allowed to be at most six.
    """
    if leader_instrument == "ZQCS":
        setup = tunable_transmon_setup_zqcs(
            n_qubits, readout_multiplex=readout_multiplex
        )
        qubits = tunable_transmon_qubits_zqcs(setup)
    else:
        setup = tunable_transmon_setup(
            n_qubits, leader_instrument, readout_multiplex=readout_multiplex
        )
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
        qubits_per_port:
            Number of qubits that share a single port. Consecutive qubits
            are grouped onto the same port.

    Attributes:
        num_devices:
            Total number of devices needed.
        qubit_to_port:
            A map from qubit uid to a tuple of the device and port number.
    """

    def __init__(self, *, ports: int, qubits: list[str], qubits_per_port: int = 1):
        n_ports = len(qubits) // qubits_per_port + (
            1 if len(qubits) % qubits_per_port > 0 else 0
        )

        self.num_devices = n_ports // ports + (1 if n_ports % ports > 0 else 0)
        self.qubit_to_port = {
            qubit_id: ((i // qubits_per_port) // ports, (i // qubits_per_port) % ports)
            for i, qubit_id in enumerate(qubits)
        }


# A single SHFQA QA channel can multiplex readout of up to 16 qubits
MAX_MULTIPLEXED_QUBITS_PER_QA_CHANNEL = 16


def _create_leader_instrument(
    leader_instrument_type: Literal["PQSC", "QHub"],
) -> PQSC | QHUB:
    if leader_instrument_type == "PQSC":
        return PQSC(uid="device_pqsc", address="dev125")

    if leader_instrument_type == "QHub":
        return QHUB(uid="device_qhub", address="dev125")

    raise ValueError(
        f"Invalid choice for leader instrument {leader_instrument_type}, "
        "expected one of PQSC, QHub"
    )


def tunable_transmon_setup(
    n_qubits: int,
    leader_instrument: Literal["PQSC", "QHub"] = "PQSC",
    *,
    readout_multiplex: bool = True,
) -> DeviceSetup:
    """Return a demo tunable transmon device setup.

    The returned setup consists of:

    - 1 leader instrument (PQSC or QHub)
    - N SHFQC (for the qubit drive lines)
    - M HDAWG (for the qubit flux lines)
    - K SHFQA (for the qubit measure lines)

    N, M and K are determined automatically from the number of qubits.

    The following device options are used:

    - PQSC (default)
    - SHFQC/QC6CH
    - HDAWG8/MF/ME/SKW/PC

    If `readout_multiplex` is `True` (default behavior), groups of up to 16
    qubits share a multiplexed readout line. Otherwise the measure and acquire
    lines are distributed over separate channels, one per qubit.

    Arguments:
        n_qubits:
            Number of qubits to include in the QPU.
        leader_instrument:
            Overrides the default PQSC device for the setup.
        readout_multiplex:
            Whether to multiplex the measure and acquire lines.

    Returns:
        The device setup.

    !!! version-changed "Changed in version 26.4.0"
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
    shfqa_measure_lines = _DeviceAndPortSet(
        ports=4,
        qubits=qubit_ids,
        qubits_per_port=(
            MAX_MULTIPLEXED_QUBITS_PER_QA_CHANNEL if readout_multiplex else 1
        ),
    )

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

    for i in range(shfqa_measure_lines.num_devices):
        setup.add_instruments(
            SHFQA(
                uid=f"device_shfqa_{i}",
                address=f"dev135{i}",
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
        # Groups of qubits share a QA channel if `readout_multiplex` is True
        # (the default); otherwise every qubit gets its own QA channel.
        shfqa, shfqa_port = shfqa_measure_lines.qubit_to_port[qubit]

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
        )
        setup.add_connections(
            f"device_shfqa_{shfqa}",
            create_connection(
                to_signal=f"{qubit}/measure",
                ports=f"QACHANNELS/{shfqa_port}/OUTPUT",
            ),
            create_connection(
                to_signal=f"{qubit}/acquire",
                ports=f"QACHANNELS/{shfqa_port}/INPUT",
            ),
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


def tunable_transmon_setup_zqcs(
    n_qubits: int,
    *,
    readout_multiplex: bool = True,
) -> DeviceSetup:
    """Return a tunable transmon device setup for ZQCS.

    The setup carries a setup description of its node cards, as a system would
    report it. Requires the `zhinst_zqcs` package.

    Arguments:
        n_qubits: The number of qubits.
        readout_multiplex: Whether to multiplex the measure and acquire lines.

    Returns:
        The device setup.
    """
    if n_qubits < 1:
        raise ValueError(
            "This testing and demonstration setup requires at least one qubit.",
        )

    setup = DeviceSetup("zqcs_setup")
    setup.add_instruments(ZQCS(uid="zqcs", address="http://localhost/shelfdemo"))

    delays = {
        "drive": _GLOBAL_DELAY,
        "drive_ef": _GLOBAL_DELAY,
        "measure": _GLOBAL_DELAY + _QA_START_DELAY,
        "acquire": _GLOBAL_DELAY + _QA_START_DELAY + _RO_INT_DELAY,
        "flux": _GLOBAL_DELAY,
    }
    for i in range(n_qubits):
        qubit = f"q{i}"
        drive_port = f"1:2:3:{i + 1}"
        qa_slot = 2 if readout_multiplex else i + 2
        setup.add_connections(
            "zqcs",
            create_connection(to_signal=f"{qubit}/drive", ports=drive_port, type="iq"),
            create_connection(
                to_signal=f"{qubit}/drive_ef", ports=drive_port, type="iq"
            ),
            create_connection(
                to_signal=f"{qubit}/measure", ports=f"1:{qa_slot}:4:1", type="iq"
            ),
            create_connection(
                to_signal=f"{qubit}/acquire", ports=f"1:{qa_slot}:4:3", type="acquire"
            ),
            create_connection(
                to_signal=f"{qubit}/flux", ports=f"1:2:1:{i + 1}", type="rf"
            ),
        )
        for line, delay in delays.items():
            setup.logical_signal_by_uid(
                f"{qubit}/{line}"
            ).calibration = SignalCalibration(delay_signal=delay)

    setup.setup_description = _setup_description_zqcs(
        n_qubits, readout_multiplex=readout_multiplex
    )
    return setup


def _setup_description_zqcs(
    n_qubits: int, *, readout_multiplex: bool
) -> "SetupDescriptionZQCS":
    """Describes the node cards that the demo layout occupies."""
    from zhinst_zqcs.emulation.setup_description import (
        Frontend,
        NodeCard,
        emulated_setup_description,
    )

    cards = [
        NodeCard(
            slot=2,
            frontends={
                1: Frontend.FLUX,
                2: Frontend.FLUX,
                3: Frontend.SG,
                4: Frontend.QA,
            },
        )
    ]
    if not readout_multiplex:
        cards += [
            NodeCard(slot=slot, frontends={4: Frontend.QA})
            for slot in range(3, n_qubits + 2)
        ]
    return emulated_setup_description(cards)


def tunable_transmon_qubits(
    n_qubits: int,
    setup: DeviceSetup,
) -> list[TunableTransmonQubit]:
    """Return demo tunable transmon device qubits.

    The qubits are constructed for the device setup returned by
    [tunable_transmon_setup][].

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
            is the setup returned by [tunable_transmon_setup][]
            called with the same number of qubits.

    Returns:
        The list of qubits.
    """

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
                drive_lo_frequency=int(_q_param(i // 20, 6.4, 1e9, dq=0.2, max_=7.2e9)),
                resonance_frequency_ge=_q_param(i, 6.5, 1e9, dq=0.01, max_=7.0e9),
                resonance_frequency_ef=_q_param(i, 6.3, 1e9, dq=0.01, max_=7.0e9),
                readout_lo_frequency=7e9,
                readout_resonator_frequency=_q_param(i, 7.1, 1e9, dq=0.01, max_=7.9e9),
                ge_drive_amplitude_pi=_q_param(i, 0.8, max_=1.0),
                ge_drive_amplitude_pi2=_q_param(i, 0.4, max_=1.0),
                ge_drive_length=51e-9,
                ge_drive_pulse={
                    "function": "drag",
                    "beta": 0.01,
                    "sigma": 0.21,
                },
                ef_drive_amplitude_pi=_q_param(i, 0.7, max_=1.0),
                ef_drive_amplitude_pi2=_q_param(i, 0.3, max_=1.0),
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


def tunable_transmon_qubits_zqcs(setup: DeviceSetup) -> list[TunableTransmonQubit]:
    """Return demo tunable transmon device qubits for ZQCS.

    Arguments:
        setup: The device setup.

    Returns:
        The list of qubits.
    """

    qubits = []
    for i, (q_uid, signal_group) in enumerate(setup.logical_signal_groups.items()):
        q = TunableTransmonQubit.from_logical_signal_group(
            q_uid,
            signal_group,
            parameters=TunableTransmonQubitParameters(
                drive_lo_frequency=int(_q_param(i // 20, 6.4, 1e9, dq=0.2, max_=8.4e9)),
                resonance_frequency_ge=_q_param(i, 6.5, 1e9),
                resonance_frequency_ef=_q_param(i, 6.3, 1e9),
                readout_lo_frequency=7e9,
                readout_resonator_frequency=_q_param(i, 7.1, 1e9),
                ge_drive_amplitude_pi=_q_param(i, 0.8, max_=1.0),
                ge_drive_amplitude_pi2=_q_param(i, 0.4, max_=1.0),
                ge_drive_length=51e-9,
                ge_drive_pulse={
                    "function": "drag",
                    "beta": 0.01,
                    "sigma": 0.21,
                },
                ef_drive_amplitude_pi=_q_param(i, 0.7, max_=1.0),
                ef_drive_amplitude_pi2=_q_param(i, 0.3, max_=1.0),
                ef_drive_length=52e-9,
                ef_drive_pulse={
                    "function": "drag",
                    "beta": 0.01,
                    "sigma": 0.21,
                },
                spectroscopy_amplitude=0.0005,
                readout_amplitude=0.2,
                readout_integration_discrimination_thresholds=[0.0],
            ),
        )
        qubits.append(q)

    return qubits


def _q_param(
    i: int, base: float, unit: float = 1.0, dq: float = 0.01, max_: float = np.inf
) -> float:
    """Tweak qubit parameter a tiny amount to distinguish them."""
    return min((base + i * dq) * unit, max_)
