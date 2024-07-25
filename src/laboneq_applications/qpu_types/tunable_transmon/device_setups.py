"""Tunable transmon qubit device setups for testing and demonstration."""

from laboneq.dsl.calibration import Oscillator, SignalCalibration
from laboneq.dsl.device import DeviceSetup, create_connection
from laboneq.dsl.device.instruments import HDAWG, PQSC, SHFQC
from laboneq.dsl.enums import ModulationType
from laboneq.dsl.session import Session

from .qubit_types import TunableTransmonQubit, TunableTransmonQubitParameters


class DeviceWithQubits:
    """A DeviceSetup and its qubits."""

    def __init__(self, device_setup: DeviceSetup, qubits: list[TunableTransmonQubit]):
        self.setup = device_setup
        self.qubits = qubits

    def session(self) -> Session:
        """Return an emulated session for the setup."""
        session = Session(self.setup)
        session.connect(do_emulation=True)
        return session


def demo_qpu(n_qubits: int) -> DeviceWithQubits:
    """Return a demo tunable transmon QPU with the specified number of qubits.

    The returned setup consists of:

    - 1 PQSC
    - 1 SHFQC (for the qubit drive and measurement lines)
    - 1 HDAWG (for the qubit flux lines)

    The maximum number of qubits is 8 (which is the number of drive lines on
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
    return DeviceWithQubits(setup, qubits)


def tunable_transmon_setup(n_qubits: int) -> DeviceSetup:
    """Return a demo tunable transmon device setup.

    The returned setup consists of:

    - 1 PQSC
    - 1 SHFQC (for the qubit drive and measurement lines)
    - 1 HDAWG (for the qubit flux lines)

    The maximum number of qubits is 8 (which is the number of drive lines on
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
    SHFQC_DRIVE_LINES: int = 8  # noqa: N806
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
    setup.add_instruments(PQSC(uid="device_pqsc", address="dev125"))

    setup.add_connections(
        "device_pqsc",
        create_connection(to_instrument="device_shfqc", ports="ZSYNCS/0"),
        create_connection(to_instrument="device_hdawg", ports="ZSYNCS/1"),
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

    def q_param(i: int, base: float, unit: float = 1.0, dq: float = 0.01) -> float:
        """Tweak qubit parameter a tiny amount to distinguish them."""
        return (base + i * dq) * unit

    qubits = []
    for i in range(n_qubits):
        q = TunableTransmonQubit.from_logical_signal_group(
            f"q{i}",
            setup.logical_signal_groups[f"q{i}"],
            parameters=TunableTransmonQubitParameters(
                drive_lo_frequency=q_param(i, 1.5, 1e9, dq=0.1),
                resonance_frequency_ge=q_param(i, 1.6, 1e9),
                resonance_frequency_ef=q_param(i, 1.7, 1e9),
                readout_lo_frequency=2e9,
                readout_resonator_frequency=q_param(i, 2.1, 1e9),
                drive_parameters_ge={
                    "amplitude_pi": q_param(i, 0.8),
                    "amplitude_pi2": q_param(i, 0.4),
                    "length": 51e-9,
                    "pulse": {
                        "function": "drag",
                        "beta": 0.01,
                        "sigma": 0.21,
                    },
                },
                drive_parameters_ef={
                    "amplitude_pi": q_param(i, 0.7),
                    "amplitude_pi2": q_param(i, 0.3),
                    "length": 52e-9,
                    "pulse": {
                        "function": "drag",
                        "beta": 0.01,
                        "sigma": 0.21,
                    },
                },
            ),
        )
        qubits.append(q)

    return qubits
