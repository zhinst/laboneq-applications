"""Device setups for tests."""

import pytest
from laboneq.dsl.calibration import Oscillator, SignalCalibration
from laboneq.dsl.device import DeviceSetup, create_connection
from laboneq.dsl.device.instruments import HDAWG, PQSC, SHFQC

from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonQubit,
    TunableTransmonQubitParameters,
)


class DeviceWithQubits:
    """A DeviceSetup and its qubits."""

    def __init__(self, device_setup, qubits):
        self.setup = device_setup
        self.qubits = qubits


@pytest.fixture()
def single_tunable_transmon():
    """Return a single tunable transmon device setup and its qubits."""
    setup = single_tunable_transmon_setup()
    qubits = single_tunable_transmon_qubits(setup)
    return DeviceWithQubits(setup, qubits)


@pytest.fixture()
def two_tunable_transmon():
    """Return a single tunable transmon device setup and its qubits."""
    setup = two_tunable_transmon_setup()
    qubits = two_tunable_transmon_qubits(setup)
    return DeviceWithQubits(setup, qubits)


def single_tunable_transmon_setup():
    """Return a single tunable transmon device setup."""
    setup = DeviceSetup("test")
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
        "device_shfqc",
        create_connection(to_signal="q0/drive", ports="SGCHANNELS/0/OUTPUT"),
        create_connection(to_signal="q0/drive_ef", ports="SGCHANNELS/0/OUTPUT"),
        create_connection(to_signal="q0/measure", ports="QACHANNELS/0/OUTPUT"),
        create_connection(to_signal="q0/acquire", ports="QACHANNELS/0/INPUT"),
    )
    setup.add_connections(
        "device_hdawg",
        create_connection(to_signal="q0/flux", ports="SIGOUTS/0"),
    )
    setup.add_connections(
        "device_pqsc",
        create_connection(to_instrument="device_shfqc", ports="ZSYNCS/0"),
        create_connection(to_instrument="device_hdawg", ports="ZSYNCS/1"),
    )

    for qubit in ["q0"]:
        for line in ["drive", "measure", "acquire"]:
            logical_signal = setup.logical_signal_by_uid(f"{qubit}/{line}")
            logical_signal.calibration = SignalCalibration(
                local_oscillator=Oscillator(frequency=5e9),
            )
        drive_lsg = setup.logical_signal_by_uid(f"{qubit}/drive")
        drive_ef_lsg = setup.logical_signal_by_uid(f"{qubit}/drive_ef")
        drive_ef_lsg.calibration = drive_lsg.calibration

    return setup


def two_tunable_transmon_setup():
    """Return a single tunable transmon device setup."""
    setup = DeviceSetup("two_qubits")
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
        "device_shfqc",
        create_connection(to_signal="q0/drive", ports="SGCHANNELS/0/OUTPUT"),
        create_connection(to_signal="q0/drive_ef", ports="SGCHANNELS/0/OUTPUT"),
        create_connection(to_signal="q0/measure", ports="QACHANNELS/0/OUTPUT"),
        create_connection(to_signal="q0/acquire", ports="QACHANNELS/0/INPUT"),
        create_connection(to_signal="q1/drive", ports="SGCHANNELS/1/OUTPUT"),
        create_connection(to_signal="q1/drive_ef", ports="SGCHANNELS/1/OUTPUT"),
        create_connection(to_signal="q1/measure", ports="QACHANNELS/0/OUTPUT"),
        create_connection(to_signal="q1/acquire", ports="QACHANNELS/0/INPUT"),
    )
    setup.add_connections(
        "device_hdawg",
        create_connection(to_signal="q0/flux", ports="SIGOUTS/0"),
        create_connection(to_signal="q1/flux", ports="SIGOUTS/1"),
    )
    setup.add_connections(
        "device_pqsc",
        create_connection(to_instrument="device_shfqc", ports="ZSYNCS/0"),
        create_connection(to_instrument="device_hdawg", ports="ZSYNCS/1"),
    )

    for qubit in ["q0", "q1"]:
        for line in ["drive", "measure", "acquire"]:
            logical_signal = setup.logical_signal_by_uid(f"{qubit}/{line}")
            logical_signal.calibration = SignalCalibration(
                local_oscillator=Oscillator(frequency=5e9),
            )
        drive_lsg = setup.logical_signal_by_uid(f"{qubit}/drive")
        drive_ef_lsg = setup.logical_signal_by_uid(f"{qubit}/drive_ef")
        drive_ef_lsg.calibration = drive_lsg.calibration

    return setup


def single_tunable_transmon_qubits(setup):
    """Return qubit q0."""
    q0 = TunableTransmonQubit.from_logical_signal_group(
        "q0",
        setup.logical_signal_groups["q0"],
        parameters=TunableTransmonQubitParameters(
            drive_parameters_ge={
                "amplitude_pi": 0.4,
                "amplitude_pi2": 0.8,
                "length": 51e-9,
                "pulse": {
                    "function": "drag",
                    "beta": 0.01,
                    "sigma": 0.21,
                },
            },
            drive_parameters_ef={
                "amplitude_pi": 0.35,
                "amplitude_pi2": 0.7,
                "length": 52e-9,
                "pulse": {
                    "function": "drag",
                    "beta": 0.01,
                    "sigma": 0.21,
                },
            },
        ),
    )
    return [q0]


def two_tunable_transmon_qubits(setup):
    """Return qubits q0 and q1."""
    q0 = TunableTransmonQubit.from_logical_signal_group(
        "q0",
        setup.logical_signal_groups["q0"],
        parameters=TunableTransmonQubitParameters(
            drive_parameters_ge={
                "amplitude_pi": 0.4,
                "amplitude_pi2": 0.8,
                "length": 51e-9,
                "pulse": {
                    "function": "drag",
                    "beta": 0.01,
                    "sigma": 0.21,
                },
            },
            drive_parameters_ef={
                "amplitude_pi": 0.35,
                "amplitude_pi2": 0.7,
                "length": 52e-9,
                "pulse": {
                    "function": "drag",
                    "beta": 0.01,
                    "sigma": 0.21,
                },
            },
        ),
    )
    q1 = TunableTransmonQubit.from_logical_signal_group(
        "q1",
        setup.logical_signal_groups["q1"],
        parameters=TunableTransmonQubitParameters(
            drive_parameters_ge={
                "amplitude_pi": 0.5,
                "amplitude_pi2": 0.25,
                "length": 51e-9,
                "pulse": {
                    "function": "drag",
                    "beta": 0.01,
                    "sigma": 0.21,
                },
            },
            drive_parameters_ef={
                "amplitude_pi": 0.4,
                "amplitude_pi2": 0.2,
                "length": 52e-9,
                "pulse": {
                    "function": "drag",
                    "beta": 0.01,
                    "sigma": 0.21,
                },
            },
        ),
    )
    return [q0, q1]
