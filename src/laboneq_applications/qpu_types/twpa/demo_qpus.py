# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tunable transmon qubit device setups for testing and demonstration."""

from laboneq.dsl.calibration import Oscillator, SignalCalibration
from laboneq.dsl.device import DeviceSetup, create_connection
from laboneq.dsl.device.instruments import SHFPPC, SHFQC
from laboneq.dsl.enums import ModulationType
from laboneq.dsl.quantum.qpu import QPU, QuantumPlatform

from .operations import TWPAOperations
from .twpa_types import (
    TWPA,
    TWPAParameters,
)


def demo_platform(n_twpas: int) -> QuantumPlatform:
    """Return a demo TWPA QPU with the specified number of TWPAS.

    The returned setup consists of:

    - 1 SHFPPC
    - 1 SHFQA/SHFQC


    The maximum number of ParamAmplifiers is 4 (which is the number of measure lines on
    the SHFQA).

    The TWPAs share a single multiplexed readout line.

    Arguments:
        n_twpas:
            Number of TWPAS to include in the QPU.

    Returns:
        The QPU.
    """
    setup = twpa_setup(n_twpas)
    twpas = travelling_wave_parameteric_amplifiers(n_twpas, setup)
    quantum_operations = TWPAOperations()
    qpu = QPU(twpas, quantum_operations=quantum_operations)
    return QuantumPlatform(setup=setup, qpu=qpu)


def twpa_setup(n_twpas: int) -> DeviceSetup:
    """Return a demo TWPA device setup.

    The returned setup consists of:

    - 1 SHFPPC
    - 1 SHFQA/SHFQC


    The maximum number of ParamAmplifiers is 4 (which is the number of measure lines on
    the SHFQA).

    The TWPAs share a single multiplexed readout line.

    Arguments:
        n_twpas:
            Number of TWPAs to include in the QPU.

    Returns:
        The device setup.
    """
    if n_twpas < 1:
        raise ValueError(
            "This testing and demonstration setup requires at least one TWPA.",
        )
    SHFPPC_DRIVE_LINES: int = 4  # noqa: N806
    if n_twpas > SHFPPC_DRIVE_LINES:
        raise ValueError(
            "This testing and demonstration setup requires 4 or fewer TWPAs.",
        )

    twpa_ids = [f"twpa{i}" for i in range(n_twpas)]

    setup = DeviceSetup(f"TravellingWaveParametericAmplifiers_{n_twpas}")
    setup.add_dataserver(host="localhost", port="8004")

    setup.add_instruments(
        SHFQC(uid="device_shfqc", address="dev12388"),
    )
    setup.add_instruments(
        SHFPPC(uid="device_shfppc", address="dev16008"),
    )

    for twpa in twpa_ids:
        setup.add_connections(
            "device_shfqc",
            # all twpas multiplex on the measure and acquire lines:
            create_connection(
                to_signal=f"{twpa}/measure",
                ports="QACHANNELS/0/OUTPUT",
            ),
            create_connection(to_signal=f"{twpa}/acquire", ports="QACHANNELS/0/INPUT"),
        )

        setup.add_connections(
            "device_shfppc",
            # each qubit has its own flux line:
            create_connection(to_signal=f"{twpa}/acquire", ports=f"PPCHANNELS/{0}"),
        )

        for line, frequency, mod_type in [
            ("measure", 4e9, ModulationType.SOFTWARE),
        ]:
            logical_signal = setup.logical_signal_by_uid(f"{twpa}/{line}")
            oscillator = Oscillator(modulation_type=mod_type)
            logical_signal.calibration = SignalCalibration(
                local_oscillator=Oscillator(frequency=frequency),
                oscillator=oscillator,
            )
            if line == "measure":
                # acquire and measure lines must share the same oscillator
                acquire_signal = setup.logical_signal_by_uid(f"{twpa}/acquire")
                acquire_signal.calibration = SignalCalibration(
                    local_oscillator=Oscillator(frequency=frequency),
                    oscillator=oscillator,
                )

    return setup


def travelling_wave_parameteric_amplifiers(
    n_twpas: int,
    setup: DeviceSetup,
) -> list[TWPA]:
    """Return demo TWPAs.

    The TWPAs are constructed for the device setup returned by
    [twpa_setup]().

    The TWPAs share a single readout local oscillator frequency
    because they share a single multiplexed readout line in the
    device setup.

    Other TWPA parameters (e.g. `drive_lo_frequency`) are set to
    slightly different values to allow them to be distinguished in
    demonstrations and tests.

    Arguments:
        n_twpas:
            Number of TWPAs to include in the QPU.
        setup:
            The device setup. It is assumed that the setup
            is the setup returned by [twpa_setup]()
            called with the same number of TWPAs.

    Returns:
        The list of TWPAs.
    """

    def twpa_param(
        i: int,
        base: float,
        unit: float = 1.0,
        dq: float = 0.01,
    ) -> float:
        """Tweak TWPA parameter a tiny amount to distinguish them."""
        return (base + i * dq) * unit

    return TWPA.from_device_setup(
        device_setup=setup,
        parameters={
            f"twpa{i}": TWPAParameters(
                readout_lo_frequency=6e9,
                probe_frequency=twpa_param(i, 6.5, 1e9),
                pump_frequency=twpa_param(i, 4, 1e9),
                pump_power=10,
                probe_power=10,
                cancellation_phase=0,
                cancellation_attenuation=10,
            )
            for i in range(n_twpas)
        },
    )
