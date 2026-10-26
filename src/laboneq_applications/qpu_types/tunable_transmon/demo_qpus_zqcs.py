# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""ZQCS tunable transmon qubit device setups for testing and demonstration."""

import numpy as np
from laboneq.dsl.calibration import SignalCalibration
from laboneq.dsl.device import DeviceSetup, create_connection
from laboneq.dsl.device.instruments import ZQCS
from laboneq.dsl.quantum import QPU, QuantumPlatform

from .operations import TunableTransmonOperations
from .qubit_types import (
    TunableTransmonQubit,
    TunableTransmonQubitParameters,
)

# Hardware constants
_GLOBAL_DELAY = 5e-6
_QA_START_DELAY = 40e-9
_RO_INT_DELAY = 456.0e-9


def demo_platform_zqcs(
    n_qubits: int, *, readout_multiplex: bool = True
) -> QuantumPlatform:
    """The demo quantum platform for ZQCS.

    Arguments:
        n_qubits: The number of qubits.
        readout_multiplex: Whether to multiplex the measure and acquire lines.

    Returns:
        The quantum platform.
    """
    setup = tunable_transmon_setup_zqcs(n_qubits, readout_multiplex=readout_multiplex)
    qubits = tunable_transmon_qubits_zqcs(setup)
    quantum_operations = TunableTransmonOperations()
    qpu = QPU(qubits, quantum_operations=quantum_operations)
    return QuantumPlatform(setup, qpu)


def tunable_transmon_setup_zqcs(
    n_qubits: int,
    *,
    readout_multiplex: bool = True,
) -> DeviceSetup:
    """The tunable transmon device setup for ZQCS.

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

    id_mapping = {
        f"q{i}": {
            "geo_sg": f"1:2:3:{i + 1}",
            "geo_sg_ef": f"1:2:3:{i + 1}",
            "geo_qa_out": "1:2:4:1" if readout_multiplex else f"1:{i + 2}:4:1",
            "geo_qa_in": "1:2:4:3" if readout_multiplex else f"1:{i + 2}:4:3",
            "geo_lf": f"1:2:1:{i + 1}",
        }
        for i in range(n_qubits)
    }

    for qubit, ports in id_mapping.items():
        setup.add_connections(
            "zqcs",
            create_connection(
                to_signal=f"{qubit}/drive", ports=ports["geo_sg"], type="iq"
            ),
            create_connection(
                to_signal=f"{qubit}/drive_ef", ports=ports["geo_sg_ef"], type="iq"
            ),
            create_connection(
                to_signal=f"{qubit}/measure", ports=ports["geo_qa_out"], type="iq"
            ),
            create_connection(
                to_signal=f"{qubit}/acquire", ports=ports["geo_qa_in"], type="acquire"
            ),
            create_connection(
                to_signal=f"{qubit}/flux", ports=ports["geo_lf"], type="rf"
            ),
        )

    for qubit in id_mapping:
        setup.logical_signal_by_uid(f"{qubit}/drive").calibration = SignalCalibration(
            delay_signal=_GLOBAL_DELAY,
        )
        setup.logical_signal_by_uid(
            f"{qubit}/drive_ef"
        ).calibration = SignalCalibration(
            delay_signal=_GLOBAL_DELAY,
        )
        setup.logical_signal_by_uid(f"{qubit}/measure").calibration = SignalCalibration(
            delay_signal=_GLOBAL_DELAY + _QA_START_DELAY,
        )
        setup.logical_signal_by_uid(f"{qubit}/acquire").calibration = SignalCalibration(
            delay_signal=_GLOBAL_DELAY + _QA_START_DELAY + _RO_INT_DELAY,
        )
        setup.logical_signal_by_uid(f"{qubit}/flux").calibration = SignalCalibration(
            delay_signal=_GLOBAL_DELAY,
        )

    return setup


def tunable_transmon_qubits_zqcs(setup: DeviceSetup) -> list[TunableTransmonQubit]:
    """The tunable transmon qubits for ZQCS.

    Arguments:
        setup: The device setup.

    Returns:
        The tunable transmon qubits.
    """

    def q_param(
        i: int, base: float, unit: float = 1.0, dq: float = 0.01, max_: float = np.inf
    ) -> float:
        """Tweak qubit parameter a tiny amount to distinguish them."""
        return min((base + i * dq) * unit, max_)

    qubits = []
    for i, (q_uid, signal_group) in enumerate(setup.logical_signal_groups.items()):
        q = TunableTransmonQubit.from_logical_signal_group(
            q_uid,
            signal_group,
            parameters=TunableTransmonQubitParameters(
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
                spectroscopy_amplitude=0.0005,
                readout_amplitude=0.2,
            ),
        )
        qubits.append(q)

    return qubits
