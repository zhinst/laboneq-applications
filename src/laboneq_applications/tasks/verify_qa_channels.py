# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module provides a task to verify that qubits use unique QA channels.

Some experiments sweep a setting of the readout line, such as the frequency of
its hardware oscillator (e.g., resonator spectroscopy) or its amplitude (e.g.,
AC Stark shift vs power). Qubits that are multiplexed on the same QA channel
share these settings, so they cannot be swept independently and in parallel.
The task in this module guards experiment workflows against being run on such
qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.dsl.quantum.quantum_element import QuantumElement

from laboneq_applications.conversion import as_quantum_element_list

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.dsl.device import DeviceSetup
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import QuantumElements


def _map_qubits_to_measure_physical_channels(
    device_setup: DeviceSetup,
    qubits: QuantumElement | Sequence[QuantumElement],
) -> dict[str, list[str]]:
    """Map the physical channels of the measure lines to the qubits using them.

    Args:
        device_setup: The device setup on which the qubits are defined.
        qubits: Either a single `QuantumElement` or a list of `QuantumElement`.

    Returns:
        A dictionary with the physical-channel UIDs of the measure lines as keys
            and the list of UIDs of the qubits whose measure lines use that physical
            channel as values.
    """
    if isinstance(qubits, QuantumElement):
        qubits = [qubits]
    channel_map: dict[str, list[str]] = {}
    for q in qubits:
        logical_signal = device_setup.logical_signal_by_uid(q.signals["measure"])
        physical_channel_uid = logical_signal.physical_channel.uid
        channel_map.setdefault(physical_channel_uid, []).append(q.uid)
    return channel_map


@workflow.task(save=False)
def _verify_unique_qa_channels(
    session: Session,
    qubits: QuantumElements,
) -> None:
    """Verify that the measure lines of the qubits use unique QA channels.

    Arguments:
        session:
            The connected session, whose device setup is used to resolve the
            physical channels of the measure lines of the qubits.
        qubits:
            The qubits to check. May be either a single qubit or a list of
            qubits. For a single qubit, this task is a no-op.

    Raises:
        ValueError:
            If two or more of the qubits have measure lines on the same
            physical QA channel.
    """
    quantum_elements = as_quantum_element_list(qubits)
    if len(quantum_elements) <= 1:
        return

    channel_map = _map_qubits_to_measure_physical_channels(
        session.device_setup, quantum_elements
    )
    shared_channels = {
        channel_uid: qubit_uids
        for channel_uid, qubit_uids in channel_map.items()
        if len(qubit_uids) > 1
    }
    if shared_channels:
        shared_channels_str = "; ".join(
            f"physical channel '{channel_uid}' is shared by qubits "
            f"{', '.join(qubit_uids)}"
            for channel_uid, qubit_uids in shared_channels.items()
        )
        raise ValueError(
            f"The following qubits share the same QA channel: "
            f"{shared_channels_str}. This experiment sweeps a setting of the "
            f"readout line, such as the frequency of its hardware oscillator or "
            f"its amplitude, which is shared by all qubits multiplexed on the "
            f"same QA channel. Therefore, this setting cannot be swept "
            f"independently for these qubits in parallel. Please run the "
            f"experiment on these qubits in separate workflow executions."
        )
