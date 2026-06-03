# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for laboneq_applications.qpu_types.tunable_transmon.device_setups."""

import re

import numpy as np
import pytest
from laboneq.dsl.session import Session

from laboneq_applications.experiments import qubit_spectroscopy
from laboneq_applications.qpu_types.tunable_transmon import demo_platform


def port_for_qubit_signal(qp, q: str, signal: str) -> str:
    """Return the physical port name for the given qubit signal."""
    logical_signal_group = f"/logical_signal_groups/{q}/{signal}"
    results = [
        connection
        for instrument in qp.setup.instruments
        for connection in instrument.connections
        if connection.remote_path == logical_signal_group
    ]
    if len(results) == 0:
        raise RuntimeError(
            f"Could not find a port that matches {logical_signal_group!r}"
        )
    if len(results) > 1:
        raise RuntimeError(
            f"Found multiple ports that match {logical_signal_group!r}: {results!r}"
        )
    return results[0].local_port


class TestDemoPlatform:
    @pytest.mark.parametrize(
        ("num_qubits", "num_shfqcs", "num_hdawgs"),
        [
            pytest.param(1, 1, 1, id="1"),
            pytest.param(2, 1, 1, id="2"),
            pytest.param(6, 1, 1, id="6"),
            pytest.param(13, 3, 2, id="13"),
            pytest.param(1000, 167, 125, id="1000"),
        ],
    )
    def test_transmon_qpu(self, num_qubits, num_shfqcs, num_hdawgs):
        qp = demo_platform(num_qubits)
        qubits = qp.qpu.quantum_elements

        # check instruments:
        assert qp.setup.uid == f"tunable_transmons_{num_qubits}"
        assert qp.setup.qubits == {}
        assert [inst.uid for inst in qp.setup.instruments] == [
            *[f"device_shfqc_{i}" for i in range(num_shfqcs)],
            *[f"device_hdawg_{i}" for i in range(num_hdawgs)],
            "device_pqsc",
        ]

        # check quantum elements and logical signal groups
        assert list(qp.setup.logical_signal_groups) == [
            f"q{i}" for i in range(num_qubits)
        ]
        assert [q.uid for q in qubits] == [f"q{i}" for i in range(num_qubits)]
        drive_lo_frequencies = [q.parameters.drive_lo_frequency for q in qubits]

        base_lo_frequency = int(6.4e9)
        max_lo_frequency = int(7.2e9)
        lo_frequency_grid = int(2e8)
        group_size = 20
        distinguishable_qubits = (
            (max_lo_frequency - base_lo_frequency) // lo_frequency_grid * group_size
        )

        assert drive_lo_frequencies[:distinguishable_qubits] == [
            int((6.4 + (i // group_size) * 0.2) * 1e9)
            for i in range(min(distinguishable_qubits, num_qubits))
        ]

        assert drive_lo_frequencies[distinguishable_qubits:] == [max_lo_frequency] * (
            max(0, num_qubits - distinguishable_qubits)
        )

        assert max(drive_lo_frequencies) <= max_lo_frequency

        # check connections
        for q in qubits:
            drive = port_for_qubit_signal(qp, q.uid, "drive")
            drive_ef = port_for_qubit_signal(qp, q.uid, "drive_ef")
            assert drive == drive_ef
            assert re.match(r"SGCHANNELS/\d+/OUTPUT", drive)
            assert re.match(
                r"QACHANNELS/\d+/OUTPUT",
                port_for_qubit_signal(qp, q.uid, "measure"),
            )
            assert re.match(
                r"QACHANNELS/\d+/INPUT",
                port_for_qubit_signal(qp, q.uid, "acquire"),
            )

    def test_too_few_qubits(self):
        with pytest.raises(ValueError) as err:
            demo_platform(0)
        assert (
            str(err.value)
            == "This testing and demonstration setup requires at least one qubit."
        )

    def test_invalid_instrument_choice(self):
        with pytest.raises(ValueError, match="Invalid choice for leader instrument"):
            demo_platform(1, "invalid")

    @pytest.mark.parametrize("num_qubits", [1, 10, 100, 192])
    def test_many_qubits_compile(self, num_qubits):
        platform = demo_platform(num_qubits, leader_instrument="QHub")
        setup = platform.setup
        session = Session(setup)
        session.connect(do_emulation=True)
        qpu = platform.qpu
        qubit_uids = qpu.quantum_element_uids

        # A smoke test that the demo platform can be used to compile an experiment
        qubit_spectroscopy.experiment_workflow(
            session=session,
            qubits=qubit_uids,
            qpu=qpu,
            frequencies=[np.linspace(6.5e9, 7.0e9, 11) for _ in qubit_uids],
        ).run(until="run_experiment")
