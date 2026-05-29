# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Smoke-tests for the memory spectroscopy experiment."""

import numpy as np

from laboneq_applications.contrib.experiments.bosonic_qubits import (
    memory_spectroscopy,
)


class TestMemorySpectroscopy:
    def test_memory_spectroscopy(self, single_bosonic_qubit_platform):
        platform = single_bosonic_qubit_platform
        qpu = platform.qpu
        qubits = qpu.quantum_elements
        qubit_uids = qpu.quantum_element_uids

        session = platform.session(do_emulation=True)

        frequencies = [
            q.parameters.memory_resonance_frequency + np.linspace(-5e6, 5e6, 5)
            for q in qubits
        ]

        wf = memory_spectroscopy.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubit_uids,
            frequencies=frequencies,
        )
        wf.run()
