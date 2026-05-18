# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Smoke-tests for the spin_locking experiments."""

import numpy as np

from laboneq_applications.contrib.experiments import spin_locking


class TestSpinLocking:
    def test_spin_locking(self, two_tunable_transmon_platform):
        platform = two_tunable_transmon_platform
        qpu = platform.qpu
        qubit_uids = qpu.quantum_element_uids
        options = spin_locking.experiment_workflow.options()
        session = platform.session(do_emulation=True)
        wf = spin_locking.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubit_uids,
            lengths=[
                np.arange(0.1e-6, 1.05e-6, 0.1e-6),
                np.arange(0.1e-6, 1.05e-6, 0.1e-6),
            ],
            options=options,
        )
        wf.run()
