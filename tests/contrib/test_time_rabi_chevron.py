# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Smoke-tests for the test_time_rabi_chevron experiments."""

import numpy as np

from laboneq_applications.contrib.experiments import time_rabi_chevron


class TestTimeRabiChevron:
    def test_time_rabi_chevron(self, two_tunable_transmon_platform):
        platform = two_tunable_transmon_platform
        qpu = platform.qpu
        qubits = platform.qpu.quantum_elements
        options = time_rabi_chevron.experiment_workflow.options()
        options.do_analysis(True)
        session = platform.session(do_emulation=True)
        frequencies = [
            q.parameters.resonance_frequency_ge + np.linspace(-100e6, 100e6, 11)
            for q in qubits
        ]
        wf = time_rabi_chevron.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubits,
            lengths=[
                np.arange(0.1e-6, 1.05e-6, 0.1e-6),
                np.arange(0.1e-6, 1.05e-6, 0.1e-6),
            ],
            frequencies=frequencies,
            options=options,
        )
        wf.run()
