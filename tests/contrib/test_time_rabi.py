# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Smoke-tests for the time_rabi experiments."""

import numpy as np

from laboneq_applications.contrib.experiments import time_rabi


class TestTimeRabi:
    def test_time_rabi(self, two_tunable_transmon_platform):
        platform = two_tunable_transmon_platform
        qpu = platform.qpu
        qubits = platform.qpu.quantum_elements
        options = time_rabi.experiment_workflow.options()
        options.do_analysis(True)
        session = platform.session(do_emulation=True)
        wf = time_rabi.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubits,
            lengths=[
                np.arange(0.1e-6, 1.05e-6, 0.1e-6),
                np.arange(0.1e-6, 1.05e-6, 0.1e-6),
            ],
            options=options,
        )
        wf.run()
