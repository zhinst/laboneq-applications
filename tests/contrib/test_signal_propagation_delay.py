# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Smoke-tests for the signal_propagation_delay experiments."""

import numpy as np

from laboneq_applications.contrib.experiments import signal_propagation_delay


class TestSignalPropagationDelay:
    def test_signal_propagation_delay(self, single_tunable_transmon_platform):
        platform = single_tunable_transmon_platform
        qpu = platform.qpu
        qubits = platform.qpu.quantum_elements
        options = signal_propagation_delay.experiment_workflow.options()
        options.do_analysis(True)
        session = platform.session(do_emulation=True)
        wf = signal_propagation_delay.experiment_workflow(
            session=session,
            qpu=qpu,
            qubit=qubits[0],
            delays=np.linspace(0, 1e-6, 21),
            options=options,
        )
        wf.run()
