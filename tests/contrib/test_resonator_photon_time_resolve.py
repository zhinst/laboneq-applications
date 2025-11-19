# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Smoke-tests for the resonator_photons_time_resolved experiments."""

import numpy as np

from laboneq_applications.contrib.experiments import resonator_photons_time_resolved


class TestResonatorPhotonsTimeResolved:
    def test_resonator_photons_time_resolved(self, two_tunable_transmon_platform):
        platform = two_tunable_transmon_platform
        qpu = platform.qpu
        qubits = platform.qpu.quantum_elements
        options = resonator_photons_time_resolved.experiment_workflow.options()
        options.do_analysis(True)
        session = platform.session(do_emulation=True)
        frequencies = [
            q.parameters.resonance_frequency_ge + np.linspace(-100e6, 100e6, 11)
            for q in qubits
        ]
        times = [np.linspace(0, 2e-6, 21) for q in qubits]
        wf = resonator_photons_time_resolved.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubits,
            times=times,
            frequencies=frequencies,
            options=options,
        )
        wf.run()
