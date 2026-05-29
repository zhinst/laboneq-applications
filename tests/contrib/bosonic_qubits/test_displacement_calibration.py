# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Smoke-tests for the displacement calibration experiment."""

import numpy as np

from laboneq_applications.contrib.experiments.bosonic_qubits import (
    displacement_calibration,
)


class TestDisplacementCalibration:
    def test_displacement_calibration(self, single_bosonic_qubit_platform):
        platform = single_bosonic_qubit_platform
        qpu = platform.qpu
        qubits = qpu.quantum_elements
        qubit_uids = qpu.quantum_element_uids

        options = displacement_calibration.experiment_workflow.options()
        options.do_analysis(True)

        session = platform.session(do_emulation=True)

        amplitudes = [np.linspace(0.0, 0.5, 5)]
        frequencies = [
            q.parameters.transmon_resonance_frequency_at_n0 + np.linspace(-3e6, 3e6, 5)
            for q in qubits
        ]

        wf = displacement_calibration.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubit_uids,
            amplitudes=amplitudes,
            frequencies=frequencies,
            options=options,
        )
        wf.run()
