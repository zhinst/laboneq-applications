# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Smoke-tests for the selective transmon Rx calibration experiment."""

import numpy as np

from laboneq_applications.contrib.experiments.bosonic_qubits import (
    selective_transmon_rx_calibration,
)


class TestSelectiveTransmonRxCalibration:
    def test_selective_transmon_rx_calibration(self, single_bosonic_qubit_platform):
        platform = single_bosonic_qubit_platform
        qpu = platform.qpu
        qubit_uids = qpu.quantum_element_uids

        options = selective_transmon_rx_calibration.experiment_workflow.options()
        options.do_analysis(True)
        options.show_plot(False)

        session = platform.session(do_emulation=True)

        amplitudes_transmon = [np.linspace(0.0, 0.2, 5)]

        wf = selective_transmon_rx_calibration.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubit_uids,
            amplitudes_transmon=amplitudes_transmon,
            photon_numbers_memory=[0, 1],
            options=options,
        )
        wf.run()
