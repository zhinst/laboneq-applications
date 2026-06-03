# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Smoke-tests for the SNAP gate calibration experiment."""

import numpy as np

from laboneq_applications.contrib.experiments.bosonic_qubits import (
    snap_calibration,
)


class TestSnapCalibration:
    def test_snap_calibration_amplitude_only(self, single_bosonic_qubit_platform):
        platform = single_bosonic_qubit_platform
        qpu = platform.qpu
        qubit_uids = qpu.quantum_element_uids

        options = snap_calibration.experiment_workflow.options()
        options.do_analysis(True)
        options.do_phase_calibration(False)
        options.show_plot(False)

        session = platform.session(do_emulation=True)

        amplitudes_transmon = [np.linspace(0.05, 0.15, 5)]

        wf = snap_calibration.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubit_uids,
            amplitudes_transmon=amplitudes_transmon,
            photon_number_memory=1,
            options=options,
        )
        wf.run()

    def test_snap_calibration_phase_without_analysis(
        self, single_bosonic_qubit_platform
    ):
        platform = single_bosonic_qubit_platform
        qpu = platform.qpu
        qubit_uids = qpu.quantum_element_uids

        options = snap_calibration.experiment_workflow.options()
        options.do_analysis(False)
        options.do_phase_calibration(True)
        options.show_plot(False)

        session = platform.session(do_emulation=True)

        amplitudes_transmon = [np.linspace(0.05, 0.15, 5)]
        phases_transmon = [np.linspace(0, 2 * np.pi, 11)]

        wf = snap_calibration.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubit_uids,
            amplitudes_transmon=amplitudes_transmon,
            phases_transmon=phases_transmon,
            photon_number_memory=1,
            options=options,
        )
        wf.run()

    def test_snap_calibration_full(self, single_bosonic_qubit_platform):
        platform = single_bosonic_qubit_platform
        qpu = platform.qpu
        qubit_uids = qpu.quantum_element_uids

        options = snap_calibration.experiment_workflow.options()
        options.do_analysis(True)
        options.do_phase_calibration(True)
        options.show_plot(False)

        session = platform.session(do_emulation=True)

        amplitudes_transmon = [np.linspace(0.05, 0.15, 5)]
        phases_transmon = [np.linspace(0, 2 * np.pi, 11)]

        wf = snap_calibration.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubit_uids,
            amplitudes_transmon=amplitudes_transmon,
            phases_transmon=phases_transmon,
            photon_number_memory=1,
            options=options,
        )
        wf.run()
