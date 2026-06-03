# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Smoke-tests for the SWAP gate calibration experiment."""

import numpy as np

from laboneq_applications.contrib.experiments.bosonic_qubits import (
    swap_calibration,
)


class TestSwapCalibration:
    def test_swap_calibration_length_sweep(self, single_bosonic_qubit_platform):
        platform = single_bosonic_qubit_platform
        qpu = platform.qpu
        qubit_uids = qpu.quantum_element_uids

        options = swap_calibration.experiment_workflow.options()
        options.do_analysis(True)
        options.show_plot(False)

        session = platform.session(do_emulation=True)

        swap_durations = [np.linspace(100e-9, 600e-9, 5)]

        wf = swap_calibration.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubit_uids,
            swap_durations=swap_durations,
            options=options,
        )
        wf.run()

    def test_swap_calibration_amplitude_sweep(self, single_bosonic_qubit_platform):
        platform = single_bosonic_qubit_platform
        qpu = platform.qpu
        qubit_uids = qpu.quantum_element_uids

        options = swap_calibration.experiment_workflow.options()
        options.do_analysis(True)
        options.show_plot(False)

        session = platform.session(do_emulation=True)

        swap_amplitudes = [np.linspace(0.05, 0.2, 5)]

        wf = swap_calibration.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubit_uids,
            swap_amplitudes=swap_amplitudes,
            options=options,
        )
        wf.run()

    def test_swap_calibration_2d_sweep(self, single_bosonic_qubit_platform):
        platform = single_bosonic_qubit_platform
        qpu = platform.qpu
        qubit_uids = qpu.quantum_element_uids

        options = swap_calibration.experiment_workflow.options()
        options.do_analysis(True)
        options.show_plot(False)

        session = platform.session(do_emulation=True)

        swap_durations = [np.linspace(100e-9, 600e-9, 5)]
        swap_amplitudes = [np.linspace(0.05, 0.2, 5)]

        wf = swap_calibration.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubit_uids,
            swap_durations=swap_durations,
            swap_amplitudes=swap_amplitudes,
            options=options,
        )
        wf.run()
