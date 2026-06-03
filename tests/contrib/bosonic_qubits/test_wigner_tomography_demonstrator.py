# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Smoke-tests for the Wigner tomography demonstrator experiment."""

import numpy as np

from laboneq_applications.contrib.experiments.bosonic_qubits import (
    wigner_tomography_demonstrator,
)


class TestWignerTomographyDemonstrator:
    def test_wigner_tomography_1d(self, single_bosonic_qubit_platform):
        platform = single_bosonic_qubit_platform
        qpu = platform.qpu
        qubit_uids = qpu.quantum_element_uids

        options = wigner_tomography_demonstrator.experiment_workflow.options()
        options.do_analysis(True)
        options.show_plot(False)

        session = platform.session(do_emulation=True)

        beta_re_values = [np.linspace(-2.0, 2.0, 5)]

        wf = wigner_tomography_demonstrator.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubit_uids,
            beta_re_values=beta_re_values,
            options=options,
        )
        wf.run()

    def test_wigner_tomography_2d(self, single_bosonic_qubit_platform):
        platform = single_bosonic_qubit_platform
        qpu = platform.qpu
        qubit_uids = qpu.quantum_element_uids

        options = wigner_tomography_demonstrator.experiment_workflow.options()
        options.do_analysis(True)
        options.show_plot(False)

        session = platform.session(do_emulation=True)

        beta_re_values = [np.linspace(-1.0, 1.0, 3)]
        beta_im_values = [np.linspace(-1.0, 1.0, 3)]

        wf = wigner_tomography_demonstrator.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubit_uids,
            beta_re_values=beta_re_values,
            beta_im_values=beta_im_values,
            options=options,
        )
        wf.run()
