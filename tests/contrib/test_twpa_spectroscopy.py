# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Smoke-tests for the twpa_spectroscopy experiments."""

import numpy as np

from laboneq_applications.contrib.experiments import twpa_spectroscopy


class TestTwpaSpectroscopy:
    def test_twpa_spectroscopy(self, single_twpa_platform):
        platform = single_twpa_platform
        qpu = platform.qpu
        [twpa] = platform.qpu.qubits
        options = twpa_spectroscopy.experiment_workflow.options()
        options.do_analysis(True)
        session = platform.session(do_emulation=True)
        wf = twpa_spectroscopy.experiment_workflow(
            session=session,
            qpu=qpu,
            parametric_amplifier=twpa,
            frequencies=twpa.parameters.probe_frequency
            + np.linspace(-100e6, 100e6, 101),
            options=options,
        )
        wf.run()
