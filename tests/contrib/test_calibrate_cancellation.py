# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Smoke-tests for the calibrate_cancellation experiments for TWPAs."""

import numpy as np

from laboneq_applications.contrib.experiments import calibrate_cancellation


class TestCalibrateCancellation:
    def test_calibrate_cancellation(self, single_twpa_platform):
        platform = single_twpa_platform
        qpu = platform.qpu
        [twpa] = platform.qpu.quantum_elements
        options = calibrate_cancellation.experiment_workflow.options()
        options.do_analysis(True)
        session = platform.session(do_emulation=True)
        wf = calibrate_cancellation.experiment_workflow(
            session=session,
            qpu=qpu,
            parametric_amplifier=twpa,
            cancel_phase=np.linspace(0, np.pi, 11),
            cancel_attenuation=np.linspace(0, 10, 11),
            options=options,
        )
        wf.run()
