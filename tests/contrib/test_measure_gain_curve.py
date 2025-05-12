# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Smoke-tests for the scan_pump_parameters experiments for TWPAs."""

import numpy as np

from laboneq_applications.contrib.experiments import measure_gain_curve


class TestMeasureGainCurve:
    def test_measure_gain_curve(self, single_twpa_platform):
        platform = single_twpa_platform
        qpu = platform.qpu
        [twpa] = platform.qpu.quantum_elements
        options = measure_gain_curve.experiment_workflow.options()
        options.do_analysis(True)
        session = platform.session(do_emulation=True)
        wf = measure_gain_curve.experiment_workflow(
            session=session,
            qpu=qpu,
            parametric_amplifier=twpa,
            probe_frequency=np.linspace(6.8e9, 7.2e9, 101),
            pump_power=np.linspace(0, 10, 11),
            options=options,
        )
        wf.run()
