# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Smoke-tests for the para_cz_corr_sq_phases_tc experiment."""

import numpy as np

from laboneq_applications.contrib.experiments.two_qubit import (
    para_cz_corr_sq_phases_tc,
)


class TestParametricCZCorrectSqPhases:
    def test_para_cz_corr_sq_phases_tc(self, four_tunable_transmon_cz_platform):
        platform = four_tunable_transmon_cz_platform
        qpu = platform.qpu
        qubit_pairs = [["q0", "q1"], ["q2", "q3"]]
        options = para_cz_corr_sq_phases_tc.experiment_workflow.options()
        options.count(2**8)
        options.do_analysis(True)
        session = platform.session(do_emulation=True)
        wf = para_cz_corr_sq_phases_tc.experiment_workflow(
            session=session,
            qpu=qpu,
            qubit_pairs=qubit_pairs,
            phases=[np.linspace(0, 2 * np.pi, 11) for _ in qubit_pairs],
            options=options,
        )
        wf.run()
