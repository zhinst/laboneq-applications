# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Smoke-tests for the two_qubit_rb experiment."""

from laboneq_applications.contrib.experiments.two_qubit import two_qubit_rb


class TestTwoQubitRB:
    def test_two_qubit_rb(self, four_tunable_transmon_cz_platform):
        platform = four_tunable_transmon_cz_platform
        qpu = platform.qpu
        qubit_pairs = [["q0", "q1"]]
        options = two_qubit_rb.experiment_workflow.options()
        options.do_analysis(True)
        session = platform.session(do_emulation=True)
        wf = two_qubit_rb.experiment_workflow(
            session=session,
            qpu=qpu,
            qubit_pairs=qubit_pairs,
            length_cliffords=[1, 4, 16],
            variations=2,
            seed=42,
            options=options,
        )
        wf.run()
