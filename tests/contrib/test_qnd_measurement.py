# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Smoke-tests for the measurement_qndness experiments."""

from laboneq_applications.contrib.experiments import measurement_qndness


class TestQNDMeasurement:
    def test_qnd_measurement(self, two_tunable_transmon_platform):
        platform = two_tunable_transmon_platform
        qpu = platform.qpu
        qubit_uids = qpu.quantum_element_uids
        options = measurement_qndness.experiment_workflow.options()
        options.do_analysis(True)
        session = platform.session(do_emulation=True)
        wf = measurement_qndness.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubit_uids,
            options=options,
        )
        wf.run()
