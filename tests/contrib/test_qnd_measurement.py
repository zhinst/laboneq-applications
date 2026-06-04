# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Smoke-tests for the measurement_qndness experiments."""

import pytest

from laboneq_applications.contrib.experiments import measurement_qndness


# The emulated acquisition data is constant, so the shots contain only
# a single label and the confusion matrix in the analysis cannot be
# given the correct shape.
@pytest.mark.filterwarnings("ignore:A single label was found:UserWarning")
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
