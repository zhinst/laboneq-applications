# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Smoke-tests for the single_qubit_randomized_benchmarking experiments."""

import pytest

from laboneq_applications.contrib.experiments import (
    single_qubit_randomized_benchmarking,
)


# Qiskit's transpiler plugin discovery eagerly instantiates qiskit-ibm-runtime's
# deprecated IBMFractionalTranslationPlugin, triggering this internal warning.
@pytest.mark.filterwarnings(
    "ignore:Since backends now support running jobs:DeprecationWarning"
)
class TestSingleQubitRandomizedBenchmarking:
    def test_single_qubit_randomized_benchmarking(self, two_tunable_transmon_platform):
        platform = two_tunable_transmon_platform
        qpu = platform.qpu
        qubit_uids = qpu.quantum_element_uids
        options = single_qubit_randomized_benchmarking.experiment_workflow.options()
        options.do_analysis(True)
        session = platform.session(do_emulation=True)
        wf = single_qubit_randomized_benchmarking.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubit_uids,
            length_cliffords=[1, 4, 16, 64],
            variations=5,
            options=options,
        )
        wf.run()
