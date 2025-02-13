# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Smoke-tests for the single_qubit_randomized_benchmarking experiments."""

from laboneq_applications.contrib.experiments import (
    single_qubit_randomized_benchmarking,
)


class TestSingleQubitRandomizedBenchmarking:
    def test_single_qubit_randomized_benchmarking(self, two_tunable_transmon_platform):
        platform = two_tunable_transmon_platform
        qpu = platform.qpu
        qubits = platform.qpu.qubits
        options = single_qubit_randomized_benchmarking.experiment_workflow.options()
        options.do_analysis(True)
        session = platform.session(do_emulation=True)
        wf = single_qubit_randomized_benchmarking.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubits,
            length_cliffords=[1, 4, 16, 64],
            variations=5,
            options=options,
        )
        wf.run()
