"""Tests for the compiled fine_amplitude experiment 
using the testing utilities provided by the LabOne Q Applications Library.
"""

import numpy as np
import pytest

from laboneq_applications.experiments import (
    fine_amplitude,
)
from laboneq_applications.testing import CompiledExperimentVerifier


def on_system_grid(time, system_grid=8):
    # time_ns is time in ns, system_grid is minimum stepsize in ns
    time_ns = time * 1e9

    remainder = round(time_ns % system_grid, 2)
    if remainder != 0.0:
        time_ns = time_ns + system_grid - remainder
    return round(time_ns * 1e-9, 12)


@pytest.mark.parametrize("iterations", [np.arange(1, 3, 1)])
@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize(
    "transition, cal_states",
    [("ge", "ge"), ("ef", "ef")],
)
class TestFineAmplitudeQubit:
    """Test for fine-amplitude on a single/two qubit

    `two_tunable_transmon_platform` is a pytest fixture that is automatically
    imported into the test function.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, two_tunable_transmon_platform):
        self.platform = two_tunable_transmon_platform
        self.qpu = self.platform.qpu
        self.qubits = self.platform.qpu.qubits

    @pytest.fixture(autouse=True)
    def _set_options(self, transition, cal_states):
        self.options = fine_amplitude.options()
        self.options.create_experiment.count = 5  # No need to be parameterized here
        self.options.create_experiment.transition = transition
        self.options.create_experiment.cal_states = cal_states

        self.transition = transition
        self.cal_states = cal_states
        self.count = self.options.create_experiment.count
        self.use_cal_traces = self.options.create_experiment.use_cal_traces

    @pytest.fixture(autouse=True)
    def create_fine_amplitude_verifier(self, iterations, num_qubits):
        res = fine_amplitude.experiment_workflow(
            session=self.platform.session(do_emulation=True),
            qubits=[self.qubits[i] for i in range(num_qubits)],
            qpu=self.qpu,
            iterations=[iterations for i in range(num_qubits)],
            options=self.options,
        ).run()
        self.num_qubits = num_qubits
        self.verifier = CompiledExperimentVerifier(
            res.tasks["compile_experiment"].output
        )

    def test_pulse_count_drive(self, iterations):
        """Test the total number of drive pulses with given iterations"""

        if self.transition == "ge":
            expected_ge = self.count * np.sum(iterations)
            expected_ef = 0
        elif self.transition == "ef":
            expected_ge = self.count * len(iterations)
            expected_ef = self.count * np.sum(iterations)

        if self.cal_states == "ge":
            expected_ge += self.count * int(self.use_cal_traces)
        elif self.cal_states in "ef":
            expected_ge += self.count * 2 * int(self.use_cal_traces)
            expected_ef += self.count * int(self.use_cal_traces)

        for i in range(self.num_qubits):
            self.verifier.assert_number_of_pulses(
                f"/logical_signal_groups/q{i}/drive",
                expected_ge,
            )
            self.verifier.assert_number_of_pulses(
                f"/logical_signal_groups/q{i}/drive_ef",
                expected_ef,
            )

    def test_pulse_count_measure_acquire(self, iterations):
        """Test the total number of meausre and acquire pulses with given iterations"""

        expected_measure = self.count * (len(iterations))
        if self.cal_states in ("ge", "ef"):
            expected_measure += self.count * 2 * int(self.use_cal_traces)
        for i in range(self.num_qubits):
            self.verifier.assert_number_of_pulses(
                f"/logical_signal_groups/q{i}/measure",
                expected_measure,
            )
            self.verifier.assert_number_of_pulses(
                f"/logical_signal_groups/q{i}/acquire",
                expected_measure,
            )

    def test_pulse_drive(self, iterations):
        """Test the timing of drive pulses at each iteration"""
        length_ge = 51e-9
        length_ef = 52e-9
        length_measure_reset = 2e-6 + 1e-6

        def total_length_drive(iteration, transition="ge"):
            if transition == "ge":
                length = length_ge
            elif transition == "ef":
                length = length_ef
            if iteration > 1:
                return length * (iteration - 1) + on_system_grid(length)
            return length * iteration

        def total_length_measure(index):
            return length_measure_reset * (index)

        time_start = 0
        for index, iteration in enumerate(iterations):
            if self.transition == "ge":
                time_end = total_length_measure(index) + total_length_drive(
                    iteration, "ge"
                )
                self.verifier.assert_pulse(
                    signal="/logical_signal_groups/q0/drive",
                    index=index,
                    start=time_start,
                    end=time_end,
                )
                time_start = on_system_grid(time_end) + length_measure_reset

            elif self.transition == "ef":
                time_end = +total_length_measure(index) + total_length_drive(
                    iteration, "ef"
                )
                self.verifier.assert_pulse(
                    signal="/logical_signal_groups/q0/drive_ef",
                    index=index,
                    start=time_start + on_system_grid(length_ge) * iteration,
                    end=time_end + on_system_grid(length_ge) * iteration,
                )
                time_start = on_system_grid(time_end) + length_measure_reset

    def test_pulse_measure(self, iterations):
        """Test the timing of measure pulses at each iteration"""
        length_ge = 51e-9
        readout_length = 2e-6

        if iterations[0] != 0:
            measure_start = on_system_grid(length_ge)
            measure_end = measure_start + readout_length
            integration_start = measure_start
            integration_end = measure_end

            if self.transition == "ef":
                measure_start += on_system_grid(length_ge)
                measure_end += on_system_grid(length_ge)
                integration_start += on_system_grid(length_ge)
                integration_end += on_system_grid(length_ge)

            for i in range(self.num_qubits):
                self.verifier.assert_pulse(
                    signal=f"/logical_signal_groups/q{i}/measure",
                    index=0,
                    start=measure_start,
                    end=measure_end,
                )
                self.verifier.assert_pulse(
                    signal=f"/logical_signal_groups/q{i}/acquire",
                    index=0,
                    start=integration_start,
                    end=integration_end,
                )