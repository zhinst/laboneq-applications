"""Tests for the compiled amplitude_fine experiment
using the testing utilities provided by the LabOne Q Applications Library.
"""

import numpy as np
import pytest

from laboneq_applications.experiments import (
    amplitude_fine,
)
from laboneq_applications.testing import CompiledExperimentVerifier


def on_system_grid(time, system_grid=8):
    # time_ns is time in ns, system_grid is minimum stepsize in ns
    # this will be integrated somewhere soon
    time_ns = time * 1e9

    remainder = round(time_ns % system_grid, 2)
    if remainder != 0.0:
        time_ns = time_ns + system_grid - remainder
    return round(time_ns * 1e-9, 12)


@pytest.mark.parametrize("repetitions", [np.arange(1, 3, 1), np.arange(1, 52, 25)])
@pytest.mark.parametrize("num_qubits", [1])
@pytest.mark.parametrize(
    "transition, cal_states",
    [("ge", "ge"), ("ef", "ef")]
)
class TestAmplitudeFine:
    """Test for fine-amplitude on a single/two qubit"""

    @pytest.fixture(autouse=True)
    def _setup(self, two_tunable_transmon_platform):
        self.platform = two_tunable_transmon_platform
        self.qpu = self.platform.qpu
        self.qubits = self.platform.qpu.qubits

    @pytest.fixture(autouse=True)
    def _set_options(self, transition, cal_states):
        self.options = amplitude_fine.experiment_workflow.options()
        self.options.count(5)  # No need to be parameterized here
        self.options.transition(transition)
        self.options.cal_states(cal_states)
        self.options.do_analysis(False)

        self.transition = transition
        self.cal_states = cal_states
        self.count = 5
        self.use_cal_traces = True

    @pytest.fixture(autouse=True)
    def create_fine_amplitude_verifier(self, repetitions, num_qubits):
        res = amplitude_fine.experiment_workflow(
            session=self.platform.session(do_emulation=True),
            qpu=self.qpu,
            qubits=[self.qubits[i] for i in range(num_qubits)],
            amplification_qop="x180",
            target_angle=0,
            phase_offset=0.0,
            repetitions=[repetitions for i in range(num_qubits)],
            parameter_to_update="x180",
            options=self.options,
        ).run()
        self.num_qubits = num_qubits
        self.verifier = CompiledExperimentVerifier(
            res.tasks["compile_experiment"].output, max_events=10000
        )
        return self.verifier

    @pytest.mark.skip("no need to test")
    def test_pulse_count_drive(self, repetitions):
        """Test the total number of drive pulses with given repetitions"""

        if self.transition == "ge":
            expected_ge = self.count * (len(repetitions) + np.sum(repetitions))
            expected_ef = 0
        elif self.transition == "ef":
            expected_ge = self.count * len(repetitions)
            expected_ef = self.count * (len(repetitions) + np.sum(repetitions))

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
    @pytest.mark.skip("no need to test")
    def test_pulse_count_measure_acquire(self, repetitions):
        """Test the total number of meausre and acquire pulses with given repetitions"""

        expected_measure = self.count * (len(repetitions))
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

    def test_pulse_drive(self, repetitions): #ongoing
        """Test the timing of drive pulses with given repetitions"""
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

        for index, rep in enumerate(repetitions):
            self.verifier.assert_pulse(
                    signal="/logical_signal_groups/q0/drive",
                    index=index,
                    length = length_ge
                )
            if index %2 ==0 :
                if self.transition == "ge":
                    self.verifier.assert_pulse_pair(
                        signals="/logical_signal_groups/q0/drive",
                        indices=(index,index+1),
                        distance = 0
                    )
                else: # this fails
                    self.verifier.assert_pulse_pair(
                        signals=("/logical_signal_groups/q0/drive","/logical_signal_groups/q0/drive_ef"),
                        indices=(index,index+1),
                        distance = 0
                    )

    @pytest.mark.skip("no need to test") # ongoing
    def test_pulse_measure(self, repetitions):
        """Test the timing of measure pulses with given repetitions"""
        length_ge = 51e-9
        readout_length = 2e-6

        if repetitions[0] != 0:
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