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
@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize(
    "transition, cal_states",
    [("ge", "ge"), ("ef", "ef")],
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
        self.options = amplitude_fine.options()
        self.options.create_experiment.count = 5  # No need to be parameterized here
        self.options.create_experiment.transition = transition
        self.options.create_experiment.cal_states = cal_states
        self.options.do_analysis = False

        self.transition = transition
        self.cal_states = cal_states
        self.count = self.options.create_experiment.count
        self.use_cal_traces = self.options.create_experiment.use_cal_traces

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
