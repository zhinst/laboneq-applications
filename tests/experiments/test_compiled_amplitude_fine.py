"""Tests for the compiled amplitude_fine experiments.

NOTES: The relative timing check are skipped and need investigations for
fixing the unwanted gaps between ef drive pulses and other pulses."""

import numpy as np
import pytest

from laboneq_applications.experiments import (
    amplitude_fine,
)
from laboneq_applications.testing import CompiledExperimentVerifier

_LENGTH_GE = 51e-9
_LENGTH_EF = 52e-9
_LENGTH_MEASURE = 2e-6
_LENGTH_MEASURE_RESET = 2e-6 + 1e-6
_COUNT = 5
_NUM_QUBITS = 2
_REPETITIONS = np.arange(1, 5, 1)


def on_system_grid(time, system_grid=8):
    # time_ns is time in ns, system_grid is minimum stepsize in ns
    # this will be integrated somewhere soon
    # TODO: move this to testing utils for sharing
    time_ns = time * 1e9

    remainder = round(time_ns % system_grid, 2)
    if remainder != 0.0:
        time_ns = time_ns + system_grid - remainder
    return round(time_ns * 1e-9, 12)


@pytest.mark.parametrize("transition, cal_states", [("ge", "ge"), ("ef", "ef")])
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
        self.options.count(_COUNT)
        self.options.transition(transition)
        self.options.cal_states(cal_states)
        self.options.do_analysis(False)

        self.transition = transition
        self.cal_states = cal_states
        self.use_cal_traces = True

    @pytest.fixture(autouse=True)
    def create_fine_amplitude_verifier(self):
        res = amplitude_fine.experiment_workflow(
            session=self.platform.session(do_emulation=True),
            qpu=self.qpu,
            qubits=self.qubits,
            amplification_qop="x180",
            target_angle=0,
            phase_offset=0.0,
            repetitions=[_REPETITIONS] * _NUM_QUBITS,
            parameter_to_update="x180",
            options=self.options,
        ).run()
        self.verifier = CompiledExperimentVerifier(
            res.tasks["compile_experiment"].output, max_events=10000
        )
        return self.verifier

    def test_pulse_count_drive(self):
        """Verify the total number of drive pulses"""

        if self.transition == "ge":
            expected_ge = _COUNT * (len(_REPETITIONS) + np.sum(_REPETITIONS))
            expected_ef = 0
        elif self.transition == "ef":
            expected_ge = _COUNT * len(_REPETITIONS)
            expected_ef = _COUNT * (len(_REPETITIONS) + np.sum(_REPETITIONS))

        if self.cal_states == "ge":
            expected_ge += _COUNT * int(self.use_cal_traces)
        elif self.cal_states in "ef":
            expected_ge += _COUNT * 2 * int(self.use_cal_traces)
            expected_ef += _COUNT * int(self.use_cal_traces)

        for i in range(_NUM_QUBITS):
            self.verifier.assert_number_of_pulses(
                f"/logical_signal_groups/q{i}/drive",
                expected_ge,
            )
            self.verifier.assert_number_of_pulses(
                f"/logical_signal_groups/q{i}/drive_ef",
                expected_ef,
            )

    def test_pulse_count_measure_acquire(self):
        """Verify the total number of measure and acquire pulses"""

        expected_measure = _COUNT * (len(_REPETITIONS))
        if self.cal_states in ("ge", "ef"):
            expected_measure += _COUNT * 2 * int(self.use_cal_traces)
        for i in range(_NUM_QUBITS):
            self.verifier.assert_number_of_pulses(
                f"/logical_signal_groups/q{i}/measure",
                expected_measure,
            )
            self.verifier.assert_number_of_pulses(
                f"/logical_signal_groups/q{i}/acquire",
                expected_measure,
            )

    def test_pulse_drive_length(self):
        """Test the timing of drive pulses"""

        # check length for state preparation
        for i in range(_NUM_QUBITS):
            self.verifier.assert_pulse(
                signal=f"/logical_signal_groups/q{i}/drive",
                index=0,
                length=_LENGTH_GE,
            )  # x90_ge
            if self.transition == "ef":
                self.verifier.assert_pulse(
                    signal=f"/logical_signal_groups/q{i}/drive_ef",
                    index=0,
                    length=_LENGTH_EF,
                )  # x180_ge + x90_ef

    @pytest.mark.skip("Skipped for testing pulse timing, issue in ef-transition")
    def test_pulse_drive_timing(self):
        # check timing gap between 1st and 2nd pulse at every repetition
        for i in range(_NUM_QUBITS):
            index_rep = 0  # index for the first pulse at every repetition
            for index, rep in enumerate(_REPETITIONS):
                if self.transition == "ge":
                    self.verifier.assert_pulse_pair(
                        signals=f"/logical_signal_groups/q{i}/drive",
                        indices=(index_rep, index_rep + 1),
                        distance=0,
                    )  # no gap
                else:
                    self.verifier.assert_pulse_pair(
                        signals=f"/logical_signal_groups/q{i}/drive_ef",
                        indices=(index_rep, index_rep + 1),
                        distance=(on_system_grid(_LENGTH_GE) - _LENGTH_EF),
                    )  # small gaps due to system grid alignment.
                    self.verifier.assert_pulse_pair(
                        signals=(
                            f"/logical_signal_groups/q{i}/drive",
                            f"/logical_signal_groups/q{i}/drive_ef",
                        ),
                        indices=(
                            index,
                            index_rep,
                        ),  # why 'index'? because single x180 added on q0/drive
                        distance=(on_system_grid(_LENGTH_GE) - _LENGTH_GE),
                    )
                index_rep += 1 + rep  # 1:state preparation, rep:repetition

    @pytest.mark.skip("Skipping for testing pulse timing, issue in ge-transition")
    def test_pulse_measure(self):
        """Test the timing of measure pulses"""

        for i in range(_NUM_QUBITS):
            num_drive = 0
            for index, rep in enumerate(_REPETITIONS):
                if self.transition == "ge":
                    num_drive += 1 + rep
                    length_drive = on_system_grid(_LENGTH_EF * num_drive)
                    # why length_ef? it has to be length_ge
                elif self.transition == "ef":
                    num_drive += 2 + rep
                    length_drive = on_system_grid(_LENGTH_GE) * num_drive

                time_start = length_drive + _LENGTH_MEASURE_RESET * index

                self.verifier.assert_pulse(
                    signal=f"/logical_signal_groups/q{i}/measure",
                    index=index,
                    start=time_start,
                    end=time_start + _LENGTH_MEASURE,
                )
                self.verifier.assert_pulse(
                    signal=f"/logical_signal_groups/q{i}/acquire",
                    index=index,
                    start=time_start,
                    end=time_start + _LENGTH_MEASURE,
                )
