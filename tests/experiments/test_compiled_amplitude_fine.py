# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for the compiled amplitude_fine experiments.

NOTES: The relative timing check are skipped and need investigations for
fixing the unwanted gaps between ef drive pulses and other pulses."""

import numpy as np
import pytest

from laboneq_applications.experiments import (
    amplitude_fine,
)
from laboneq_applications.testing import CompiledExperimentVerifier

_LENGTH_GE = 32e-9
_LENGTH_EF = 64e-9
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
@pytest.mark.parametrize("readout_lengths", [[1e-6, 1e-6], [100e-9, 200e-9]])
class TestAmplitudeFine:
    """Test for fine-amplitude on a single/two qubit"""

    @pytest.fixture(autouse=True)
    def _setup(self, two_tunable_transmon_platform, readout_lengths):
        self.platform = two_tunable_transmon_platform
        self.qpu = self.platform.qpu
        self.qubits = self.platform.qpu.qubits
        assert len(readout_lengths) == len(self.qubits)
        for i, rl in enumerate(readout_lengths):
            self.qubits[i].parameters.readout_length = rl
        for q in self.qubits:
            q.parameters.ge_drive_length = _LENGTH_GE
            q.parameters.ef_drive_length = _LENGTH_EF

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
                f"q{i}/drive",
                expected_ge,
            )
            self.verifier.assert_number_of_pulses(
                f"q{i}/drive_ef",
                expected_ef,
            )

    def test_pulse_count_measure_acquire(self):
        """Verify the total number of measure and acquire pulses"""

        expected_measure = _COUNT * (len(_REPETITIONS))
        if self.cal_states in ("ge", "ef"):
            expected_measure += _COUNT * 2 * int(self.use_cal_traces)
        for i in range(_NUM_QUBITS):
            self.verifier.assert_number_of_pulses(
                f"q{i}/measure",
                expected_measure,
            )
            self.verifier.assert_number_of_pulses(
                f"q{i}/acquire",
                expected_measure,
            )

    def test_pulse_drive_length(self):
        """Test the timing of drive pulses"""

        # check length for state preparation
        for i in range(_NUM_QUBITS):
            self.verifier.assert_pulse(
                signal=f"q{i}/drive",
                index=0,
                length=_LENGTH_GE,
            )  # x90_ge
            if self.transition == "ef":
                self.verifier.assert_pulse(
                    signal=f"q{i}/drive_ef",
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
                        signals=f"q{i}/drive",
                        indices=(index_rep, index_rep + 1),
                        distance=0,
                    )  # no gap
                else:
                    self.verifier.assert_pulse_pair(
                        signals=f"q{i}/drive_ef",
                        indices=(index_rep, index_rep + 1),
                        distance=(on_system_grid(_LENGTH_GE) - _LENGTH_EF),
                    )  # small gaps due to system grid alignment.
                    self.verifier.assert_pulse_pair(
                        signals=(
                            f"q{i}/drive",
                            f"q{i}/drive_ef",
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
                    signal=f"q{i}/measure",
                    index=index,
                    start=time_start,
                    end=time_start + _LENGTH_MEASURE,
                )
                self.verifier.assert_pulse(
                    signal=f"q{i}/acquire",
                    index=index,
                    start=time_start,
                    end=time_start + _LENGTH_MEASURE,
                )


@pytest.mark.parametrize(
    ("transition", "cal_states", "active_reset_states"),
    [("ge", "ge", "ge"), ("ef", "ef", "gef")],
)
@pytest.mark.parametrize(
    "active_reset_repetitions",
    [1, 5],
)
def test_single_qubit_run_with_active_reset(
    single_tunable_transmon_platform,
    transition,
    cal_states,
    active_reset_states,
    active_reset_repetitions,
):
    options = amplitude_fine.experiment_workflow_x180.options()
    options.transition(transition)
    options.cal_states(cal_states)
    options.active_reset(True)
    options.active_reset_states(active_reset_states)
    options.active_reset_repetitions(active_reset_repetitions)
    options.do_analysis(False)
    [q0] = single_tunable_transmon_platform.qpu.qubits
    repetitions = np.arange(21)
    workflow_result = amplitude_fine.experiment_workflow_x180(
        session=single_tunable_transmon_platform.session(do_emulation=True),
        qubits=q0,
        qpu=single_tunable_transmon_platform.qpu,
        repetitions=repetitions,
        options=options,
    ).run()

    exp = workflow_result.tasks["create_experiment"].output
    active_reset_section = exp.sections[0].children[0].children[0]
    assert active_reset_section.uid == "active_reset_q0_0"
    truth_len = len(q0.signals) + active_reset_repetitions
    assert len(active_reset_section.children) == truth_len

    data = workflow_result.output
    assert "active_reset" in data.q0
    shape_truth = (
        (len(repetitions), active_reset_repetitions)
        if active_reset_repetitions > 1
        else (len(repetitions),)
    )
    assert np.shape(data.q0.active_reset.result.data) == shape_truth
    for s in cal_states:
        cal_trace_data = data.q0.active_reset.cal_trace[s].data
        if active_reset_repetitions == 1:
            assert isinstance(cal_trace_data, np.complex128)
        else:
            assert len(cal_trace_data) == active_reset_repetitions


@pytest.mark.parametrize(
    ("transition", "cal_states", "active_reset_states"),
    [("ge", "ge", "ge"), ("ef", "ef", "gef")],
)
@pytest.mark.parametrize(
    "active_reset_repetitions",
    [1, 5],
)
def test_two_qubit_run_with_active_reset(
    two_tunable_transmon_platform,
    transition,
    cal_states,
    active_reset_states,
    active_reset_repetitions,
):
    options = amplitude_fine.experiment_workflow_x180.options()
    options.transition(transition)
    options.cal_states(cal_states)
    options.active_reset(True)
    options.active_reset_states(active_reset_states)
    options.active_reset_repetitions(active_reset_repetitions)
    options.do_analysis(False)
    qubits = two_tunable_transmon_platform.qpu.qubits
    repetitions = np.arange(21)
    workflow_result = amplitude_fine.experiment_workflow_x180(
        session=two_tunable_transmon_platform.session(do_emulation=True),
        qubits=qubits,
        qpu=two_tunable_transmon_platform.qpu,
        repetitions=[repetitions, repetitions],
        options=options,
    ).run()

    exp = workflow_result.tasks["create_experiment"].output
    active_reset_section = exp.sections[0].children[0].children[0]
    assert active_reset_section.uid == "active_reset_q0_q1_0"
    truth_len = len(qubits[0].signals) * len(qubits) + active_reset_repetitions
    assert len(active_reset_section.children) == truth_len

    data = workflow_result.output
    assert "active_reset" in data.q0
    assert "active_reset" in data.q1
    shape_truth = (
        (len(repetitions), active_reset_repetitions)
        if active_reset_repetitions > 1
        else (len(repetitions),)
    )
    assert np.shape(data.q0.active_reset.result.data) == shape_truth
    assert np.shape(data.q1.active_reset.result.data) == shape_truth
    for s in cal_states:
        for q in qubits:
            cal_trace_data = data[q.uid].active_reset.cal_trace[s].data
            if active_reset_repetitions == 1:
                assert isinstance(cal_trace_data, np.complex128)
            else:
                assert len(cal_trace_data) == active_reset_repetitions


def test_invalid_averaging_mode(single_tunable_transmon_platform):
    [q0] = single_tunable_transmon_platform.qpu.qubits
    session = single_tunable_transmon_platform.session(do_emulation=True)
    options = amplitude_fine.experiment_workflow.options()
    options.averaging_mode("sequential")
    options.use_cal_traces(True)
    options.do_analysis(False)

    with pytest.raises(ValueError) as err:
        amplitude_fine.experiment_workflow(
            session=session,
            qubits=q0,
            qpu=single_tunable_transmon_platform.qpu,
            amplification_qop="x180",
            target_angle=0,
            phase_offset=0.0,
            repetitions=[0, 1, 2],
            parameter_to_update="x180",
            options=options,
        ).run()

    assert str(err.value) == (
        "'AveragingMode.SEQUENTIAL' (or {AveragingMode.SEQUENTIAL}) cannot be used "
        "with calibration traces because the calibration traces are added "
        "outside the sweep."
    )
