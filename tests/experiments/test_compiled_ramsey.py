# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for the compiled ramsey experiment using the testing utilities
provided by the LabOne Q Applications Library.
"""

from typing import ClassVar

import numpy as np
import pytest

from laboneq_applications.experiments import ramsey
from laboneq_applications.testing import CompiledExperimentVerifier

_LENGTH_GE = 32e-9
_LENGTH_EF = 64e-9


def create_ramsey_verifier(
    tunable_transmon_platform,
    delays,
    count,
    transition,
    use_cal_traces,
    readout_lengths=None,
):
    """Create a CompiledExperimentVerifier for the ramsey experiment."""
    qubits = tunable_transmon_platform.qpu.qubits
    for q in qubits:
        q.parameters.ge_drive_length = _LENGTH_GE
        q.parameters.ef_drive_length = _LENGTH_EF
    if len(qubits) == 1:
        qubits = qubits[0]
    if readout_lengths is not None:
        assert len(readout_lengths) == len(qubits)
        for i, rl in enumerate(readout_lengths):
            qubits[i].parameters.readout_length = rl
    session = tunable_transmon_platform.session(do_emulation=True)
    options = ramsey.experiment_workflow.options()
    options.count(count)
    options.transition(transition)
    options.use_cal_traces(use_cal_traces)
    options.do_analysis(False)
    res = ramsey.experiment_workflow(
        session=session,
        qubits=qubits,
        qpu=tunable_transmon_platform.qpu,
        delays=delays,
        options=options,
    ).run()

    return CompiledExperimentVerifier(
        res.tasks["compile_experiment"].output, max_events=10000
    )


@pytest.mark.parametrize(
    "count",
    [2, 4],
)
@pytest.mark.parametrize(
    "transition",
    ["ge", "ef"],
)
@pytest.mark.parametrize(
    "use_cal_traces",
    [True, False],
)
class TestRamseySingleQubit:
    _DELAYS = tuple(np.arange(8e-9, 1.6e-6, 128e-9))

    def test_pulse_count_drive(
        self,
        single_tunable_transmon_platform,
        count,
        transition,
        use_cal_traces,
    ):
        """Test the number of drive pulses.`"""
        verifier = create_ramsey_verifier(
            single_tunable_transmon_platform,
            self._DELAYS,
            count,
            transition,
            use_cal_traces,
        )

        # with cal_state on, there is 1 additional drive pulse
        if transition == "ge":
            expected_drive_count = count * (2 * len(self._DELAYS) + int(use_cal_traces))
            verifier.assert_number_of_pulses(
                "q0/drive",
                expected_drive_count,
            )

        if transition == "ef":
            expected_drive_count = count * 2 * len(self._DELAYS)
            verifier.assert_number_of_pulses(
                "q0/drive_ef",
                expected_drive_count,
            )

    def test_pulse_count_measure_acquire(
        self,
        single_tunable_transmon_platform,
        count,
        transition,
        use_cal_traces,
    ):
        """Test the number of measure and acquire pulses."""
        verifier = create_ramsey_verifier(
            single_tunable_transmon_platform,
            self._DELAYS,
            count,
            transition,
            use_cal_traces,
        )
        # with cal_state on, there are 2 additional measure pulses
        expected_measure_count = count * (len(self._DELAYS) + 2 * int(use_cal_traces))
        verifier.assert_number_of_pulses(
            "q0/measure",
            expected_measure_count,
        )

        # acquire and measure pulses have the same count
        verifier.assert_number_of_pulses(
            "q0/acquire",
            expected_measure_count,
        )

    def test_pulse_drive(
        self,
        single_tunable_transmon_platform,
        count,
        transition,
        use_cal_traces,
    ):
        """Test the properties of drive pulses."""
        verifier = create_ramsey_verifier(
            single_tunable_transmon_platform,
            self._DELAYS,
            count,
            transition,
            use_cal_traces,
        )
        if transition == "ge":
            verifier.assert_pulse(
                signal="q0/drive",
                index=0,
                start=0,
                end=_LENGTH_GE,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="q0/drive",
                index=1,
                start=_LENGTH_GE + self._DELAYS[0],
                end=2 * _LENGTH_GE + self._DELAYS[0],
                parameterized_with=["x90_phases_q0"],
            )
        elif transition == "ef":
            start_ef = _LENGTH_GE
            verifier.assert_pulse(
                signal="q0/drive_ef",
                index=0,
                start=start_ef,
                end=start_ef + _LENGTH_EF,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="q0/drive_ef",
                index=1,
                start=start_ef + _LENGTH_EF + self._DELAYS[0],
                end=start_ef + 2 * _LENGTH_EF + self._DELAYS[0],
                parameterized_with=["x90_phases_q0"],
            )

    def test_pulse_measure(
        self,
        single_tunable_transmon_platform,
        count,
        transition,
        use_cal_traces,
    ):
        """Test the properties of measure pulses."""
        verifier = create_ramsey_verifier(
            single_tunable_transmon_platform,
            self._DELAYS,
            count,
            transition,
            use_cal_traces,
        )
        # The starting of measure pulses depends
        # on subsequent drive pulses and "jumps"
        # that happen when aligning to system_grid
        # hardcoded it here

        if transition == "ge":
            start_measure = 2 * _LENGTH_GE + self._DELAYS[0]
            verifier.assert_pulse(
                signal="q0/measure",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )
            verifier.assert_pulse(
                signal="q0/acquire",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )
        elif transition == "ef":
            start_measure = _LENGTH_GE + 2 * _LENGTH_EF + self._DELAYS[0]
            verifier.assert_pulse(
                signal="q0/measure",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )
            verifier.assert_pulse(
                signal="q0/acquire",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )


@pytest.mark.parametrize(
    ("count", "readout_lengths"),
    [(2, [1e-6, 1e-6]), (4, [100e-9, 200e-9])],
)
@pytest.mark.parametrize(
    "transition",
    ["ge", "ef"],
)
@pytest.mark.parametrize(
    "use_cal_traces",
    [True, False],
)
class TestRamseyTwoQubits:
    _DELAYS: ClassVar = [
        tuple(np.arange(8e-9, 1.6e-6, 128e-9)),
        tuple(np.arange(8e-9, 1.6e-6, 128e-9)),
    ]  # validate_and_convert_sweeps_to_arrays requires a list

    def test_pulse_count(
        self,
        two_tunable_transmon_platform,
        count,
        transition,
        use_cal_traces,
        readout_lengths,
    ):
        """Test the number of pulses."""
        verifier = create_ramsey_verifier(
            two_tunable_transmon_platform,
            self._DELAYS,
            count,
            transition,
            use_cal_traces,
            readout_lengths,
        )

        # with cal_state on, there is 1 additional drive pulse
        if transition == "ge":
            expected_drive_count = count * (
                2 * len(self._DELAYS[0]) + int(use_cal_traces)
            )
            verifier.assert_number_of_pulses(
                "q0/drive",
                expected_drive_count,
            )

            expected_drive_count = count * (
                2 * len(self._DELAYS[1]) + int(use_cal_traces)
            )
            verifier.assert_number_of_pulses(
                "q1/drive",
                expected_drive_count,
            )

        if transition == "ef":
            expected_drive_count = count * (len(self._DELAYS[0]) + int(use_cal_traces))
            verifier.assert_number_of_pulses(
                "q0/drive",
                expected_drive_count,
            )

            expected_drive_count = count * (len(self._DELAYS[1]) + int(use_cal_traces))
            verifier.assert_number_of_pulses(
                "q1/drive",
                expected_drive_count,
            )

            expected_drive_count = count * (2 * len(self._DELAYS[0]))
            verifier.assert_number_of_pulses(
                "q0/drive_ef",
                expected_drive_count,
            )

            expected_drive_count = count * (2 * len(self._DELAYS[1]))
            verifier.assert_number_of_pulses(
                "q1/drive_ef",
                expected_drive_count,
            )

        # with cal_state on, there are 2 additional measure pulses
        expected_measure_count = count * (
            len(self._DELAYS[0]) + 2 * int(use_cal_traces)
        )
        verifier.assert_number_of_pulses(
            "q0/measure",
            expected_measure_count,
        )

        # acquire and measure pulses have the same count
        verifier.assert_number_of_pulses(
            "q0/acquire",
            expected_measure_count,
        )

        verifier.assert_number_of_pulses(
            "q1/measure",
            expected_measure_count,
        )

        # acquire and measure pulses have the same count
        verifier.assert_number_of_pulses(
            "q1/acquire",
            expected_measure_count,
        )

    def test_pulse_drive(
        self,
        two_tunable_transmon_platform,
        count,
        transition,
        use_cal_traces,
        readout_lengths,
    ):
        """Test the properties of drive pulses."""
        verifier = create_ramsey_verifier(
            two_tunable_transmon_platform,
            self._DELAYS,
            count,
            transition,
            use_cal_traces,
            readout_lengths,
        )
        if transition == "ge":
            verifier.assert_pulse(
                signal="q0/drive",
                index=0,
                end=_LENGTH_GE,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="q0/drive",
                index=1,
                start=_LENGTH_GE + self._DELAYS[0][0],
                end=2 * _LENGTH_GE + self._DELAYS[0][0],
                parameterized_with=["x90_phases_q0"],
            )
            verifier.assert_pulse(
                signal="q1/drive",
                index=0,
                end=_LENGTH_GE,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="q1/drive",
                index=1,
                start=_LENGTH_GE + self._DELAYS[0][0],
                end=2 * _LENGTH_GE + self._DELAYS[0][0],
                parameterized_with=["x90_phases_q1"],
            )
        elif transition == "ef":
            start_ef_q0 = _LENGTH_GE
            start_ef_q1 = _LENGTH_GE
            verifier.assert_pulse(
                signal="q0/drive_ef",
                index=0,
                end=start_ef_q0 + _LENGTH_EF,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="q0/drive_ef",
                index=1,
                start=start_ef_q0 + _LENGTH_EF + self._DELAYS[0][0],
                end=start_ef_q0 + 2 * _LENGTH_EF + self._DELAYS[0][0],
                parameterized_with=["x90_phases_q0"],
            )
            verifier.assert_pulse(
                signal="q1/drive_ef",
                index=0,
                start=start_ef_q1,
                end=start_ef_q1 + _LENGTH_EF,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="q1/drive_ef",
                index=1,
                start=start_ef_q1 + _LENGTH_EF + self._DELAYS[1][0],
                end=start_ef_q1 + 2 * _LENGTH_EF + self._DELAYS[0][0],
                parameterized_with=["x90_phases_q1"],
            )

    def test_pulse_measure(
        self,
        two_tunable_transmon_platform,
        count,
        transition,
        use_cal_traces,
        readout_lengths,
    ):
        """Test the properties of measure pulses."""
        verifier = create_ramsey_verifier(
            two_tunable_transmon_platform,
            self._DELAYS,
            count,
            transition,
            use_cal_traces,
            readout_lengths,
        )

        if transition == "ge":
            # See the explanation for the hardcoding of start_measure
            # in the single-qubit tests
            start_measure = 2 * _LENGTH_GE + self._DELAYS[0][0]
            verifier.assert_pulse(
                signal="q0/measure",
                index=0,
                start=start_measure,
                end=start_measure + readout_lengths[0],
            )
            verifier.assert_pulse(
                signal="q0/acquire",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )

            start_measure = 2 * _LENGTH_GE + self._DELAYS[1][0]
            verifier.assert_pulse(
                signal="q1/measure",
                index=0,
                start=start_measure,
                end=start_measure + readout_lengths[1],
            )
            verifier.assert_pulse(
                signal="q1/acquire",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )
        elif transition == "ef":
            start_measure = _LENGTH_GE + 2 * _LENGTH_EF + self._DELAYS[0][0]
            verifier.assert_pulse(
                signal="q0/measure",
                index=0,
                start=start_measure,
                end=start_measure + readout_lengths[0],
            )
            verifier.assert_pulse(
                signal="q0/acquire",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )
            start_measure = _LENGTH_GE + 2 * _LENGTH_EF + self._DELAYS[1][0]
            verifier.assert_pulse(
                signal="q1/measure",
                index=0,
                start=start_measure,
                end=start_measure + readout_lengths[1],
            )
            verifier.assert_pulse(
                signal="q1/acquire",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
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
    options = ramsey.experiment_workflow.options()
    options.transition(transition)
    options.cal_states(cal_states)
    options.active_reset(True)
    options.active_reset_states(active_reset_states)
    options.active_reset_repetitions(active_reset_repetitions)
    options.do_analysis(False)
    [q0] = single_tunable_transmon_platform.qpu.qubits
    delays = np.linspace(0, 1e-6, 11)
    workflow_result = ramsey.experiment_workflow(
        session=single_tunable_transmon_platform.session(do_emulation=True),
        qubits=q0,
        qpu=single_tunable_transmon_platform.qpu,
        delays=delays,
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
        (len(delays), active_reset_repetitions)
        if active_reset_repetitions > 1
        else (len(delays),)
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
    options = ramsey.experiment_workflow.options()
    options.transition(transition)
    options.cal_states(cal_states)
    options.active_reset(True)
    options.active_reset_states(active_reset_states)
    options.active_reset_repetitions(active_reset_repetitions)
    options.do_analysis(False)
    qubits = two_tunable_transmon_platform.qpu.qubits
    delays = [np.linspace(0, 1e-6, 11), np.linspace(0, 10e-6, 11)]
    workflow_result = ramsey.experiment_workflow(
        session=two_tunable_transmon_platform.session(do_emulation=True),
        qubits=qubits,
        qpu=two_tunable_transmon_platform.qpu,
        delays=delays,
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
        (len(delays[0]), active_reset_repetitions)
        if active_reset_repetitions > 1
        else (len(delays[0]),)
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
    options = ramsey.experiment_workflow.options()
    options.averaging_mode("sequential")
    options.use_cal_traces(True)
    options.do_analysis(False)

    with pytest.raises(ValueError) as err:
        ramsey.experiment_workflow(
            session=session,
            qubits=q0,
            qpu=single_tunable_transmon_platform.qpu,
            delays=np.linspace(0, 10e-6, 10),
            detunings=0.67e6,
            options=options,
        ).run()

    assert str(err.value) == (
        "'AveragingMode.SEQUENTIAL' (or {AveragingMode.SEQUENTIAL}) cannot be used "
        "with calibration traces because the calibration traces are added "
        "outside the sweep."
    )
