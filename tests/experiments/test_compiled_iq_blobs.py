# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for the compiled iq_blobs experiment using the testing utilities
provided by the LabOne Q Applications Library.
"""

import numpy as np
import pytest

from laboneq_applications.experiments import (
    iq_blobs,
)
from laboneq_applications.testing import CompiledExperimentVerifier

_LENGTH_GE = 32e-9
_LENGTH_EF = 64e-9
_LENGTH_MEASURE = 2e-6
_LENGTH_MEASURE_RESET = 2e-6 + 1e-6


def create_iq_blobs_verifier(
    tunable_transmon_platform,
    count,
    states,
    readout_lengths=None,
):
    """Create a CompiledExperimentVerifier for the iq_blobs experiment."""
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
    options = iq_blobs.experiment_workflow.options()
    options.count(count)
    options.do_analysis(False)
    res = iq_blobs.experiment_workflow(
        session=session,
        qubits=qubits,
        qpu=tunable_transmon_platform.qpu,
        states=states,
        options=options,
    ).run()
    return CompiledExperimentVerifier(res.tasks["compile_experiment"].output)


@pytest.mark.parametrize(
    "states",
    [
        "ge",
        "gef",
    ],
)
@pytest.mark.parametrize(
    "count",
    [2, 4],
)
class TestIQBlobsSingleQubit:
    def test_pulse_count(
        self,
        single_tunable_transmon_platform,
        count,
        states,
    ):
        """Test the number of drive pulses."""

        verifier = create_iq_blobs_verifier(
            single_tunable_transmon_platform,
            count,
            states,
        )
        if "f" in states:
            expected_drive_count = count * (states.count("f"))
            verifier.assert_number_of_pulses(
                "q0/drive_ef",
                expected_drive_count,
            )
        if "e" in states:
            expected_drive_count = count * (states.count("e") + states.count("f"))
            verifier.assert_number_of_pulses(
                "q0/drive",
                expected_drive_count,
            )

        # Note that with cal_state on, there are 2 additional measure pulses
        expected_measure_count = count * (
            states.count("g") + states.count("e") + states.count("f")
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

    def test_pulse_drive(
        self,
        single_tunable_transmon_platform,
        count,
        states,
    ):
        """Test the properties of drive pulses."""

        verifier = create_iq_blobs_verifier(
            single_tunable_transmon_platform,
            count,
            states,
        )
        if states == "ge":
            verifier.assert_pulse(
                signal="q0/drive",
                index=0,
                start=_LENGTH_MEASURE_RESET,
                end=_LENGTH_MEASURE_RESET + _LENGTH_GE,
                parameterized_with=[],
            )
        elif states == "gef":
            verifier.assert_pulse(
                signal="q0/drive_ef",
                index=0,
                start=2 * _LENGTH_MEASURE_RESET + 2 * _LENGTH_GE,
                end=2 * _LENGTH_MEASURE_RESET + 2 * _LENGTH_GE + _LENGTH_EF,
                parameterized_with=[],
            )

    def test_pulse_measure(
        self,
        single_tunable_transmon_platform,
        count,
        states,
    ):
        """Test the properties of measure pulses.

        Here, we assert the start, end, and the parameterization of the pulses.

        """

        verifier = create_iq_blobs_verifier(
            single_tunable_transmon_platform,
            count,
            states,
        )

        verifier.assert_pulse(
            signal="q0/measure",
            index=0,
            start=0e-9,
            end=_LENGTH_MEASURE,
        )
        verifier.assert_pulse(
            signal="q0/acquire",
            index=0,
            start=0e-9,
            end=_LENGTH_MEASURE,
        )


@pytest.mark.parametrize(
    ("states", "readout_lengths"),
    [
        ("ge", [1e-6, 1e-6]),
        ("gef", [100e-9, 200e-9]),
    ],
)
@pytest.mark.parametrize(
    "count",
    [2, 4],
)
class TestIQBlobsTwoQubit:
    def test_pulse_count(
        self,
        two_tunable_transmon_platform,
        count,
        states,
        readout_lengths,
    ):
        """Test the number of drive pulses.

        `single_tunable_transmon` is a pytest fixture that is automatically
        imported into the test function.

        """

        verifier = create_iq_blobs_verifier(
            two_tunable_transmon_platform,
            count,
            states,
            readout_lengths,
        )

        # Note that with cal_state on, there is 1 additional drive pulse.
        if "f" in states:
            expected_drive_count = count * (states.count("f"))

            verifier.assert_number_of_pulses(
                "q0/drive_ef",
                expected_drive_count,
            )
            verifier.assert_number_of_pulses(
                "q1/drive_ef",
                expected_drive_count,
            )
        if "e" in states:
            expected_drive_count = count * (states.count("e") + states.count("f"))

            verifier.assert_number_of_pulses(
                "q0/drive",
                expected_drive_count,
            )
            verifier.assert_number_of_pulses(
                "q1/drive",
                expected_drive_count,
            )

        # Note that with cal_state on, there are 2 additional measure pulses
        expected_measure_count = count * (
            states.count("g") + states.count("e") + states.count("f")
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
        states,
        readout_lengths,
    ):
        """Test the properties of drive pulses."""

        verifier = create_iq_blobs_verifier(
            two_tunable_transmon_platform,
            count,
            states,
            readout_lengths,
        )
        if states == "ge":
            verifier.assert_pulse(
                signal="q0/drive",
                index=0,
                start=_LENGTH_MEASURE_RESET,
                end=_LENGTH_MEASURE_RESET + _LENGTH_GE,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="q1/drive",
                index=0,
                start=_LENGTH_MEASURE_RESET,
                end=_LENGTH_MEASURE_RESET + _LENGTH_GE,
                parameterized_with=[],
            )
        elif states == "gef":
            verifier.assert_pulse(
                signal="q0/drive_ef",
                index=0,
                start=2 * _LENGTH_MEASURE_RESET + 2 * _LENGTH_GE,
                end=2 * _LENGTH_MEASURE_RESET + 2 * _LENGTH_GE + _LENGTH_EF,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="q1/drive_ef",
                index=0,
                start=2 * _LENGTH_MEASURE_RESET + 2 * _LENGTH_GE,
                end=2 * _LENGTH_MEASURE_RESET + 2 * _LENGTH_GE + _LENGTH_EF,
                parameterized_with=[],
            )

    def test_pulse_measure(
        self,
        two_tunable_transmon_platform,
        count,
        states,
        readout_lengths,
    ):
        """Test the properties of measure pulses."""

        verifier = create_iq_blobs_verifier(
            two_tunable_transmon_platform,
            count,
            states,
            readout_lengths,
        )

        verifier.assert_pulse(
            signal="q0/measure",
            index=0,
            start=0e-9,
            end=readout_lengths[0],
        )
        verifier.assert_pulse(
            signal="q0/acquire",
            index=0,
            start=0e-9,
            end=_LENGTH_MEASURE,
        )
        verifier.assert_pulse(
            signal="q1/measure",
            index=0,
            start=0e-9,
            end=readout_lengths[1],
        )
        verifier.assert_pulse(
            signal="q1/acquire",
            index=0,
            start=0e-9,
            end=_LENGTH_MEASURE,
        )


@pytest.mark.parametrize(
    ("states", "active_reset_states"),
    [("ge", "ge"), ("ef", "gef"), ("gef", "gef")],
)
@pytest.mark.parametrize(
    "active_reset_repetitions",
    [1, 5],
)
def test_single_qubit_run_with_active_reset(
    single_tunable_transmon_platform,
    states,
    active_reset_states,
    active_reset_repetitions,
):
    options = iq_blobs.experiment_workflow.options()
    count = 1024
    options.count(count)
    options.active_reset(True)
    options.active_reset_states(active_reset_states)
    options.active_reset_repetitions(active_reset_repetitions)
    options.do_analysis(False)
    [q0] = single_tunable_transmon_platform.qpu.qubits
    workflow_result = iq_blobs.experiment_workflow(
        session=single_tunable_transmon_platform.session(do_emulation=True),
        qubits=q0,
        qpu=single_tunable_transmon_platform.qpu,
        states=states,
        options=options,
    ).run()

    exp = workflow_result.tasks["create_experiment"].output
    active_reset_section = exp.sections[0].children[0]
    assert active_reset_section.uid == "active_reset_q0_0"
    truth_len = len(q0.signals) + active_reset_repetitions
    assert len(active_reset_section.children) == truth_len

    data = workflow_result.output
    assert "active_reset" in data.q0
    for s in states:
        cal_trace_data = data.q0.active_reset.cal_trace[s].data
        assert (
            np.shape(cal_trace_data) == (count,)
            if active_reset_repetitions == 1
            else (count, active_reset_repetitions)
        )


@pytest.mark.parametrize(
    ("states", "active_reset_states"),
    [("ge", "ge"), ("ef", "gef"), ("gef", "gef")],
)
@pytest.mark.parametrize(
    "active_reset_repetitions",
    [1, 5],
)
def test_two_qubit_run_with_active_reset(
    two_tunable_transmon_platform,
    states,
    active_reset_states,
    active_reset_repetitions,
):
    options = iq_blobs.experiment_workflow.options()
    count = 1024
    options.count(count)
    options.active_reset(True)
    options.active_reset_states(active_reset_states)
    options.active_reset_repetitions(active_reset_repetitions)
    options.do_analysis(False)
    qubits = two_tunable_transmon_platform.qpu.qubits
    workflow_result = iq_blobs.experiment_workflow(
        session=two_tunable_transmon_platform.session(do_emulation=True),
        qubits=qubits,
        qpu=two_tunable_transmon_platform.qpu,
        states=states,
        options=options,
    ).run()

    exp = workflow_result.tasks["create_experiment"].output
    active_reset_section = exp.sections[0].children[0]
    assert active_reset_section.uid == "active_reset_q0_q1_0"
    truth_len = len(qubits[0].signals) * len(qubits) + active_reset_repetitions
    assert len(active_reset_section.children) == truth_len

    data = workflow_result.output
    assert "active_reset" in data.q0
    for s in states:
        for q in qubits:
            cal_trace_data = data[q.uid].active_reset.cal_trace[s].data
            assert (
                np.shape(cal_trace_data) == (count,)
                if active_reset_repetitions == 1
                else (count, active_reset_repetitions)
            )
