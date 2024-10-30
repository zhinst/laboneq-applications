"""Tests for the compiled iq_blobs experiment using the testing utilities
provided by the LabOne Q Applications Library.
"""

import pytest

from laboneq_applications.experiments import (
    iq_blobs,
)
from laboneq_applications.testing import CompiledExperimentVerifier


def create_iq_blobs_verifier(
    tunable_transmon_platform,
    count,
    states,
    readout_lengths=None,
):
    """Create a CompiledExperimentVerifier for the iq_blobs experiment."""
    qubits = tunable_transmon_platform.qpu.qubits
    if len(qubits) == 1:
        qubits = qubits[0]
    if readout_lengths is not None:
        assert len(readout_lengths) == len(qubits)
        for i, rl in enumerate(readout_lengths):
            qubits[i].parameters.readout_length = rl
    session = tunable_transmon_platform.session(do_emulation=True)
    options = iq_blobs.experiment_workflow.options()
    options.count(count)
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
                "/logical_signal_groups/q0/drive_ef",
                expected_drive_count,
            )
        if "e" in states:
            expected_drive_count = count * (states.count("e") + states.count("f"))
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive",
                expected_drive_count,
            )

        # Note that with cal_state on, there are 2 additional measure pulses
        expected_measure_count = count * (
            states.count("g") + states.count("e") + states.count("f")
        )

        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/measure",
            expected_measure_count,
        )

        # acquire and measure pulses have the same count
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/acquire",
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
                signal="/logical_signal_groups/q0/drive",
                index=0,
                start=2e-6 + 1e-6,
                end=2e-6 + 1e-6 + 51e-9,
                parameterized_with=[],
            )
        elif states == "gef":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=0,
                start=2e-6 + 1e-6 + 56e-9 + 2e-6 + 1e-6 + 56e-9,
                end=2e-6 + 1e-6 + 56e-9 + 2e-6 + 1e-6 + 56e-9 + 52e-9,
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
            signal="/logical_signal_groups/q0/measure",
            index=0,
            start=0e-9,
            end=2e-6,
        )
        verifier.assert_pulse(
            signal="/logical_signal_groups/q0/acquire",
            index=0,
            start=0e-9,
            end=2e-6,
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
                "/logical_signal_groups/q0/drive_ef",
                expected_drive_count,
            )
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q1/drive_ef",
                expected_drive_count,
            )
        if "e" in states:
            expected_drive_count = count * (states.count("e") + states.count("f"))

            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive",
                expected_drive_count,
            )
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q1/drive",
                expected_drive_count,
            )

        # Note that with cal_state on, there are 2 additional measure pulses
        expected_measure_count = count * (
            states.count("g") + states.count("e") + states.count("f")
        )

        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/measure",
            expected_measure_count,
        )

        # acquire and measure pulses have the same count
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/acquire",
            expected_measure_count,
        )

        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q1/measure",
            expected_measure_count,
        )

        # acquire and measure pulses have the same count
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q1/acquire",
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
                signal="/logical_signal_groups/q0/drive",
                index=0,
                start=2e-6 + 1e-6,
                end=2e-6 + 1e-6 + 51e-9,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive",
                index=0,
                start=2e-6 + 1e-6,
                end=2e-6 + 1e-6 + 51e-9,
                parameterized_with=[],
            )
        elif states == "gef":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=0,
                start=2e-6 + 1e-6 + 56e-9 + 2e-6 + 1e-6 + 56e-9,
                end=2e-6 + 1e-6 + 56e-9 + 2e-6 + 1e-6 + 56e-9 + 52e-9,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive_ef",
                index=0,
                start=2e-6 + 1e-6 + 56e-9 + 2e-6 + 1e-6 + 56e-9,
                end=2e-6 + 1e-6 + 56e-9 + 2e-6 + 1e-6 + 56e-9 + 52e-9,
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
            signal="/logical_signal_groups/q0/measure",
            index=0,
            start=0e-9,
            end=readout_lengths[0],
        )
        verifier.assert_pulse(
            signal="/logical_signal_groups/q0/acquire",
            index=0,
            start=0e-9,
            end=2e-6,
        )
        verifier.assert_pulse(
            signal="/logical_signal_groups/q1/measure",
            index=0,
            start=0e-9,
            end=readout_lengths[1],
        )
        verifier.assert_pulse(
            signal="/logical_signal_groups/q1/acquire",
            index=0,
            start=0e-9,
            end=2e-6,
        )
