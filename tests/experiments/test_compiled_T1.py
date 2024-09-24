"""Tests for the compiled T1 experiment using the testing utilities
provided by the LabOne Q Applications Library.
"""  # noqa: N999

import pytest

from laboneq_applications.experiments import (
    T1,
)
from laboneq_applications.testing import CompiledExperimentVerifier


def create_T1_verifier(  # noqa: N802
    tunable_transmon_platform,
    delays,
    count,
    transition,
    use_cal_traces,
    cal_states,
):
    """Create a CompiledExperimentVerifier for the T1 experiment."""
    qubits = tunable_transmon_platform.qpu.qubits
    if len(qubits) == 1:
        qubits = qubits[0]
    session = tunable_transmon_platform.session(do_emulation=True)
    options = T1.experiment_workflow.options()
    options.count(count)
    options.transition(transition)
    options.use_cal_traces(use_cal_traces)
    options.cal_states(cal_states)
    res = T1.experiment_workflow(
        session=session,
        qubits=qubits,
        qpu=tunable_transmon_platform.qpu,
        delays=delays,
        options=options,
    ).run()
    return CompiledExperimentVerifier(res.tasks["compile_experiment"].output)


@pytest.mark.parametrize(
    "delays",
    [
        [56e-9, 112e-9, 224e-9],
        [56e-9, 112e-9, 224e-9, 448e-9],
    ],
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
@pytest.mark.parametrize(
    "cal_states",
    ["ge"],
)
class TestT1SingleQubit:
    def test_pulse_count_drive(
        self,
        single_tunable_transmon_platform,
        delays,
        count,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the number of drive pulses.

        `single_tunable_transmon` is a pytest fixture that is automatically
        imported into the test function.

        """
        verifier = create_T1_verifier(
            single_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,
        )

        expected_drive_count = count * (len(delays) + int(use_cal_traces))
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive",
            expected_drive_count,
        )

        if transition == "ef":
            expected_drive_count = count * len(delays)
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive_ef",
                expected_drive_count,
            )

    def test_pulse_count_measure_acquire(
        self,
        single_tunable_transmon_platform,
        delays,
        count,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the number of measure and acquire pulses."""
        verifier = create_T1_verifier(
            single_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,
        )
        # Note that with cal_state on, there are 2 additional measure pulses
        expected_measure_count = count * (len(delays) + 2 * int(use_cal_traces))
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
        delays,
        count,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the properties of drive pulses."""

        verifier = create_T1_verifier(
            single_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,
        )
        if transition == "ge":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=0,
                start=0e-6,
                end=51e-9,
            )
        elif transition == "ef":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=0,
                start=56e-9,
                end=56e-9 + 52e-9,
            )

    def test_pulse_measure(
        self,
        single_tunable_transmon_platform,
        delays,
        count,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the properties of measure pulses.

        Here, we assert the start, end, and the parameterization of the pulses.

        """

        verifier = create_T1_verifier(
            single_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,
        )

        if transition == "ge":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start=56e-9 + 56e-9,
                end=56e-9 + 2e-6 + 56e-9,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start=56e-9 + 56e-9,
                end=56e-9 + 2e-6 + 56e-9,
            )
        elif transition == "ef":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start=112e-9 + 56e-9,
                end=112e-9 + 2e-6 + 56e-9,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start=112e-9 + 56e-9,
                end=112e-9 + 2e-6 + 56e-9,
            )


@pytest.mark.parametrize(
    "delays",
    [
        [[56e-9, 112e-9, 224e-9], [56e-9, 112e-9, 224e-9]],
        [[56e-9, 112e-9, 224e-9, 448e-9], [56e-9, 112e-9, 224e-9, 448e-9]],
    ],
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
@pytest.mark.parametrize(
    "cal_states",
    ["ge"],
)
class TestT1TwoQubits:
    def test_pulse_count_drive(
        self,
        two_tunable_transmon_platform,
        delays,
        count,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the number of drive pulses.

        `two_tunable_transmon` is a pytest fixture that is automatically
        imported into the test function.

        """
        # create a verifier for the experiment

        verifier = create_T1_verifier(
            two_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,
        )
        expected_drive_count = count * (len(delays[0]) + int(use_cal_traces))
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive",
            expected_drive_count,
        )

        expected_drive_count = count * (len(delays[1]) + int(use_cal_traces))
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q1/drive",
            expected_drive_count,
        )

        if transition == "ef":
            # Check for q0
            expected_drive_count = count * len(delays[0])

            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive_ef",
                expected_drive_count,
            )

            # Check for q1
            expected_drive_count = count * len(delays[1])

            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q1/drive_ef",
                expected_drive_count,
            )

    def test_pulse_count_measure_acquire(
        self,
        two_tunable_transmon_platform,
        delays,
        count,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the number of measure and acquire pulses."""

        verifier = create_T1_verifier(
            two_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,
        )
        # Check for q0
        # Note that with cal_state on, there are 2 additional measure pulses
        expected_measure_count = count * (len(delays[0]) + 2 * int(use_cal_traces))

        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/measure",
            expected_measure_count,
        )

        # acquire and measure pulses have the same count
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/acquire",
            expected_measure_count,
        )

        # Check for q1
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
        delays,
        count,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the properties of drive pulses."""
        verifier = create_T1_verifier(
            two_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,
        )
        if transition == "ge":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=0,
                start=0e-6,
                end=51e-9,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive",
                index=0,
                start=0e-6,
                end=51e-9,
                parameterized_with=[],
            )
        elif transition == "ef":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=0,
                start=56e-9,
                end=56e-9 + 52e-9,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive_ef",
                index=0,
                start=56e-9,
                end=56e-9 + 52e-9,
                parameterized_with=[],
            )

    def test_pulse_measure(
        self,
        two_tunable_transmon_platform,
        delays,
        count,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the properties of measure pulses.

        Here, we can assert the start, end, and the parameterization of the pulses.

        """
        verifier = create_T1_verifier(
            two_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,
        )

        if transition == "ge":
            # Check for q0
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start=56e-9 + 56e-9,
                end=56e-9 + 2e-6 + 56e-9,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start=56e-9 + 56e-9,
                end=56e-9 + 2e-6 + 56e-9,
            )

            # Check for q1
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/measure",
                index=0,
                start=56e-9 + 56e-9,
                end=56e-9 + 2e-6 + 56e-9,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/acquire",
                index=0,
                start=56e-9 + 56e-9,
                end=56e-9 + 2e-6 + 56e-9,
            )
        elif transition == "ef":
            # Check for q0
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start=112e-9 + 56e-9,
                end=112e-9 + 2e-6 + 56e-9,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start=112e-9 + 56e-9,
                end=112e-9 + 2e-6 + 56e-9,
            )
            # Check for q1
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/measure",
                index=0,
                start=112e-9 + 56e-9,
                end=112e-9 + 2e-6 + 56e-9,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/acquire",
                index=0,
                start=112e-9 + 56e-9,
                end=112e-9 + 2e-6 + 56e-9,
            )
