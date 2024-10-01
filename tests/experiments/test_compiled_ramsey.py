"""Tests for the compiled ramsey experiment using the testing utilities
provided by the LabOne Q Applications Library.
"""

import pytest

from laboneq_applications.experiments import ramsey
from laboneq_applications.testing import CompiledExperimentVerifier


def create_ramsey_verifier(
    tunable_transmon_platform, delays, count, transition, use_cal_traces, detunings
):
    """Create a CompiledExperimentVerifier for the ramsey experiment."""
    qubits = tunable_transmon_platform.qpu.qubits
    if len(qubits) == 1:
        qubits = qubits[0]
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
        detunings=detunings,
        options=options,
    ).run()
    return CompiledExperimentVerifier(res.tasks["compile_experiment"].output)


@pytest.mark.parametrize(
    "delays",
    [
        [0.1e-6, 0.2e-6, 0.3e-6, 0.4e-6, 0.5e-6, 0.6e-6, 0.7e-6, 0.8e-6, 0.9e-6, 1e-6],
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
    "detunings",
    [None],
)
class TestRamseySingleQubit:
    def test_pulse_count_drive(
        self,
        single_tunable_transmon_platform,
        delays,
        count,
        transition,
        use_cal_traces,
        detunings,
    ):
        """Test the number of drive pulses.

        `single_tunable_transmon_platform` is a pytest fixture and automatically
        imported into the test function.

        """
        verifier = create_ramsey_verifier(
            single_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            detunings,
        )

        # with cal_state on, there is 1 additional drive pulse
        if transition == "ge":
            expected_drive_count = count * (2 * len(delays) + int(use_cal_traces))
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive",
                expected_drive_count,
            )

        if transition == "ef":
            expected_drive_count = count * 2 * len(delays)
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
        detunings,
    ):
        """Test the number of measure and acquire pulses."""
        verifier = create_ramsey_verifier(
            single_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            detunings,
        )
        # with cal_state on, there are 2 additional measure pulses
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
        detunings,
    ):
        """Test the properties of drive pulses."""
        [q0] = single_tunable_transmon_platform.qpu.qubits
        q0_pulse_length_ge = q0.parameters.ge_drive_length
        q0_pulse_length_ef = q0.parameters.ef_drive_length
        verifier = create_ramsey_verifier(
            single_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            detunings,
        )
        if transition == "ge":
            offset = 6e-9
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=0,
                start=offset,
                end=offset + q0_pulse_length_ge,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=1,
                start=offset + q0_pulse_length_ge + delays[0],
                end=offset + 2 * q0_pulse_length_ge + delays[0],
                parameterized_with=["x90_phases_q0"],
            )
        elif transition == "ef":
            offset = 5e-9
            start_ef = offset + q0_pulse_length_ge
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=0,
                start=start_ef,
                end=start_ef + q0_pulse_length_ef,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=1,
                start=start_ef + q0_pulse_length_ef + delays[0],
                end=start_ef + 2 * q0_pulse_length_ef + delays[0],
                parameterized_with=["x90_phases_q0"],
            )

    def test_pulse_measure(
        self,
        single_tunable_transmon_platform,
        delays,
        count,
        transition,
        use_cal_traces,
        detunings,
    ):
        """Test the properties of measure pulses."""
        verifier = create_ramsey_verifier(
            single_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            detunings,
        )
        # The starting of measure pulses depends
        # on subsequent drive pulses and "jumps"
        # that happen when aligning to system_grid
        # hardcoded it here

        if transition == "ge":
            start_measure = 208e-9
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )
        elif transition == "ef":
            start_measure = 264e-9
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )


@pytest.mark.parametrize(
    "delays",
    [
        [
            [
                0.1e-6,
                0.2e-6,
                0.3e-6,
                0.4e-6,
                0.5e-6,
                0.6e-6,
                0.7e-6,
                0.8e-6,
                0.9e-6,
                1e-6,
            ],
            [
                0.1e-6,
                0.2e-6,
                0.3e-6,
                0.4e-6,
                0.5e-6,
                0.6e-6,
                0.7e-6,
                0.8e-6,
                0.9e-6,
                1e-6,
            ],
        ],
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
    "detunings",
    [None],
)
class TestRamseyTwoQubits:
    def test_pulse_count_drive(
        self,
        two_tunable_transmon_platform,
        delays,
        count,
        transition,
        use_cal_traces,
        detunings,
    ):
        """Test the number of drive pulses."""
        verifier = create_ramsey_verifier(
            two_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            detunings,
        )

        # with cal_state on, there is 1 additional drive pulse
        if transition == "ge":
            expected_drive_count = count * (2 * len(delays[0]) + int(use_cal_traces))
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive",
                expected_drive_count,
            )

            expected_drive_count = count * (2 * len(delays[1]) + int(use_cal_traces))
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q1/drive",
                expected_drive_count,
            )

        if transition == "ef":
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

            expected_drive_count = count * (2 * len(delays[0]))
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive_ef",
                expected_drive_count,
            )

            expected_drive_count = count * (2 * len(delays[1]))
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
        detunings,
    ):
        """Test the number of measure and acquire pulses."""
        verifier = create_ramsey_verifier(
            two_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            detunings,
        )
        # with cal_state on, there are 2 additional measure pulses
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
        detunings,
    ):
        """Test the properties of drive pulses."""
        [q0, q1] = two_tunable_transmon_platform.qpu.qubits
        q0_pulse_length_ge = q0.parameters.ge_drive_length
        q0_pulse_length_ef = q0.parameters.ef_drive_length
        q1_pulse_length_ge = q1.parameters.ge_drive_length
        q1_pulse_length_ef = q1.parameters.ef_drive_length

        verifier = create_ramsey_verifier(
            two_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            detunings,
        )
        if transition == "ge":
            # Offset at the beginning of experiment is not crucial.
            offset = 6e-9
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=0,
                end=offset + q0_pulse_length_ge,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=1,
                start=offset + q0_pulse_length_ge + delays[0][0],
                end=offset + 2 * q0_pulse_length_ge + delays[0][0],
                parameterized_with=["x90_phases_q0"],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive",
                index=0,
                end=offset + q1_pulse_length_ge,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive",
                index=1,
                start=offset + q1_pulse_length_ge + delays[0][0],
                end=offset + 2 * q1_pulse_length_ge + delays[0][0],
                parameterized_with=["x90_phases_q1"],
            )
        elif transition == "ef":
            offset = 5e-9
            start_ef_q0 = offset + q0_pulse_length_ge
            start_ef_q1 = offset + q1_pulse_length_ge
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=0,
                end=start_ef_q0 + q0_pulse_length_ef,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=1,
                start=start_ef_q0 + q0_pulse_length_ef + delays[0][0],
                end=start_ef_q0 + 2 * q0_pulse_length_ef + delays[0][0],
                parameterized_with=["x90_phases_q0"],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive_ef",
                index=0,
                start=start_ef_q1,
                end=start_ef_q1 + q1_pulse_length_ef,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive_ef",
                index=1,
                start=start_ef_q1 + q1_pulse_length_ef + delays[1][0],
                end=start_ef_q1 + 2 * q1_pulse_length_ef + delays[0][0],
                parameterized_with=["x90_phases_q1"],
            )

    def test_pulse_measure(
        self,
        two_tunable_transmon_platform,
        delays,
        count,
        transition,
        use_cal_traces,
        detunings,
    ):
        """Test the properties of measure pulses.

        Here, we can assert the start, end, and the parameterization of the pulses.

        """
        verifier = create_ramsey_verifier(
            two_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            detunings,
        )

        if transition == "ge":
            # See the explanation for the hardcoding of start_measure
            # in the single-qubit tests
            start_measure = 208e-9
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )

            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/measure",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/acquire",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )
        elif transition == "ef":
            start_measure = 264e-9
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/measure",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/acquire",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )
