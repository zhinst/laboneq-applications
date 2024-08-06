"""Tests for the compiled amplitude_rabi experiment using the testing utilities
provided by the LabOne Q Applications Library.

Furthermore, the tests in this module serve as reference examples for future
contributions to the library. To add tests for a newly added experiment, please
follow these steps:

  - Verify that your experiment is correct by validating it on actual hardware
  - Read the reference documentation of the testing utilities for more information
  - Write a `create_<YOUR_EXPERIMENT_NAME>_verifier` function following the steps
    in `create_rabi_verifier`
  - Write a `Test<YOUR_EXPERIMENT_NAME><USE_CASE>` class following
    `TestAmplitudeRabiSingleQubit`
  - Add `pytest.mark.parametrize` decorators to this class to define individual
    test runs. Tests will be defined for all combination of parameters provided.
    Focus your tests on the most relevant parameter ranges, to keep the runtime
    of tests manageable.
"""

import pytest

from laboneq_applications.experiments import amplitude_rabi
from laboneq_applications.testing import CompiledExperimentVerifier


def create_rabi_verifier(
    tunable_transmons,
    amplitudes,
    count,
    transition,
    use_cal_traces,
):
    """Create a CompiledExperimentVerifier for the amplitude rabi experiment."""
    qubits = tunable_transmons.qubits
    if len(qubits) == 1:
        qubits = qubits[0]
    session = tunable_transmons.session(do_emulation=True)
    options = amplitude_rabi.options()
    options.create_experiment.count = count
    options.create_experiment.transition = transition
    options.create_experiment.use_cal_traces = use_cal_traces
    res = amplitude_rabi.experiment_workflow(
        session=session,
        qubits=qubits,
        qpu=tunable_transmons,
        amplitudes=amplitudes,
        options=options,
    ).run()
    return CompiledExperimentVerifier(res.tasks["compile_experiment"].output)


# use pytest.mark.parametrize to generate test cases for
# all combinations of the parameters.
@pytest.mark.parametrize(
    "amplitudes",
    [
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3, 0.4],
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
class TestAmplitudeRabiSingleQubit:
    def test_pulse_count_drive(
        self,
        single_tunable_transmon,
        amplitudes,
        count,
        transition,
        use_cal_traces,
    ):
        """Test the number of drive pulses.

        `single_tunable_transmon` is a pytest fixture and automatically
        imported into the test function.

        """
        # create a verifier for the experiment
        verifier = create_rabi_verifier(
            single_tunable_transmon,
            amplitudes,
            count,
            transition,
            use_cal_traces,
        )

        # with cal_state on, there is 1 additional drive pulse
        # The signal names can be looked up in device_setup
        # but typically they are in the form of
        # /logical_signal_groups/q{i}/drive(measure/acquire/drive_ef)
        expected_drive_count = count * (len(amplitudes) + int(use_cal_traces))
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive",
            expected_drive_count,
        )

        if transition == "ef":
            expected_drive_count = count * len(amplitudes)
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive_ef",
                expected_drive_count,
            )

    def test_pulse_count_measure_acquire(
        self,
        single_tunable_transmon,
        amplitudes,
        count,
        transition,
        use_cal_traces,
    ):
        """Test the number of measure and acquire pulses."""
        verifier = create_rabi_verifier(
            single_tunable_transmon,
            amplitudes,
            count,
            transition,
            use_cal_traces,
        )
        # with cal_state on, there are 2 additional measure pulses
        expected_measure_count = count * (len(amplitudes) + 2 * int(use_cal_traces))
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
        single_tunable_transmon,
        amplitudes,
        count,
        transition,
        use_cal_traces,
    ):
        """Test the properties of drive pulses."""
        # Here, we can assert the start, end, and the parameterization of the pulses.
        # The parameterization is the list of parameters that are used to parameterize.
        # If the pulse is not parameterized, the list should be empty.
        # The name of the parameter should match with the uid of SweepParameter
        # in the experiment.

        verifier = create_rabi_verifier(
            single_tunable_transmon,
            amplitudes,
            count,
            transition,
            use_cal_traces,
        )
        if transition == "ge":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=0,
                start=0e-6,
                end=51e-9,
                parameterized_with=["amplitude_q0"],
            )
        elif transition == "ef":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=0,
                start=56e-9,
                end=56e-9 + 52e-9,
                parameterized_with=["amplitude_q0"],
            )

    def test_pulse_measure(
        self,
        single_tunable_transmon,
        amplitudes,
        count,
        transition,
        use_cal_traces,
    ):
        """Test the properties of measure pulses.

        Here, we can assert the start, end, and the parameterization of the pulses.

        """
        verifier = create_rabi_verifier(
            single_tunable_transmon,
            amplitudes,
            count,
            transition,
            use_cal_traces,
        )

        if transition == "ge":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start=56e-9,
                end=56e-9 + 2e-6,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start=56e-9,
                end=56e-9 + 2e-6,
            )
        elif transition == "ef":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start=112e-9,
                end=112e-9 + 2e-6,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start=112e-9,
                end=112e-9 + 2e-6,
            )


# use pytest.mark.parametrize to generate test cases for
# all combinations of the parameters.
@pytest.mark.parametrize(
    "amplitudes",
    [
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
        [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]],
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
class TestAmplitudeRabiTwoQubits:
    def test_pulse_count_drive(
        self,
        two_tunable_transmon,
        amplitudes,
        count,
        transition,
        use_cal_traces,
    ):
        """Test the number of drive pulses.

        `two_tunable_transmon` is a pytest fixture and automatically
        imported into the test function.

        """
        # create a verifier for the experiment
        verifier = create_rabi_verifier(
            two_tunable_transmon,
            amplitudes,
            count,
            transition,
            use_cal_traces,
        )

        # with cal_state on, there is 1 additional drive pulse
        # The signal names can be looked up in device_setup
        # but typically they are in the form of
        # /logical_signal_groups/q{i}/drive(measure/acquire/drive_ef)
        expected_drive_count = count * (len(amplitudes[0]) + int(use_cal_traces))
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive",
            expected_drive_count,
        )

        expected_drive_count = count * (len(amplitudes[1]) + int(use_cal_traces))
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q1/drive",
            expected_drive_count,
        )

        if transition == "ef":
            expected_drive_count = count * len(amplitudes[0])
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive_ef",
                expected_drive_count,
            )

            expected_drive_count = count * len(amplitudes[1])
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q1/drive_ef",
                expected_drive_count,
            )

    def test_pulse_count_measure_acquire(
        self,
        two_tunable_transmon,
        amplitudes,
        count,
        transition,
        use_cal_traces,
    ):
        """Test the number of measure and acquire pulses."""
        verifier = create_rabi_verifier(
            two_tunable_transmon,
            amplitudes,
            count,
            transition,
            use_cal_traces,
        )
        # with cal_state on, there are 2 additional measure pulses
        expected_measure_count = count * (len(amplitudes[0]) + 2 * int(use_cal_traces))
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
        two_tunable_transmon,
        amplitudes,
        count,
        transition,
        use_cal_traces,
    ):
        """Test the properties of drive pulses."""
        # Here, we can assert the start, end, and the parameterization of the pulses.
        # The parameterization is the list of parameters that are used to parameterize.
        # If the pulse is not parameterized, the list should be empty.
        # The name of the parameter should match with the uid of SweepParameter
        # in the experiment.

        verifier = create_rabi_verifier(
            two_tunable_transmon,
            amplitudes,
            count,
            transition,
            use_cal_traces,
        )
        if transition == "ge":
            # Here, we give an example of verifying the first drive pulse of q0
            # More pulses should be tested in a similar way
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=0,
                start=0e-6,
                end=51e-9,
                parameterized_with=["amplitude_q0"],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive",
                index=0,
                start=0e-6,
                end=51e-9,
                parameterized_with=["amplitude_q1"],
            )
        elif transition == "ef":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=0,
                start=56e-9,
                end=56e-9 + 52e-9,
                parameterized_with=["amplitude_q0"],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive_ef",
                index=0,
                start=56e-9,
                end=56e-9 + 52e-9,
                parameterized_with=["amplitude_q1"],
            )

    def test_pulse_measure(
        self,
        two_tunable_transmon,
        amplitudes,
        count,
        transition,
        use_cal_traces,
    ):
        """Test the properties of measure pulses.

        Here, we can assert the start, end, and the parameterization of the pulses.

        """
        verifier = create_rabi_verifier(
            two_tunable_transmon,
            amplitudes,
            count,
            transition,
            use_cal_traces,
        )

        if transition == "ge":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start=56e-9,
                end=56e-9 + 2e-6,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start=56e-9,
                end=56e-9 + 2e-6,
            )

            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/measure",
                index=0,
                start=56e-9,
                end=56e-9 + 2e-6,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/acquire",
                index=0,
                start=56e-9,
                end=56e-9 + 2e-6,
            )
        elif transition == "ef":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start=112e-9,
                end=112e-9 + 2e-6,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start=112e-9,
                end=112e-9 + 2e-6,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/measure",
                index=0,
                start=112e-9,
                end=112e-9 + 2e-6,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/acquire",
                index=0,
                start=112e-9,
                end=112e-9 + 2e-6,
            )
