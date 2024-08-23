"""Tests for the compiled T1 experiment using the testing utilities
provided by the LabOne Q Applications Library.
"""  # noqa: N999

import pytest

from laboneq_applications.experiments import (
    T1,
)

# For new experiments: import the relevant experiment
from laboneq_applications.testing import CompiledExperimentVerifier


# For new experiments: replace rabi with the name of the new experiment
def create_T1_verifier(  # noqa: N802
    tunable_transmon_platform,
    delays,
    count,
    transition,  # For new experiments: use if relevant, or remove
    use_cal_traces,
    cal_states,  # For new experiments: use if relevant, or remove
    # For new experiments: add more arguments here if needed
):
    """Create a CompiledExperimentVerifier for the T1 experiment."""
    qubits = tunable_transmon_platform.qpu.qubits
    if len(qubits) == 1:
        qubits = qubits[0]
    session = tunable_transmon_platform.session(do_emulation=True)
    options = T1.options()
    options.create_experiment.count = count
    # For new experiments: use the lines below or remove if not needed, and add
    # new ones for any additional input parameters you might have added
    options.create_experiment.transition = transition
    options.create_experiment.use_cal_traces = use_cal_traces
    options.create_experiment.cal_states = cal_states
    # Run the experiment workflow
    # For new experiments: use the correct experiment name
    res = T1.experiment_workflow(
        session=session,
        qubits=qubits,
        qpu=tunable_transmon_platform.qpu,
        delays=delays,
        options=options,
    ).run()
    return CompiledExperimentVerifier(res.tasks["compile_experiment"].output)


### Single-Qubit Tests ###


# use pytest.mark.parametrize to generate test cases for
# all combinations of the parameters.
@pytest.mark.parametrize(  # For new experiments: replace with relevant name and values
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
@pytest.mark.parametrize(  # For new experiments: keep or remove as needed
    "transition",
    ["ge", "ef"],
)
@pytest.mark.parametrize(  # For new experiments: keep or remove as needed
    "use_cal_traces",
    [True, False],
)
# For new experiments: add more parameterizations for other parameters in your
# experiment over the values of which you think it makes sense to iterate in the tests.
@pytest.mark.parametrize(  # For new experiments: keep or remove as needed
    "cal_states",
    ["ge"],
)
# For new experiments: change AmplitudeRabi in the class name below to the name of
# your experiment
class TestT1SingleQubit:
    def test_pulse_count_drive(
        self,
        single_tunable_transmon_platform,
        delays,
        count,
        transition,  # For new experiments: use if relevant, or remove
        use_cal_traces,
        cal_states,  # For new experiments: use if relevant, or remove
        # For new experiments: add more arguments here if needed
    ):
        """Test the number of drive pulses.

        `single_tunable_transmon` is a pytest fixture that is automatically
        imported into the test function.

        """
        # create a verifier for the experiment
        # For new experiments: replace rabi with the name of your experiment
        verifier = create_T1_verifier(
            single_tunable_transmon_platform,
            delays,
            count,
            transition,  # For new experiments: use if relevant, or remove
            use_cal_traces,
            cal_states,  # For new experiments: use if relevant, or remove
            # For new experiments: add more arguments here if needed
        )

        # The signal names can be looked up in device_setup,
        # but typically they are in the form of
        # /logical_signal_groups/q{i}/drive(measure/acquire/drive_ef)
        # Note that with cal_state on, there is 1 additional drive pulse.
        expected_drive_count = count * (len(delays) + int(use_cal_traces))
        # For new experiments: change the line above as needed
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive",
            expected_drive_count,
        )

        if transition == "ef":  # For new experiments: remove if not relevant
            expected_drive_count = count * len(delays)
            # For new experiments: change the line above as needed
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive_ef",
                expected_drive_count,
            )

    def test_pulse_count_measure_acquire(
        self,
        single_tunable_transmon_platform,
        delays,
        count,
        transition,  # For new experiments: use if relevant, or remove
        use_cal_traces,
        cal_states,  # For new experiments: use if relevant, or remove
        # For new experiments: add more arguments here if needed
    ):
        """Test the number of measure and acquire pulses."""
        # For new experiments: replace rabi with the name of your experiment
        verifier = create_T1_verifier(
            single_tunable_transmon_platform,
            delays,
            count,
            transition,  # For new experiments: use if relevant, or remove
            use_cal_traces,
            cal_states,  # For new experiments: use if relevant, or remove
            # For new experiments: add more arguments here if needed
        )
        # Note that with cal_state on, there are 2 additional measure pulses
        expected_measure_count = count * (len(delays) + 2 * int(use_cal_traces))
        # For new experiments: change the line above as needed
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
        transition,  # For new experiments: use if relevant, or remove
        use_cal_traces,
        cal_states,  # For new experiments: use if relevant, or remove
        # For new experiments: add more arguments here if needed
    ):
        """Test the properties of drive pulses."""
        # Here, we can assert the start, end, and the parameterization of the pulses.
        # In the function `assert_pulse` below, index is the index of the pulse in the
        # pulse sequence, and `parameterized_with` is the list of SweepParameter names
        # used for that pulse. The name of the parameter should
        # match with the uid of SweepParameter in the experiment.
        # If none of the pulse parameters are swept, the list should be empty.

        # In this test, all the qubit ge drive pulses have lengths of 51ns,
        # and all the ef pulses have lengths of 52ns.

        # For new experiments: replace rabi with the name of your experiment
        verifier = create_T1_verifier(
            single_tunable_transmon_platform,
            delays,
            count,
            transition,  # For new experiments: use if relevant, or remove
            use_cal_traces,
            cal_states,  # For new experiments: use if relevant, or remove
            # For new experiments: add more arguments here if needed
        )
        if transition == "ge":
            # ge pulses
            # Here, we give an example of verifying the first drive pulse of q0
            # More pulses should be tested in a similar way
            # For new experiments: Please write tests for more than one pulse,
            # if applicable (one call to verifier.assert_pulse for every pulse)
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=0,  # For new experiments: change this value as needed
                start=0e-6,  # For new experiments: change this value as needed
                end=51e-9,  # For new experiments: change this value as needed
            )
        elif transition == "ef":
            # For new experiments: remove if not relevant
            # For new experiments: Please write tests for more than one pulse,
            # if applicable (one call to verifier.assert_pulse for every pulse)
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=0,  # For new experiments: change this value as needed
                start=56e-9,  # For new experiments: change this value as needed
                end=56e-9 + 52e-9,  # For new experiments: change this value as needed
            )

    def test_pulse_measure(
        self,
        single_tunable_transmon_platform,
        delays,
        count,
        transition,  # For new experiments: use if relevant, or remove
        use_cal_traces,
        cal_states,  # For new experiments: use if relevant, or remove
        # For new experiments: add more arguments here if needed
    ):
        """Test the properties of measure pulses.

        Here, we assert the start, end, and the parameterization of the pulses.

        """
        # For new experiments: replace rabi with the name of your experiment
        verifier = create_T1_verifier(
            single_tunable_transmon_platform,
            delays,
            count,
            transition,  # For new experiments: use if relevant, or remove
            use_cal_traces,
            cal_states,  # For new experiments: use if relevant, or remove
            # For new experiments: add more arguments here if needed
        )

        if transition == "ge":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,  # For new experiments: change this value as needed
                start=56e-9 + 56e-9,  # For new experiments: change this value as needed
                end=56e-9
                + 2e-6
                + 56e-9,  # For new experiments: change this value as needed  # noqa: E501
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,  # For new experiments: change this value as needed
                start=56e-9 + 56e-9,  # For new experiments: change this value as needed
                end=56e-9
                + 2e-6
                + 56e-9,  # For new experiments: change this value as needed  # noqa: E501
            )
        elif transition == "ef":
            # For new experiments: remove if not relevant
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,  # For new experiments: change this value as needed
                start=112e-9
                + 56e-9,  # For new experiments: change this value as needed  # noqa: E501
                end=112e-9
                + 2e-6
                + 56e-9,  # For new experiments: change this value as needed  # noqa: E501
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,  # For new experiments: change this value as needed
                start=112e-9
                + 56e-9,  # For new experiments: change this value as needed  # noqa: E501
                end=112e-9
                + 2e-6
                + 56e-9,  # For new experiments: change this value as needed  # noqa: E501
            )


# use pytest.mark.parametrize to generate test cases for
# all combinations of the parameters.
@pytest.mark.parametrize(  # For new experiments: replace with relevant name and values
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
@pytest.mark.parametrize(  # For new experiments: keep or remove as needed
    "transition",
    ["ge", "ef"],
)
@pytest.mark.parametrize(  # For new experiments: keep or remove as needed
    "use_cal_traces",
    [True, False],
)
@pytest.mark.parametrize(  # For new experiments: keep or remove as needed
    "cal_states",
    ["ge"],
)
# For new experiments: add more parameterizations for other parameters in your
# experiment over the values of which you think it makes sense to iterate in the tests.
# For new experiments: change AmplitudeRabi in the class name below to the name of
# your experiment
class TestT1TwoQubits:
    def test_pulse_count_drive(
        self,
        two_tunable_transmon_platform,
        delays,
        count,
        transition,  # For new experiments: use if relevant, or remove
        use_cal_traces,
        cal_states,  # For new experiments: use if relevant, or remove
        # For new experiments: add more arguments here if needed
    ):
        """Test the number of drive pulses.

        `two_tunable_transmon` is a pytest fixture that is automatically
        imported into the test function.

        """
        # create a verifier for the experiment
        # For new experiments: replace rabi with the name of your experiment
        verifier = create_T1_verifier(
            two_tunable_transmon_platform,
            delays,
            count,
            transition,  # For new experiments: use if relevant, or remove
            use_cal_traces,
            cal_states,  # For new experiments: use if relevant, or remove
            # For new experiments: add more arguments here if needed
        )

        # The signal names can be looked up in device_setup,
        # but typically they are in the form of
        # /logical_signal_groups/q{i}/drive(measure/acquire/drive_ef)
        # Note that with cal_state on, there is 1 additional drive pulse.
        # Check for q0
        expected_drive_count = count * (len(delays[0]) + int(use_cal_traces))
        # For new experiments: change the line above as needed
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive",
            expected_drive_count,
        )

        # Check for q1
        expected_drive_count = count * (len(delays[1]) + int(use_cal_traces))
        # For new experiments: change the line above as needed
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q1/drive",
            expected_drive_count,
        )

        if transition == "ef":  # For new experiments: remove if not relevant
            # Check for q0
            expected_drive_count = count * len(delays[0])
            # For new experiments: change the line above as needed
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive_ef",
                expected_drive_count,
            )

            # Check for q1
            expected_drive_count = count * len(delays[1])
            # For new experiments: change the line above as needed
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q1/drive_ef",
                expected_drive_count,
            )

    def test_pulse_count_measure_acquire(
        self,
        two_tunable_transmon_platform,
        delays,
        count,
        transition,  # For new experiments: use if relevant, or remove
        use_cal_traces,
        cal_states,  # For new experiments: use if relevant, or remove
        # For new experiments: add more arguments here if needed
    ):
        """Test the number of measure and acquire pulses."""
        # For new experiments: replace rabi with the name of your experiment
        verifier = create_T1_verifier(
            two_tunable_transmon_platform,
            delays,
            count,
            transition,  # For new experiments: use if relevant, or remove
            use_cal_traces,
            cal_states,  # For new experiments: use if relevant, or remove
            # For new experiments: add more arguments here if needed
        )
        # Check for q0
        # Note that with cal_state on, there are 2 additional measure pulses
        expected_measure_count = count * (len(delays[0]) + 2 * int(use_cal_traces))
        # For new experiments: change the line above as needed
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
        transition,  # For new experiments: use if relevant, or remove
        use_cal_traces,
        cal_states,  # For new experiments: use if relevant, or remove
        # For new experiments: add more arguments here if needed
    ):
        """Test the properties of drive pulses."""
        # Here, we can assert the start, end, and the parameterization of the pulses.
        # In the function `assert_pulse` below, index is the index of the pulse in the
        # pulse sequence, and `parameterized_with` is the list of SweepParameter names
        # used for that pulse. The name of the parameter should
        # match with the uid of SweepParameter in the experiment.
        # If none of the pulse parameters are swept, the list should be empty.

        # In this test, all the qubit ge drive pulses have lengths of 51ns,
        # and all the ef pulses have lengths of 52ns.

        # For new experiments: replace rabi with the name of your experiment
        verifier = create_T1_verifier(
            two_tunable_transmon_platform,
            delays,
            count,
            transition,  # For new experiments: use if relevant, or remove
            use_cal_traces,
            cal_states,  # For new experiments: use if relevant, or remove
            # For new experiments: add more arguments here if needed
        )
        if transition == "ge":
            # Here, we give an example of verifying the first drive pulse of q0
            # More pulses should be tested in a similar way
            # For new experiments: Please write tests for more than one pulse,
            # if applicable (one call to verifier.assert_pulse for every pulse)
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=0,  # For new experiments: change this value as needed
                start=0e-6,  # For new experiments: change this value as needed
                end=51e-9,  # For new experiments: change this value as needed
                parameterized_with=[],  # For new experiments: list of SweepParameter names or leave empty  # noqa: E501
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive",
                index=0,  # For new experiments: change this value as needed
                start=0e-6,  # For new experiments: change this value as needed
                end=51e-9,  # For new experiments: change this value as needed
                parameterized_with=[],  # For new experiments: list of SweepParameter names or leave empty  # noqa: E501
            )
        elif transition == "ef":  # For new experiments: remove if not relevant
            # For new experiments: Please write tests for more than one pulse,
            # if applicable (one call to verifier.assert_pulse for every pulse)
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=0,  # For new experiments: change this value as needed
                start=56e-9,  # For new experiments: change this value as needed
                end=56e-9 + 52e-9,  # For new experiments: change this value as needed
                parameterized_with=[],  # For new experiments: list of SweepParameter names or leave empty  # noqa: E501
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive_ef",
                index=0,  # For new experiments: change this value as needed
                start=56e-9,  # For new experiments: change this value as needed
                end=56e-9 + 52e-9,  # For new experiments: change this value as needed
                parameterized_with=[],  # For new experiments: list of SweepParameter names or leave empty  # noqa: E501
            )

    def test_pulse_measure(
        self,
        two_tunable_transmon_platform,
        delays,
        count,
        transition,  # For new experiments: use if relevant, or remove
        use_cal_traces,
        cal_states,  # For new experiments: use if relevant, or remove
        # For new experiments: add more arguments here if needed
    ):
        """Test the properties of measure pulses.

        Here, we can assert the start, end, and the parameterization of the pulses.

        """
        # For new experiments: replace rabi with the name of your experiment
        verifier = create_T1_verifier(
            two_tunable_transmon_platform,
            delays,
            count,
            transition,  # For new experiments: use if relevant, or remove
            use_cal_traces,
            cal_states,  # For new experiments: use if relevant, or remove
            # For new experiments: add more arguments here if needed
        )

        if transition == "ge":
            # Check for q0
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,  # For new experiments: change this value as needed
                start=56e-9 + 56e-9,  # For new experiments: change this value as needed
                end=56e-9
                + 2e-6
                + 56e-9,  # For new experiments: change this value as needed  # noqa: E501
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,  # For new experiments: change this value as needed
                start=56e-9 + 56e-9,  # For new experiments: change this value as needed
                end=56e-9
                + 2e-6
                + 56e-9,  # For new experiments: change this value as needed  # noqa: E501
            )

            # Check for q1
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/measure",
                index=0,  # For new experiments: change this value as needed
                start=56e-9 + 56e-9,  # For new experiments: change this value as needed
                end=56e-9
                + 2e-6
                + 56e-9,  # For new experiments: change this value as needed  # noqa: E501
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/acquire",
                index=0,  # For new experiments: change this value as needed
                start=56e-9 + 56e-9,  # For new experiments: change this value as needed
                end=56e-9
                + 2e-6
                + 56e-9,  # For new experiments: change this value as needed  # noqa: E501
            )
        elif transition == "ef":  # For new experiments: remove if not relevant
            # Check for q0
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,  # For new experiments: change this value as needed
                start=112e-9
                + 56e-9,  # For new experiments: change this value as needed  # noqa: E501
                end=112e-9
                + 2e-6
                + 56e-9,  # For new experiments: change this value as needed  # noqa: E501
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,  # For new experiments: change this value as needed
                start=112e-9
                + 56e-9,  # For new experiments: change this value as needed  # noqa: E501
                end=112e-9
                + 2e-6
                + 56e-9,  # For new experiments: change this value as needed  # noqa: E501
            )
            # Check for q1
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/measure",
                index=0,  # For new experiments: change this value as needed
                start=112e-9
                + 56e-9,  # For new experiments: change this value as needed  # noqa: E501
                end=112e-9
                + 2e-6
                + 56e-9,  # For new experiments: change this value as needed  # noqa: E501
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/acquire",
                index=0,  # For new experiments: change this value as needed
                start=112e-9
                + 56e-9,  # For new experiments: change this value as needed  # noqa: E501
                end=112e-9
                + 2e-6
                + 56e-9,  # For new experiments: change this value as needed  # noqa: E501
            )
