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
):
    """Create a CompiledExperimentVerifier for the iq_blobs experiment."""
    qubits = tunable_transmon_platform.qpu.qubits
    if len(qubits) == 1:
        qubits = qubits[0]
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


### Single-Qubit Tests ###


# use pytest.mark.parametrize to generate test cases for
# all combinations of the parameters.
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
    def test_pulse_count_drive(
        self,
        single_tunable_transmon_platform,
        count,
        states,
    ):
        """Test the number of drive pulses.

        `single_tunable_transmon` is a pytest fixture that is automatically
        imported into the test function.

        """
        # create a verifier for the experiment

        verifier = create_iq_blobs_verifier(
            single_tunable_transmon_platform,
            count,
            states,
        )

        # The signal names can be looked up in device_setup,
        # but typically they are in the form of
        # /logical_signal_groups/q{i}/drive(measure/acquire/drive_ef)
        # Note that with cal_state on, there is 1 additional drive pulse.
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

    def test_pulse_count_measure_acquire(
        self,
        single_tunable_transmon_platform,
        count,
        states,
    ):
        """Test the number of measure and acquire pulses."""

        verifier = create_iq_blobs_verifier(
            single_tunable_transmon_platform,
            count,
            states,
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
        # Here, we can assert the start, end, and the parameterization of the pulses.
        # In the function `assert_pulse` below, index is the index of the pulse in the
        # pulse sequence, and `parameterized_with` is the list of SweepParameter names
        # used for that pulse. The name of the parameter should
        # match with the uid of SweepParameter in the experiment.
        # If none of the pulse parameters are swept, the list should be empty.

        # In this test, all the qubit ge drive pulses have lengths of 51ns,
        # and all the ef pulses have lengths of 52ns.

        verifier = create_iq_blobs_verifier(
            single_tunable_transmon_platform,
            count,
            states,
        )
        if states == "ge":
            # ge pulses
            # Here, we give an example of verifying the first drive pulse of q0
            # More pulses should be tested in a similar way

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


# use pytest.mark.parametrize to generate test cases for
# all combinations of the parameters.
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
class TestIQBlobsTwoQubit:
    def test_pulse_count_drive(
        self,
        two_tunable_transmon_platform,
        count,
        states,
    ):
        """Test the number of drive pulses.

        `single_tunable_transmon` is a pytest fixture that is automatically
        imported into the test function.

        """
        # create a verifier for the experiment

        verifier = create_iq_blobs_verifier(
            two_tunable_transmon_platform,
            count,
            states,
        )

        # The signal names can be looked up in device_setup,
        # but typically they are in the form of
        # /logical_signal_groups/q{i}/drive(measure/acquire/drive_ef)
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

    def test_pulse_count_measure_acquire(
        self,
        two_tunable_transmon_platform,
        count,
        states,
    ):
        """Test the number of measure and acquire pulses."""

        verifier = create_iq_blobs_verifier(
            two_tunable_transmon_platform,
            count,
            states,
        )
        # Note that with cal_state on, there are 2 additional measure pulses
        expected_measure_count = count * (
            states.count("g") + states.count("e") + states.count("f")
        )
        # q0
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/measure",
            expected_measure_count,
        )

        # acquire and measure pulses have the same count
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/acquire",
            expected_measure_count,
        )
        # q1
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

        verifier = create_iq_blobs_verifier(
            two_tunable_transmon_platform,
            count,
            states,
        )
        if states == "ge":
            # ge pulses
            # Here, we give an example of verifying the first drive pulse of q0
            # More pulses should be tested in a similar way

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
    ):
        """Test the properties of measure pulses.

        Here, we assert the start, end, and the parameterization of the pulses.

        """

        verifier = create_iq_blobs_verifier(
            two_tunable_transmon_platform,
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
        verifier.assert_pulse(
            signal="/logical_signal_groups/q1/measure",
            index=0,
            start=0e-9,
            end=2e-6,
        )
        verifier.assert_pulse(
            signal="/logical_signal_groups/q1/acquire",
            index=0,
            start=0e-9,
            end=2e-6,
        )
