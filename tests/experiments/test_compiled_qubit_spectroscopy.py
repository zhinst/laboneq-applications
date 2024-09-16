"""Tests for the compiled qubit-spectroscopy experiment using the testing utilities
provided by the LabOne Q Applications Library.
"""

import numpy as np
import pytest

from laboneq_applications.experiments import (
    qubit_spectroscopy,
)
from laboneq_applications.testing import CompiledExperimentVerifier


def create_qubitspec_verifier(
    tunable_transmon_platform,
    frequencies,
    count,
):
    """Create a CompiledExperimentVerifier for the amplitude rabi experiment."""
    qubits = tunable_transmon_platform.qpu.qubits
    if len(qubits) == 1:
        qubits = qubits[0]
    session = tunable_transmon_platform.session(do_emulation=True)
    options = qubit_spectroscopy.options()
    options.create_experiment.count = count

    # Run the experiment workflow
    res = qubit_spectroscopy.experiment_workflow(
        session=session,
        qubits=qubits,
        qpu=tunable_transmon_platform.qpu,
        frequencies=frequencies,
        options=options,
    ).run()
    return CompiledExperimentVerifier(res.tasks["compile_experiment"].output)


### Single-Qubit Tests ###


# use pytest.mark.parametrize to generate test cases for
# all combinations of the parameters.
@pytest.mark.parametrize(
    "frequencies",
    [
        np.linspace(6.5e9, 6.7e9, 1),
        np.linspace(6.5e9, 6.7e9, 6),
        np.linspace(6.5e9, 6.7e9, 9),
    ],
)
@pytest.mark.parametrize(
    "count",
    [2, 4],
)
class TestQubitSpectroscopySingleQubit:
    def test_pulse_count_drive(
        self,
        single_tunable_transmon_platform,
        frequencies,
        count,
    ):
        """Test the number of drive pulses.

        `single_tunable_transmon` is a pytest fixture that is automatically
        imported into the test function.

        """
        # create a verifier for the experiment
        verifier = create_qubitspec_verifier(
            single_tunable_transmon_platform,
            frequencies,
            count,
        )

        expected_drive_count = count * len(frequencies)
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive",
            expected_drive_count,
        )

    def test_pulse_count_measure_acquire(
        self,
        single_tunable_transmon_platform,
        frequencies,
        count,
    ):
        """Test the number of measure and acquire pulses."""

        verifier = create_qubitspec_verifier(
            single_tunable_transmon_platform,
            frequencies,
            count,
        )

        expected_measure_count = count * len(frequencies)
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
        frequencies,
        count,
    ):
        """Test the properties of drive pulses.

        In this test, all the qubit ge drive pulses have lengths of 51ns,
        and all the ef pulses have lengths of 52ns.

        """

        verifier = create_qubitspec_verifier(
            single_tunable_transmon_platform,
            frequencies,
            count,
        )
        verifier.assert_pulse(
            signal="/logical_signal_groups/q0/drive",
            index=0,
            start=88e-9,
            end=88e-9 + 5e-6,
            parameterized_with=[],
        )

    def test_pulse_measure(
        self,
        single_tunable_transmon_platform,
        frequencies,
        count,
    ):
        """Test the properties of measure pulses.

        Here, we assert the start, end, and the parameterization of the pulses.

        """
        verifier = create_qubitspec_verifier(
            single_tunable_transmon_platform,
            frequencies,
            count,
        )

        verifier.assert_pulse(
            signal="/logical_signal_groups/q0/measure",
            index=0,
            start=88e-9 + 5e-6,
            end=88e-9 + 5e-6 + 2e-6,
        )
        verifier.assert_pulse(
            signal="/logical_signal_groups/q0/acquire",
            index=0,
            start=88e-9 + 5e-6,
            end=88e-9 + 5e-6 + 2e-6,
        )


# use pytest.mark.parametrize to generate test cases for
# all combinations of the parameters.
@pytest.mark.parametrize(
    "frequencies",
    [
        [np.linspace(6.5e9, 6.7e9, 1), np.linspace(6.5e9, 6.7e9, 1)],
        [np.linspace(6.5e9, 6.7e9, 4), np.linspace(6.5e9, 6.7e9, 4)],
    ],
)
@pytest.mark.parametrize(
    "count",
    [2, 4],
)
class TestQubitSpectroscopyTwoQubits:
    def test_pulse_count_drive(
        self,
        two_tunable_transmon_platform,
        frequencies,
        count,
    ):
        """Test the number of drive pulses.

        `two_tunable_transmon` is a pytest fixture that is automatically
        imported into the test function.

        """
        # create a verifier for the experiment

        verifier = create_qubitspec_verifier(
            two_tunable_transmon_platform,
            frequencies,
            count,
        )

        # Check for q0
        expected_drive_count = count * len(frequencies[0])
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive",
            expected_drive_count,
        )

        # Check for q1
        expected_drive_count = count * len(frequencies[1])
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q1/drive",
            expected_drive_count,
        )

    def test_pulse_count_measure_acquire(
        self,
        two_tunable_transmon_platform,
        frequencies,
        count,
    ):
        """Test the number of measure and acquire pulses."""
        verifier = create_qubitspec_verifier(
            two_tunable_transmon_platform,
            frequencies,
            count,
        )
        # Check for q0
        expected_measure_count = count * len(frequencies[0])
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
        frequencies,
        count,
    ):
        """Test the properties of drive pulses.

        In this test, all the qubit ge drive pulses have lengths of 51ns,
        and all the ef pulses have lengths of 52ns.
        """

        verifier = create_qubitspec_verifier(
            two_tunable_transmon_platform,
            frequencies,
            count,
        )

        verifier.assert_pulse(
            signal="/logical_signal_groups/q0/drive",
            index=0,
            start=88e-9,
            end=88e-9 + 5e-6,
            parameterized_with=[],
        )

        verifier.assert_pulse(
            signal="/logical_signal_groups/q1/drive",
            index=0,
            start=88e-9,
            end=88e-9 + 5e-6,
            parameterized_with=[],
        )

    def test_pulse_measure(
        self,
        two_tunable_transmon_platform,
        frequencies,
        count,
    ):
        """Test the properties of measure pulses.

        Here, we can assert the start, end, and the parameterization of the pulses.

        """
        verifier = create_qubitspec_verifier(
            two_tunable_transmon_platform,
            frequencies,
            count,
        )
        # Check for q0
        verifier.assert_pulse(
            signal="/logical_signal_groups/q0/measure",
            index=0,
            start=88e-9 + 5e-6,
            end=88e-9 + 5e-6 + 2e-6,
        )
        verifier.assert_pulse(
            signal="/logical_signal_groups/q0/acquire",
            index=0,
            start=88e-9 + 5e-6,
            end=88e-9 + 5e-6 + 2e-6,
        )

        # Check for q1
        verifier.assert_pulse(
            signal="/logical_signal_groups/q1/measure",
            index=0,
            start=88e-9 + 5e-6,
            end=88e-9 + 5e-6 + 2e-6,
        )
        verifier.assert_pulse(
            signal="/logical_signal_groups/q1/acquire",
            index=0,
            start=88e-9 + 5e-6,
            end=88e-9 + 5e-6 + 2e-6,
        )
