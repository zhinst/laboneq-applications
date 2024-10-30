"""Tests for the compiled time_traces experiment using the testing utilities
provided by the LabOne Q Applications Library.

"""

import pytest

from laboneq_applications.experiments import (
    time_traces,
)
from laboneq_applications.testing import CompiledExperimentVerifier


def create_time_traces_verifier(
    tunable_transmon_platform,
    states,
    count,
):
    """Create a CompiledExperimentVerifier for the time trace measurement."""
    qubits = tunable_transmon_platform.qpu.qubits
    if len(qubits) == 1:
        qubits = [qubits[0]]
    session = tunable_transmon_platform.session(do_emulation=True)
    options = time_traces.experiment_workflow.options()
    options.count(count)
    # TODO: fix tests to work with do_analysis=True when the new options feature is in
    options.do_analysis(False)

    # Run the experiment workflow
    res = time_traces.experiment_workflow(
        session=session,
        qubits=qubits,
        qpu=tunable_transmon_platform.qpu,
        states=states,
        options=options,
    ).run()
    return CompiledExperimentVerifier(res.tasks["compile_experiment"].output)


### Single-Qubit Tests ###


@pytest.mark.parametrize(
    "states",
    [
        ("g"),
        ("e"),
        ("f"),
    ],
)
@pytest.mark.parametrize(
    "count",
    [2, 4],
)
class TestTimeTracesSingleQubit:
    def test_pulse_count(
        self,
        single_tunable_transmon_platform,
        states,
        count,
    ):
        """Test the number of pulses."""
        # create a verifier for the experiment
        verifier = create_time_traces_verifier(
            single_tunable_transmon_platform,
            states,
            count,
        )

        expected_ge_drive_count = count * (
            0 * states.count("g") + 1 * states.count("e") + 1 * states.count("f")
        )
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive",
            expected_ge_drive_count,
        )
        expected_ef_drive_count = count * (
            0 * states.count("g") + 0 * states.count("e") + 1 * states.count("f")
        )
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive_ef",
            expected_ef_drive_count,
        )

        expected_measure_count = count * len(states)
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
        states,
        count,
    ):
        """Test the properties of drive pulses."""

        verifier = create_time_traces_verifier(
            single_tunable_transmon_platform,
            states,
            count,
        )
        if states == ("e"):
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=0,
                start=0e-6,
                end=51e-9,
                parameterized_with=[],
            )
        elif states == ("f"):
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=0,
                start=0e-6,
                end=51e-9,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=0,
                start=56e-9,
                end=56e-9 + 52e-9,
                parameterized_with=[],
            )

    def test_pulse_measure(
        self,
        single_tunable_transmon_platform,
        states,
        count,
    ):
        """Test the properties of measure pulses."""
        verifier = create_time_traces_verifier(
            single_tunable_transmon_platform,
            states,
            count,
        )
        if states == ("g"):
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
        if states == ("e"):
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
        if states == ("f"):
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
