# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

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

When writing a text for a new experiment using this file as a template:
 - Search this file for "For new experiment"
 - This string indicates the places in this file where you need to make changes to
 adapt this test file for a new experiment
"""

import numpy as np
import pytest

from laboneq_applications.experiments import (
    amplitude_rabi,
)

# For new experiments: import the relevant experiment
from laboneq_applications.testing import CompiledExperimentVerifier

_LENGTH_GE = 32e-9
_LENGTH_EF = 64e-9


# For new experiments: replace rabi with the name of the new experiment
def create_rabi_verifier(
    tunable_transmon_platform,
    amplitudes,
    count,
    transition,  # For new experiments: use if relevant, or remove
    use_cal_traces,  # For new experiments: use if relevant, or remove
    readout_lengths=None,
    # For new experiments: add more arguments here if needed
):
    """Create a CompiledExperimentVerifier for the amplitude rabi experiment."""
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
    options = amplitude_rabi.experiment_workflow.options()
    options.count(count)
    # For new experiments: use the lines below or remove if not needed, and add
    # new ones for any additional input parameters you might have added
    options.transition(transition)
    options.use_cal_traces(use_cal_traces)
    # TODO: fix tests to work with do_analysis=True
    options.do_analysis(False)
    # Run the experiment workflow
    # For new experiments: use the correct experiment name
    res = amplitude_rabi.experiment_workflow(
        session=session,
        qubits=qubits,
        qpu=tunable_transmon_platform.qpu,
        amplitudes=amplitudes,
        options=options,
    ).run()
    return CompiledExperimentVerifier(res.tasks["compile_experiment"].output)


### Single-Qubit Tests ###


# use pytest.mark.parametrize to generate test cases for
# all combinations of the parameters.
@pytest.mark.parametrize(  # For new experiments: replace with relevant name and values
    "amplitudes",
    [
        np.linspace(0, 1, 11),
        np.linspace(0, 0.5, 21),
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
# For new experiments: change AmplitudeRabi in the class name below to the name of
# your experiment
class TestAmplitudeRabiSingleQubit:
    def test_pulse_count(
        self,
        single_tunable_transmon_platform,
        amplitudes,
        count,
        transition,  # For new experiments: use if relevant, or remove
        use_cal_traces,  # For new experiments: use if relevant, or remove
        # For new experiments: add more arguments here if needed
    ):
        """Test the number of drive pulses.

        `single_tunable_transmon` is a pytest fixture that is automatically
        imported into the test function.

        """
        # create a verifier for the experiment
        # For new experiments: replace rabi with the name of your experiment
        verifier = create_rabi_verifier(
            single_tunable_transmon_platform,
            amplitudes,
            count,
            transition,  # For new experiments: use if relevant, or remove
            use_cal_traces,  # For new experiments: use if relevant, or remove
            # For new experiments: add more arguments here if needed
        )

        # The signal names can be looked up in device_setup,
        # but typically they are in the form of
        # /logical_signal_groups/q{i}/drive(measure/acquire/drive_ef)
        # Note that with cal_state on, there is 1 additional drive pulse.
        expected_drive_count = count * (len(amplitudes) + int(use_cal_traces))
        # For new experiments: change the line above as needed
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive",
            expected_drive_count,
        )

        if transition == "ef":  # For new experiments: remove if not relevant
            expected_drive_count = count * len(amplitudes)
            # For new experiments: change the line above as needed
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive_ef",
                expected_drive_count,
            )

        # Note that with cal_state on, there are 2 additional measure pulses
        expected_measure_count = count * (len(amplitudes) + 2 * int(use_cal_traces))
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
        amplitudes,
        count,
        transition,  # For new experiments: use if relevant, or remove
        use_cal_traces,  # For new experiments: use if relevant, or remove
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
        verifier = create_rabi_verifier(
            single_tunable_transmon_platform,
            amplitudes,
            count,
            transition,  # For new experiments: use if relevant, or remove
            use_cal_traces,  # For new experiments: use if relevant, or remove
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
                end=_LENGTH_GE,  # For new experiments: change this value as needed
                parameterized_with=[
                    "amplitude_q0",
                ],  # For new experiments: list of SweepParameter names or leave empty
            )
        elif transition == "ef":
            # For new experiments: remove if not relevant
            # For new experiments: Please write tests for more than one pulse,
            # if applicable (one call to verifier.assert_pulse for every pulse)
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=0,  # For new experiments: change this value as needed
                start=_LENGTH_GE,  # For new experiments: change this value as needed
                end=_LENGTH_GE
                + _LENGTH_EF,  # For new experiments: change this value as needed
                parameterized_with=[
                    "amplitude_q0",
                ],  # For new experiments: list of SweepParameter names or leave empty
            )

    def test_pulse_measure(
        self,
        single_tunable_transmon_platform,
        amplitudes,
        count,
        transition,  # For new experiments: use if relevant, or remove
        use_cal_traces,  # For new experiments: use if relevant, or remove
        # For new experiments: add more arguments here if needed
    ):
        """Test the properties of measure pulses.

        Here, we assert the start, end, and the parameterization of the pulses.

        """
        # For new experiments: replace rabi with the name of your experiment
        verifier = create_rabi_verifier(
            single_tunable_transmon_platform,
            amplitudes,
            count,
            transition,  # For new experiments: use if relevant, or remove
            use_cal_traces,  # For new experiments: use if relevant, or remove
            # For new experiments: add more arguments here if needed
        )

        if transition == "ge":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,  # For new experiments: change this value as needed
                start=_LENGTH_GE,  # For new experiments: change this value as needed
                end=_LENGTH_GE
                + 2e-6,  # For new experiments: change this value as needed
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,  # For new experiments: change this value as needed
                start=_LENGTH_GE,  # For new experiments: change this value as needed
                end=_LENGTH_GE
                + 2e-6,  # For new experiments: change this value as needed
            )
        elif transition == "ef":
            # For new experiments: remove if not relevant
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,  # For new experiments: change this value as needed
                start=_LENGTH_GE
                + _LENGTH_EF,  # For new experiments: change this value as needed
                end=_LENGTH_GE
                + _LENGTH_EF
                + 2e-6,  # For new experiments: change this value as needed
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,  # For new experiments: change this value as needed
                start=_LENGTH_GE
                + _LENGTH_EF,  # For new experiments: change this value as needed
                end=_LENGTH_GE
                + _LENGTH_EF
                + 2e-6,  # For new experiments: change this value as needed
            )


# use pytest.mark.parametrize to generate test cases for
# all combinations of the parameters.
@pytest.mark.parametrize(  # For new experiments: replace with relevant name and values
    ("amplitudes", "readout_lengths"),
    [
        ([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], [1e-6, 1e-6]),
        ([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]], [100e-9, 200e-9]),
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
# For new experiments: change AmplitudeRabi in the class name below to the name of
# your experiment
class TestAmplitudeRabiTwoQubits:
    def test_pulse_count(
        self,
        two_tunable_transmon_platform,
        amplitudes,
        count,
        transition,  # For new experiments: use if relevant, or remove
        use_cal_traces,  # For new experiments: use if relevant, or remove
        readout_lengths,
        # For new experiments: add more arguments here if needed
    ):
        """Test the number of pulses.

        `two_tunable_transmon` is a pytest fixture that is automatically
        imported into the test function.

        """
        # create a verifier for the experiment
        # For new experiments: replace rabi with the name of your experiment
        verifier = create_rabi_verifier(
            two_tunable_transmon_platform,
            amplitudes,
            count,
            transition,  # For new experiments: use if relevant, or remove
            use_cal_traces,  # For new experiments: use if relevant, or remove
            readout_lengths,
            # For new experiments: add more arguments here if needed
        )

        # The signal names can be looked up in device_setup,
        # but typically they are in the form of
        # /logical_signal_groups/q{i}/drive(measure/acquire/drive_ef)
        # Note that with cal_state on, there is 1 additional drive pulse.
        # Check for q0
        expected_drive_count = count * (len(amplitudes[0]) + int(use_cal_traces))
        # For new experiments: change the line above as needed
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive",
            expected_drive_count,
        )

        # Check for q1
        expected_drive_count = count * (len(amplitudes[1]) + int(use_cal_traces))
        # For new experiments: change the line above as needed
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q1/drive",
            expected_drive_count,
        )

        if transition == "ef":  # For new experiments: remove if not relevant
            # Check for q0
            expected_drive_count = count * len(amplitudes[0])
            # For new experiments: change the line above as needed
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive_ef",
                expected_drive_count,
            )

            # Check for q1
            expected_drive_count = count * len(amplitudes[1])
            # For new experiments: change the line above as needed
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q1/drive_ef",
                expected_drive_count,
            )

        # Check for q0
        # Note that with cal_state on, there are 2 additional measure pulses
        expected_measure_count = count * (len(amplitudes[0]) + 2 * int(use_cal_traces))
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
        amplitudes,
        count,
        transition,  # For new experiments: use if relevant, or remove
        use_cal_traces,  # For new experiments: use if relevant, or remove
        readout_lengths,
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
        verifier = create_rabi_verifier(
            two_tunable_transmon_platform,
            amplitudes,
            count,
            transition,  # For new experiments: use if relevant, or remove
            use_cal_traces,  # For new experiments: use if relevant, or remove
            readout_lengths,
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
                end=_LENGTH_GE,  # For new experiments: change this value as needed
                parameterized_with=[
                    "amplitude_q0",
                ],  # For new experiments: list of SweepParameter names or leave empty
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive",
                index=0,  # For new experiments: change this value as needed
                start=0e-6,  # For new experiments: change this value as needed
                end=_LENGTH_GE,  # For new experiments: change this value as needed
                parameterized_with=[
                    "amplitude_q1",
                ],  # For new experiments: list of SweepParameter names or leave empty
            )
        elif transition == "ef":  # For new experiments: remove if not relevant
            # For new experiments: Please write tests for more than one pulse,
            # if applicable (one call to verifier.assert_pulse for every pulse)
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=0,  # For new experiments: change this value as needed
                start=_LENGTH_GE,  # For new experiments: change this value as needed
                end=_LENGTH_GE
                + _LENGTH_EF,  # For new experiments: change this value as needed
                parameterized_with=[
                    "amplitude_q0",
                ],  # For new experiments: list of SweepParameter names or leave empty
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive_ef",
                index=0,  # For new experiments: change this value as needed
                start=_LENGTH_GE,  # For new experiments: change this value as needed
                end=_LENGTH_GE
                + _LENGTH_EF,  # For new experiments: change this value as needed
                parameterized_with=[
                    "amplitude_q1",
                ],  # For new experiments: list of SweepParameter names or leave empty
            )

    def test_pulse_measure(
        self,
        two_tunable_transmon_platform,
        amplitudes,
        count,
        transition,  # For new experiments: use if relevant, or remove
        use_cal_traces,  # For new experiments: use if relevant, or remove
        readout_lengths,
        # For new experiments: add more arguments here if needed
    ):
        """Test the properties of measure pulses.

        Here, we can assert the start, end, and the parameterization of the pulses.

        """
        # For new experiments: replace rabi with the name of your experiment
        verifier = create_rabi_verifier(
            two_tunable_transmon_platform,
            amplitudes,
            count,
            transition,  # For new experiments: use if relevant, or remove
            use_cal_traces,  # For new experiments: use if relevant, or remove
            readout_lengths,
            # For new experiments: add more arguments here if needed
        )

        if transition == "ge":
            # Check for q0
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,  # For new experiments: change this value as needed
                start=_LENGTH_GE,  # For new experiments: change this value as needed
                end=_LENGTH_GE
                + readout_lengths[
                    0
                ],  # For new experiments: change this value as needed
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,  # For new experiments: change this value as needed
                start=_LENGTH_GE,  # For new experiments: change this value as needed
                end=_LENGTH_GE
                + 2e-6,  # For new experiments: change this value as needed
            )

            # Check for q1
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/measure",
                index=0,  # For new experiments: change this value as needed
                start=_LENGTH_GE,  # For new experiments: change this value as needed
                end=_LENGTH_GE
                + readout_lengths[
                    1
                ],  # For new experiments: change this value as needed
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/acquire",
                index=0,  # For new experiments: change this value as needed
                start=_LENGTH_GE,  # For new experiments: change this value as needed
                end=_LENGTH_GE
                + 2e-6,  # For new experiments: change this value as needed
            )
        elif transition == "ef":  # For new experiments: remove if not relevant
            # Check for q0
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,  # For new experiments: change this value as needed
                start=_LENGTH_GE
                + _LENGTH_EF,  # For new experiments: change this value as needed
                end=_LENGTH_GE
                + _LENGTH_EF
                + readout_lengths[
                    0
                ],  # For new experiments: change this value as needed
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,  # For new experiments: change this value as needed
                start=_LENGTH_GE
                + _LENGTH_EF,  # For new experiments: change this value as needed
                end=_LENGTH_GE
                + _LENGTH_EF
                + 2e-6,  # For new experiments: change this value as needed
            )
            # Check for q1
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/measure",
                index=0,  # For new experiments: change this value as needed
                start=_LENGTH_GE
                + _LENGTH_EF,  # For new experiments: change this value as needed
                end=_LENGTH_GE
                + _LENGTH_EF
                + readout_lengths[
                    1
                ],  # For new experiments: change this value as needed
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/acquire",
                index=0,  # For new experiments: change this value as needed
                start=_LENGTH_GE
                + _LENGTH_EF,  # For new experiments: change this value as needed
                end=_LENGTH_GE
                + _LENGTH_EF
                + 2e-6,  # For new experiments: change this value as needed
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
    options = amplitude_rabi.experiment_workflow.options()
    options.transition(transition)
    options.cal_states(cal_states)
    options.active_reset(True)
    options.active_reset_states(active_reset_states)
    options.active_reset_repetitions(active_reset_repetitions)
    [q0] = single_tunable_transmon_platform.qpu.qubits
    amplitudes = np.linspace(0, 1, 11)
    workflow_result = amplitude_rabi.experiment_workflow(
        session=single_tunable_transmon_platform.session(do_emulation=True),
        qubits=q0,
        qpu=single_tunable_transmon_platform.qpu,
        amplitudes=amplitudes,
        options=options,
    ).run()

    exp = workflow_result.tasks["create_experiment"].output
    active_reset_section = exp.sections[0].children[0].children[0]
    assert active_reset_section.uid == "active_reset_q0_0"
    truth_len = len(q0.signals) + active_reset_repetitions * 3
    assert len(active_reset_section.children) == truth_len

    data = workflow_result.output
    assert "active_reset" in data.q0
    shape_truth = (
        (len(amplitudes), active_reset_repetitions)
        if active_reset_repetitions > 1
        else (len(amplitudes),)
    )
    assert np.shape(data.q0.active_reset.result.data) == shape_truth
    for s in cal_states:
        cal_trace_data = data.q0.active_reset.cal_trace[s].data
        if active_reset_repetitions == 1:
            assert isinstance(cal_trace_data, np.complex128)
        else:
            assert len(cal_trace_data) == active_reset_repetitions


def test_invalid_averaging_mode(single_tunable_transmon_platform):
    [q0] = single_tunable_transmon_platform.qpu.qubits
    session = single_tunable_transmon_platform.session(do_emulation=True)
    options = amplitude_rabi.experiment_workflow.options()
    options.averaging_mode("sequential")
    options.use_cal_traces(True)
    options.do_analysis(False)

    with pytest.raises(ValueError) as err:
        amplitude_rabi.experiment_workflow(
            session=session,
            qubits=q0,
            qpu=single_tunable_transmon_platform.qpu,
            amplitudes=np.linspace(0, 1, 10),
            options=options,
        ).run()

    assert str(err.value) == (
        "'AveragingMode.SEQUENTIAL' (or {AveragingMode.SEQUENTIAL}) cannot be used "
        "with calibration traces because the calibration traces are added "
        "outside the sweep."
    )
