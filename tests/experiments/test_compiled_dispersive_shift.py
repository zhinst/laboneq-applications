"""Tests for the compiled dispersive-shift experiment using the testing utilities
provided by the LabOne Q Applications Library.
"""

import numpy as np
import pytest
from laboneq.dsl.enums import AcquisitionType

from laboneq_applications.experiments import (
    dispersive_shift,
)
from laboneq_applications.testing import CompiledExperimentVerifier


def create_dispers_shift_verifier(
    tunable_transmon_platform,
    frequencies,
    count,
    states,
):
    """Create a CompiledExperimentVerifier for the amplitude rabi experiment."""
    qubits = tunable_transmon_platform.qpu.qubits
    session = tunable_transmon_platform.session(do_emulation=True)
    options = dispersive_shift.options()
    options.create_experiment.count = count
    # It is okay to keep it as default integration mode.
    # However, readout length will be limited in such SW modulation case.
    options.create_experiment.acquisition_type = AcquisitionType.SPECTROSCOPY

    # Run the experiment workflow
    res = dispersive_shift.experiment_workflow(
        session=session,
        qubit=qubits[0],
        qpu=tunable_transmon_platform.qpu,
        frequencies=frequencies,
        states=states,
        options=options,
    ).run()
    return CompiledExperimentVerifier(res.tasks["compile_experiment"].output)


### Single-Qubit Tests ###


# use pytest.mark.parametrize to generate test cases for
# all combinations of the parameters.
@pytest.mark.parametrize(
    "frequencies",
    [
        np.linspace(1.8e9, 2.2e9, 1),
        np.linspace(1.5e9, 1.7e9, 3),
    ],
)
@pytest.mark.parametrize(
    "count",
    [2, 4],
)
@pytest.mark.parametrize(
    "states",
    ["g", "e", "f", "ge", "ef"],
)
class TestDispersiveShiftSingleQubit:
    def test_pulse_count_drive(
        self,
        single_tunable_transmon_platform,
        frequencies,
        count,
        states,
    ):
        """Test the number of drive pulses.

        `single_tunable_transmon` is a pytest fixture that is automatically
        imported into the test function.

        """
        # create a verifier for the experiment
        verifier = create_dispers_shift_verifier(
            single_tunable_transmon_platform,
            frequencies,
            count,
            states,
        )

        expected_drive_count = (
            count * len(frequencies) * (states.count("e") + states.count("f"))
        )
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive",
            expected_drive_count,
        )

        if "f" in states:
            expected_drive_count = count * len(frequencies) * states.count("f")
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive_ef",
                expected_drive_count,
            )

    def test_pulse_count_measure_acquire(
        self,
        single_tunable_transmon_platform,
        frequencies,
        count,
        states,
    ):
        """Test the number of measure and acquire pulses."""
        verifier = create_dispers_shift_verifier(
            single_tunable_transmon_platform,
            frequencies,
            count,
            states,
        )

        # Note that with cal_state on, there are 2 additional measure pulses
        expected_measure_count = count * len(frequencies) * len(states)
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
        states,
    ):
        """Test the properties of drive pulses.

        In this test, all the qubit ge drive pulses have lengths of 51ns,
        and all the ef pulses have lengths of 52ns.

        """

        verifier = create_dispers_shift_verifier(
            single_tunable_transmon_platform,
            frequencies,
            count,
            states,
        )
        g_num = states.count("g")
        e_num = states.count("e")
        f_num = states.count("f")
        if g_num == len(states):
            pass
        else:
            if e_num > 0 and f_num > 0:
                pulse_timing_offset = 88e-9 + 3e-6 * min(
                    states.index("e"), states.index("f")
                )
            elif e_num > 0:
                pulse_timing_offset = 88e-9 + 3e-6 * states.index("e")
            elif f_num > 0:
                pulse_timing_offset = 88e-9 + 3e-6 * states.index("f")
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=0,
                start=pulse_timing_offset,
                end=pulse_timing_offset + 51e-9,
                parameterized_with=[],
            )
            if f_num > 0:
                prev_states = states[: states.index("f")]
                pulse_timing_offset = (
                    88e-9
                    + 3e-6 * prev_states.count("g")
                    + (3e-6 + 56e-9) * (prev_states.count("e"))
                    + 56e-9
                )
                verifier.assert_pulse(
                    signal="/logical_signal_groups/q0/drive_ef",
                    index=0,
                    start=pulse_timing_offset,
                    end=pulse_timing_offset + 52e-9,
                    parameterized_with=[],
                )

    def test_pulse_measure(
        self,
        single_tunable_transmon_platform,
        frequencies,
        count,
        states,
    ):
        """Test the properties of measure pulses.

        Here, we assert the start, end, and the parameterization of the pulses.

        """
        verifier = create_dispers_shift_verifier(
            single_tunable_transmon_platform,
            frequencies,
            count,
            states,
        )
        for qa_pair in ["measure", "acquire"]:
            verifier.assert_pulse(
                signal=f"/logical_signal_groups/q0/{qa_pair}",
                index=0,
                start=88e-9 + 56e-9 * (states[0] == "e") + 112e-9 * (states[0] == "f"),
                end=88e-9
                + 56e-9 * (states[0] == "e")
                + 112e-9 * (states[0] == "f")
                + 2e-6,
            )
