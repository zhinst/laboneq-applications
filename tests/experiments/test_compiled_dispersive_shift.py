# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

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

_LENGTH_GE = 32e-9
_LENGTH_EF = 64e-9
_LENGTH_MEASURE = 2e-6
_LENGTH_MEASURE_RESET = 2e-6 + 1e-6


def create_dispers_shift_verifier(
    tunable_transmon_platform,
    frequencies,
    count,
    states,
):
    """Create a CompiledExperimentVerifier for the amplitude rabi experiment."""
    qubits = tunable_transmon_platform.qpu.qubits
    for q in qubits:
        q.parameters.ge_drive_length = _LENGTH_GE
        q.parameters.ef_drive_length = _LENGTH_EF
    session = tunable_transmon_platform.session(do_emulation=True)
    options = dispersive_shift.experiment_workflow.options()
    options.count(count)
    # It is okay to keep it as default integration mode.
    # However, readout length will be limited in such SW modulation case.
    options.acquisition_type(AcquisitionType.SPECTROSCOPY)
    options.do_analysis(False)  # TODO: fix tests to work with do_analysis=True

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


@pytest.mark.parametrize(
    "frequencies",
    [
        np.linspace(6.8e9, 7.2e9, 1),
        np.linspace(6.5e9, 6.7e9, 3),
    ],
)
@pytest.mark.parametrize(
    "count",
    [2, 4],
)
@pytest.mark.parametrize(
    "states",
    ["ge", "ef", "gef"],
)
class TestDispersiveShiftSingleQubit:
    def test_pulse_count_drive(
        self,
        single_tunable_transmon_platform,
        frequencies,
        count,
        states,
    ):
        """Test the number of pulses."""
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
        """Test the properties of drive pulses."""

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
                pulse_timing_offset = 88e-9 + _LENGTH_MEASURE_RESET * min(
                    states.index("e"), states.index("f")
                )
            elif e_num > 0:
                pulse_timing_offset = 88e-9 + _LENGTH_MEASURE_RESET * states.index("e")
            elif f_num > 0:
                pulse_timing_offset = 88e-9 + _LENGTH_MEASURE_RESET * states.index("f")
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=0,
                start=pulse_timing_offset,
                end=pulse_timing_offset + _LENGTH_GE,
                parameterized_with=[],
            )
            if f_num > 0:
                prev_states = states[: states.index("f")]
                pulse_timing_offset = (
                    88e-9
                    + _LENGTH_MEASURE_RESET * prev_states.count("g")
                    + (_LENGTH_MEASURE_RESET + _LENGTH_GE) * (prev_states.count("e"))
                    + _LENGTH_GE
                )
                verifier.assert_pulse(
                    signal="/logical_signal_groups/q0/drive_ef",
                    index=0,
                    start=pulse_timing_offset,
                    end=pulse_timing_offset + _LENGTH_EF,
                    parameterized_with=[],
                )

    def test_pulse_measure(
        self,
        single_tunable_transmon_platform,
        frequencies,
        count,
        states,
    ):
        """Test the properties of measure pulses."""
        verifier = create_dispers_shift_verifier(
            single_tunable_transmon_platform,
            frequencies,
            count,
            states,
        )
        for qa_pair in ["measure", "acquire"]:
            start = (
                88e-9
                + _LENGTH_GE * (states[0] == "e")
                + _LENGTH_EF * (states[0] == "f")
            )
            verifier.assert_pulse(
                signal=f"/logical_signal_groups/q0/{qa_pair}",
                index=0,
                start=start,
                end=start + _LENGTH_MEASURE,
            )
