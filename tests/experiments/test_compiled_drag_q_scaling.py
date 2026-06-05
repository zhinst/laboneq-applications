# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for the compiled qubit-spectroscopy experiment using the testing utilities
provided by the LabOne Q Applications Library.
"""

import numpy as np
import pytest

from laboneq_applications.experiments import (
    drag_q_scaling,
)
from laboneq_applications.testing import CompiledExperimentVerifier

_START_TIME = 2e-9
_LENGTH_GE_DRIVE = 51e-9
_LENGTH_ACQUIRE = 2e-6


def create_qubitspec_verifier(
    tunable_transmon_platform,
    q_scalings,
    count,
    use_cal_traces,
    readout_lengths=None,
):
    """Create a CompiledExperimentVerifier."""
    qpu = tunable_transmon_platform.qpu
    qubits = qpu.quantum_elements
    qubit_uids = qpu.quantum_element_uids
    if len(qubit_uids) == 1:
        qubit_uids = qubit_uids[0]
    if readout_lengths is not None:
        assert len(readout_lengths) == len(qubits)
        for i, rl in enumerate(readout_lengths):
            qubits[i].parameters.readout_length = rl
    session = tunable_transmon_platform.session(do_emulation=True)
    options = drag_q_scaling.experiment_workflow.options()
    options.count(count)
    options.use_cal_traces(use_cal_traces)
    options.do_analysis(False)

    res = drag_q_scaling.experiment_workflow(
        session=session,
        qubits=qubit_uids,
        qpu=qpu,
        q_scalings=q_scalings,
        options=options,
    ).run()
    return CompiledExperimentVerifier(res.tasks["compile_experiment"].output)


@pytest.mark.parametrize(
    ("q_scalings", "readout_lengths"),
    [
        (np.linspace(-0.02, 0.03, 5), [1e-6]),
        (
            np.linspace(-0.02, 0.03, 4),
            [100e-9],
        ),
    ],
)
@pytest.mark.parametrize(
    "use_cal_traces",
    [True, False],
)
@pytest.mark.parametrize(
    "count",
    [2, 4],
)
class TestQubitSpectroscopySingleQubit:
    def test_pulse_count_drive(
        self,
        single_tunable_transmon_platform,
        count,
        q_scalings,
        use_cal_traces,
        readout_lengths,
    ):
        """Test the number of drive pulses."""
        verifier = create_qubitspec_verifier(
            single_tunable_transmon_platform,
            q_scalings,
            count,
            use_cal_traces,
            readout_lengths,
        )

        # Loop over 3 qops per q_scaling, each iteration having x90 followed by qop.
        expected_drive_count = count * (
            3 * 2 * len(q_scalings) + int(use_cal_traces)
        )
        verifier.assert_number_of_pulses(
            "q0/drive",
            expected_drive_count,
        )

    def test_pulse_count_measure_acquire(
        self,
        single_tunable_transmon_platform,
        q_scalings,
        count,
        use_cal_traces,
        readout_lengths,
    ):
        """Test the number of measure and acquire pulses."""

        verifier = create_qubitspec_verifier(
            single_tunable_transmon_platform,
            q_scalings,
            count,
            use_cal_traces,
            readout_lengths,
        )

        expected_measure_count = count * (
            3 * len(q_scalings) + int(use_cal_traces) * 2
        ) # 3 quantum operations per q_scaling
        verifier.assert_number_of_pulses(
            "q0/measure",
            expected_measure_count,
        )

        # acquire and measure pulses have the same count
        verifier.assert_number_of_pulses(
            "q0/acquire",
            expected_measure_count,
        )

    def test_pulse_drive(
        self,
        single_tunable_transmon_platform,
        q_scalings,
        count,
        use_cal_traces,
        readout_lengths,
    ):
        """Test the properties of drive pulses."""

        verifier = create_qubitspec_verifier(
            single_tunable_transmon_platform,
            q_scalings,
            count,
            use_cal_traces,
            readout_lengths,
        )
        verifier.assert_pulse(
            signal="q0/drive",
            index=0,
            start=_START_TIME,
            end=_START_TIME + _LENGTH_GE_DRIVE,
            parameterized_with=[],
        )

    def test_pulse_measure(
        self,
        single_tunable_transmon_platform,
        q_scalings,
        count,
        use_cal_traces,
        readout_lengths,
    ):
        """Test the properties of measure pulses."""
        verifier = create_qubitspec_verifier(
            single_tunable_transmon_platform,
            q_scalings,
            count,
            use_cal_traces,
            readout_lengths,
        )

        start_measure = _START_TIME + 2 * _LENGTH_GE_DRIVE
        verifier.assert_pulse(
            signal="q0/measure",
            index=0,
            start=start_measure,
            end=start_measure + readout_lengths[0],
        )
        verifier.assert_pulse(
            signal="q0/acquire",
            index=0,
            start=start_measure,
            end=_LENGTH_ACQUIRE + start_measure,
        )


@pytest.mark.parametrize(
    ("q_scalings", "readout_lengths"),
    [
        ([np.linspace(-0.02, 0.03, 5), np.linspace(-0.1, 0.1, 5)], [1e-6, 1e-6]),
        (
            [np.linspace(-0.02, 0.03, 4), np.linspace(0.01, 0.05, 4)],
            [100e-9, 200e-9],
        ),
    ],
)
@pytest.mark.parametrize(
    "use_cal_traces",
    [True, False],
)
@pytest.mark.parametrize(
    "count",
    [2, 4],
)
class TestQubitSpectroscopyTwoQubits:
    def test_pulse_count_drive(
        self,
        two_tunable_transmon_platform,
        q_scalings,
        count,
        use_cal_traces,
        readout_lengths,
    ):
        """Test the number of drive pulses."""

        verifier = create_qubitspec_verifier(
            two_tunable_transmon_platform,
            q_scalings,
            count,
            use_cal_traces,
            readout_lengths,
        )

        # Check for q0
        # Loop over 3 qops per q_scaling, each iteration having x90 followed by qop.
        expected_drive_count = count * (
            3 * 2 * len(q_scalings[0]) + int(use_cal_traces)
        )
        verifier.assert_number_of_pulses(
            "q0/drive",
            expected_drive_count,
        )

        # Check for q1
        # Loop over 3 qops per q_scaling, each iteration having x90 followed by qop.
        expected_drive_count = count * (
            3 * 2 * len(q_scalings[1]) + int(use_cal_traces)
        )
        verifier.assert_number_of_pulses(
            "q1/drive",
            expected_drive_count,
        )

    def test_pulse_count_measure_acquire(
        self,
        two_tunable_transmon_platform,
        q_scalings,
        count,
        use_cal_traces,
        readout_lengths,
    ):
        """Test the number of measure and acquire pulses."""
        verifier = create_qubitspec_verifier(
            two_tunable_transmon_platform,
            q_scalings,
            count,
            use_cal_traces,
            readout_lengths,
        )
        # Check for q0
        expected_measure_count = count * (
            3 * len(q_scalings[0]) + 2 * int(use_cal_traces)
        )  # 3 quantum operations per q_scaling
        verifier.assert_number_of_pulses(
            "q0/measure",
            expected_measure_count,
        )

        # acquire and measure pulses have the same count
        verifier.assert_number_of_pulses(
            "q0/acquire",
            expected_measure_count,
        )

        # Check for q1
        verifier.assert_number_of_pulses(
            "q1/measure",
            expected_measure_count,
        )

        # acquire and measure pulses have the same count
        verifier.assert_number_of_pulses(
            "q1/acquire",
            expected_measure_count,
        )

    def test_pulse_drive(
        self,
        two_tunable_transmon_platform,
        q_scalings,
        count,
        use_cal_traces,
        readout_lengths,
    ):
        """Test the properties of drive pulses."""

        verifier = create_qubitspec_verifier(
            two_tunable_transmon_platform,
            q_scalings,
            count,
            use_cal_traces,
            readout_lengths,
        )

        verifier.assert_pulse(
            signal="q0/drive",
            index=0,
            start=_START_TIME,
            end=_START_TIME + _LENGTH_GE_DRIVE,
            parameterized_with=[],
        )

        verifier.assert_pulse(
            signal="q1/drive",
            index=0,
            start=_START_TIME,
            end=_START_TIME + _LENGTH_GE_DRIVE,
            parameterized_with=[],
        )

    def test_pulse_measure(
        self,
        two_tunable_transmon_platform,
        q_scalings,
        count,
        use_cal_traces,
        readout_lengths,
    ):
        """Test the properties of measure pulses."""
        verifier = create_qubitspec_verifier(
            two_tunable_transmon_platform,
            q_scalings,
            count,
            use_cal_traces,
            readout_lengths,
        )
        start_measure = _START_TIME + 2 * _LENGTH_GE_DRIVE
        verifier.assert_pulse(
            signal="q0/measure",
            index=0,
            start=start_measure,
            end=start_measure + readout_lengths[0],
        )
        verifier.assert_pulse(
            signal="q0/acquire",
            index=0,
            start=start_measure,
            end=start_measure + _LENGTH_ACQUIRE,
        )

        verifier.assert_pulse(
            signal="q1/measure",
            index=0,
            start=start_measure,
            end=start_measure + readout_lengths[1],
        )
        verifier.assert_pulse(
            signal="q1/acquire",
            index=0,
            start=start_measure,
            end=start_measure + _LENGTH_ACQUIRE,
        )
