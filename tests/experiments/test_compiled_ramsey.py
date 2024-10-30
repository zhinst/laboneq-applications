"""Tests for the compiled ramsey experiment using the testing utilities
provided by the LabOne Q Applications Library.
"""

from typing import ClassVar

import numpy as np
import pytest

from laboneq_applications.experiments import ramsey
from laboneq_applications.testing import CompiledExperimentVerifier


def create_ramsey_verifier(
    tunable_transmon_platform,
    delays,
    count,
    transition,
    use_cal_traces,
    readout_lengths=None,
):
    """Create a CompiledExperimentVerifier for the ramsey experiment."""
    qubits = tunable_transmon_platform.qpu.qubits
    if len(qubits) == 1:
        qubits = qubits[0]
    if readout_lengths is not None:
        assert len(readout_lengths) == len(qubits)
        for i, rl in enumerate(readout_lengths):
            qubits[i].parameters.readout_length = rl
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
        options=options,
    ).run()
    return CompiledExperimentVerifier(res.tasks["compile_experiment"].output)


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
class TestRamseySingleQubit:
    _DELAYS = tuple(0.1e-6 * i for i in range(1, 11))

    def test_pulse_count_drive(
        self,
        single_tunable_transmon_platform,
        count,
        transition,
        use_cal_traces,
    ):
        """Test the number of drive pulses.`"""
        verifier = create_ramsey_verifier(
            single_tunable_transmon_platform,
            self._DELAYS,
            count,
            transition,
            use_cal_traces,
        )

        # with cal_state on, there is 1 additional drive pulse
        if transition == "ge":
            expected_drive_count = count * (2 * len(self._DELAYS) + int(use_cal_traces))
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive",
                expected_drive_count,
            )

        if transition == "ef":
            expected_drive_count = count * 2 * len(self._DELAYS)
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive_ef",
                expected_drive_count,
            )

    def test_pulse_count_measure_acquire(
        self,
        single_tunable_transmon_platform,
        count,
        transition,
        use_cal_traces,
    ):
        """Test the number of measure and acquire pulses."""
        verifier = create_ramsey_verifier(
            single_tunable_transmon_platform,
            self._DELAYS,
            count,
            transition,
            use_cal_traces,
        )
        # with cal_state on, there are 2 additional measure pulses
        expected_measure_count = count * (len(self._DELAYS) + 2 * int(use_cal_traces))
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
        transition,
        use_cal_traces,
    ):
        """Test the properties of drive pulses."""
        [q0] = single_tunable_transmon_platform.qpu.qubits
        q0_pulse_length_ge = q0.parameters.ge_drive_length
        q0_pulse_length_ef = q0.parameters.ef_drive_length
        verifier = create_ramsey_verifier(
            single_tunable_transmon_platform,
            self._DELAYS,
            count,
            transition,
            use_cal_traces,
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
                start=offset + q0_pulse_length_ge + self._DELAYS[0],
                end=offset + 2 * q0_pulse_length_ge + self._DELAYS[0],
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
                start=start_ef + q0_pulse_length_ef + self._DELAYS[0],
                end=start_ef + 2 * q0_pulse_length_ef + self._DELAYS[0],
                parameterized_with=["x90_phases_q0"],
            )

    def test_pulse_measure(
        self,
        single_tunable_transmon_platform,
        count,
        transition,
        use_cal_traces,
    ):
        """Test the properties of measure pulses."""
        verifier = create_ramsey_verifier(
            single_tunable_transmon_platform,
            self._DELAYS,
            count,
            transition,
            use_cal_traces,
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
    ("count", "readout_lengths"),
    [(2, [1e-6, 1e-6]), (4, [100e-9, 200e-9])],
)
@pytest.mark.parametrize(
    "transition",
    ["ge", "ef"],
)
@pytest.mark.parametrize(
    "use_cal_traces",
    [True, False],
)
class TestRamseyTwoQubits:
    _DELAYS: ClassVar = [
        [0.1e-6 * i for i in range(1, 11)],
        [0.1e-6 * i for i in range(1, 11)],
    ]  # validate_and_convert_sweeps_to_arrays requires a list

    def test_pulse_count(
        self,
        two_tunable_transmon_platform,
        count,
        transition,
        use_cal_traces,
        readout_lengths,
    ):
        """Test the number of pulses."""
        verifier = create_ramsey_verifier(
            two_tunable_transmon_platform,
            self._DELAYS,
            count,
            transition,
            use_cal_traces,
            readout_lengths,
        )

        # with cal_state on, there is 1 additional drive pulse
        if transition == "ge":
            expected_drive_count = count * (
                2 * len(self._DELAYS[0]) + int(use_cal_traces)
            )
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive",
                expected_drive_count,
            )

            expected_drive_count = count * (
                2 * len(self._DELAYS[1]) + int(use_cal_traces)
            )
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q1/drive",
                expected_drive_count,
            )

        if transition == "ef":
            expected_drive_count = count * (len(self._DELAYS[0]) + int(use_cal_traces))
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive",
                expected_drive_count,
            )

            expected_drive_count = count * (len(self._DELAYS[1]) + int(use_cal_traces))
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q1/drive",
                expected_drive_count,
            )

            expected_drive_count = count * (2 * len(self._DELAYS[0]))
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive_ef",
                expected_drive_count,
            )

            expected_drive_count = count * (2 * len(self._DELAYS[1]))
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q1/drive_ef",
                expected_drive_count,
            )

        # with cal_state on, there are 2 additional measure pulses
        expected_measure_count = count * (
            len(self._DELAYS[0]) + 2 * int(use_cal_traces)
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
        transition,
        use_cal_traces,
        readout_lengths,
    ):
        """Test the properties of drive pulses."""
        [q0, q1] = two_tunable_transmon_platform.qpu.qubits
        q0_pulse_length_ge = q0.parameters.ge_drive_length
        q0_pulse_length_ef = q0.parameters.ef_drive_length
        q1_pulse_length_ge = q1.parameters.ge_drive_length
        q1_pulse_length_ef = q1.parameters.ef_drive_length

        verifier = create_ramsey_verifier(
            two_tunable_transmon_platform,
            self._DELAYS,
            count,
            transition,
            use_cal_traces,
            readout_lengths,
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
                start=offset + q0_pulse_length_ge + self._DELAYS[0][0],
                end=offset + 2 * q0_pulse_length_ge + self._DELAYS[0][0],
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
                start=offset + q1_pulse_length_ge + self._DELAYS[0][0],
                end=offset + 2 * q1_pulse_length_ge + self._DELAYS[0][0],
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
                start=start_ef_q0 + q0_pulse_length_ef + self._DELAYS[0][0],
                end=start_ef_q0 + 2 * q0_pulse_length_ef + self._DELAYS[0][0],
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
                start=start_ef_q1 + q1_pulse_length_ef + self._DELAYS[1][0],
                end=start_ef_q1 + 2 * q1_pulse_length_ef + self._DELAYS[0][0],
                parameterized_with=["x90_phases_q1"],
            )

    def test_pulse_measure(
        self,
        two_tunable_transmon_platform,
        count,
        transition,
        use_cal_traces,
        readout_lengths,
    ):
        """Test the properties of measure pulses."""
        verifier = create_ramsey_verifier(
            two_tunable_transmon_platform,
            self._DELAYS,
            count,
            transition,
            use_cal_traces,
            readout_lengths,
        )

        if transition == "ge":
            # See the explanation for the hardcoding of start_measure
            # in the single-qubit tests
            start_measure = 208e-9
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start=start_measure,
                end=start_measure + readout_lengths[0],
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
                end=start_measure + readout_lengths[1],
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
                end=start_measure + readout_lengths[0],
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
                end=start_measure + readout_lengths[1],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/acquire",
                index=0,
                start=start_measure,
                end=start_measure + 2e-6,
            )


def test_invalid_averaging_mode(single_tunable_transmon_platform):
    [q0] = single_tunable_transmon_platform.qpu.qubits
    session = single_tunable_transmon_platform.session(do_emulation=True)
    options = ramsey.experiment_workflow.options()
    options.averaging_mode("sequential")
    options.use_cal_traces(True)
    options.do_analysis(False)

    with pytest.raises(ValueError) as err:
        ramsey.experiment_workflow(
            session=session,
            qubits=q0,
            qpu=single_tunable_transmon_platform.qpu,
            delays=np.linspace(0, 10e-6, 10),
            detunings=0.67e6,
            options=options,
        ).run()

    assert str(err.value) == (
        "'AveragingMode.SEQUENTIAL' (or {AveragingMode.SEQUENTIAL}) cannot be used "
        "with calibration traces because the calibration traces are added "
        "outside the sweep."
    )
