"""Tests for the compiled lifetime_measurement experiments.
The followings are checked:
- The number of measure, drive and acquire pulses.
- The length of measure, drive and acquire pulses.
- In lifetime_measurement experiments, the delays between the excitation pulses
and measure are most critical and should be verified to be
equal to `delays`.
"""

import numpy as np
import pytest

from laboneq_applications.experiments import (
    lifetime_measurement,
)
from laboneq_applications.testing import CompiledExperimentVerifier

_COUNT = 2  # The number of averaging in experiments
_LENGTH_GE = 51e-9
_LENGTH_EF = 52e-9
_LENGTH_MEASURE = 2e-6


def create_T1_verifier(  # noqa: N802
    tunable_transmon_platform,
    delays,
    count,
    transition,
    use_cal_traces,
    cal_states,
):
    """Create a CompiledExperimentVerifier for the lifetime_measurement experiment."""
    qubits = tunable_transmon_platform.qpu.qubits
    if len(qubits) == 1:
        qubits = qubits[0]
    session = tunable_transmon_platform.session(do_emulation=True)
    options = lifetime_measurement.experiment_workflow.options()
    options.count(count)
    options.transition(transition)
    options.use_cal_traces(use_cal_traces)
    options.cal_states(cal_states)
    # TODO: fix tests to work with do_analysis=True when the new options feature is in
    options.do_analysis(False)

    res = lifetime_measurement.experiment_workflow(
        session=session,
        qubits=qubits,
        qpu=tunable_transmon_platform.qpu,
        delays=delays,
        options=options,
    ).run()
    return CompiledExperimentVerifier(res.tasks["compile_experiment"].output)


@pytest.mark.parametrize(
    "delays",
    [
        [56e-9, 112e-9, 224e-9, 448e-9],
    ],
)
@pytest.mark.parametrize("transition", ["ge", "ef"], ids=["trans_ge", "trans_ef"])
@pytest.mark.parametrize(
    ["use_cal_traces", "cal_states"],
    [[True, "ge"], [False, "ge"]],
    ids=["use_cal_traces_ge", "no_cal_traces_ge"],
)
class TestT1SingleQubit:
    def test_pulse_count(
        self,
        single_tunable_transmon_platform,
        delays,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the number of drive pulses."""
        verifier = create_T1_verifier(
            single_tunable_transmon_platform,
            delays,
            _COUNT,
            transition,
            use_cal_traces,
            cal_states,
        )

        expected_drive_count = _COUNT * (len(delays) + int(use_cal_traces))
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive",
            expected_drive_count,
        )

        if transition == "ef":
            expected_drive_count = _COUNT * len(delays)
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive_ef",
                expected_drive_count,
            )

        verifier = create_T1_verifier(
            single_tunable_transmon_platform,
            delays,
            _COUNT,
            transition,
            use_cal_traces,
            cal_states,
        )
        # Note that with cal_state on, there are 2 additional measure pulses
        expected_measure_count = _COUNT * (len(delays) + 2 * int(use_cal_traces))
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/measure",
            expected_measure_count,
        )

        # acquire and measure pulses have the same _COUNT
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/acquire",
            expected_measure_count,
        )

    def test_pulse_drive(
        self,
        single_tunable_transmon_platform,
        delays,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the properties of drive pulses."""

        verifier = create_T1_verifier(
            single_tunable_transmon_platform,
            delays,
            _COUNT,
            transition,
            use_cal_traces,
            cal_states,
        )

        # the start times of drive pulses are not critical
        if transition == "ge":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=0,
                length=_LENGTH_GE,
            )
            # test pulse and measure distance
            verifier.assert_pulse_pair(
                signals=[
                    "/logical_signal_groups/q0/drive",
                    "/logical_signal_groups/q0/measure",
                ],
                indices=[0, 0],
                distance=delays[0],
            )
        elif transition == "ef":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=0,
                length=_LENGTH_EF,
            )
            verifier.assert_pulse_pair(
                signals=[
                    "/logical_signal_groups/q0/drive_ef",
                    "/logical_signal_groups/q0/measure",
                ],
                indices=[0, 0],
                distance=delays[0],
            )

    def test_pulse_measure(
        self,
        single_tunable_transmon_platform,
        delays,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the properties of measure pulses."""

        verifier = create_T1_verifier(
            single_tunable_transmon_platform,
            delays,
            _COUNT,
            transition,
            use_cal_traces,
            cal_states,
        )

        verifier.assert_pulse(
            signal="/logical_signal_groups/q0/measure",
            index=0,
            length=_LENGTH_MEASURE,
        )
        verifier.assert_pulse(
            signal="/logical_signal_groups/q0/acquire",
            index=0,
            length=_LENGTH_MEASURE,
        )

        verifier.assert_pulse_pair(
            signals=[
                "/logical_signal_groups/q0/measure",
                "/logical_signal_groups/q0/acquire",
            ],
            indices=[0, 0],
            start=0,
        )


@pytest.mark.parametrize(
    "delays",
    [
        [[56e-9, 112e-9, 224e-9, 448e-9], [58e-9, 112e-9, 224e-9, 448e-9]],
    ],
)
@pytest.mark.parametrize("transition", ["ge", "ef"], ids=["trans_ge", "trans_ef"])
@pytest.mark.parametrize(
    ["use_cal_traces", "cal_states"],
    [[True, "ge"], [False, "ge"]],
    ids=["use_cal_traces_ge", "no_cal_traces_ge"],
)
class TestT1TwoQubits:
    def test_pulse_count(
        self,
        two_tunable_transmon_platform,
        delays,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the number of pulses."""
        verifier = create_T1_verifier(
            two_tunable_transmon_platform,
            delays,
            _COUNT,
            transition,
            use_cal_traces,
            cal_states,
        )
        expected_drive_count = _COUNT * (len(delays[0]) + int(use_cal_traces))
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive",
            expected_drive_count,
        )

        expected_drive_count = _COUNT * (len(delays[1]) + int(use_cal_traces))
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q1/drive",
            expected_drive_count,
        )

        if transition == "ef":
            expected_drive_count = _COUNT * len(delays[0])

            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive_ef",
                expected_drive_count,
            )

            expected_drive_count = _COUNT * len(delays[1])

            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q1/drive_ef",
                expected_drive_count,
            )

        # Note that with cal_state on, there are 2 additional measure pulses
        expected_measure_count = _COUNT * (len(delays[0]) + 2 * int(use_cal_traces))

        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/measure",
            expected_measure_count,
        )

        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/acquire",
            expected_measure_count,
        )

        # Check for q1
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q1/measure",
            expected_measure_count,
        )

        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q1/acquire",
            expected_measure_count,
        )

    def test_pulse_drive(
        self,
        two_tunable_transmon_platform,
        delays,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the properties of drive pulses."""
        verifier = create_T1_verifier(
            two_tunable_transmon_platform,
            delays,
            _COUNT,
            transition,
            use_cal_traces,
            cal_states,
        )
        if transition == "ge":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=0,
                length=_LENGTH_GE,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive",
                index=0,
                length=_LENGTH_GE,
                parameterized_with=[],
            )
            verifier.assert_pulse_pair(
                signals=[
                    "/logical_signal_groups/q0/drive",
                    "/logical_signal_groups/q0/measure",
                ],
                indices=[0, 0],
                distance=delays[0][0],
            )
            verifier.assert_pulse_pair(
                signals=[
                    "/logical_signal_groups/q1/drive",
                    "/logical_signal_groups/q1/measure",
                ],
                indices=[0, 0],
                distance=delays[1][0],
            )

        elif transition == "ef":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=0,
                length=_LENGTH_EF,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive_ef",
                index=0,
                length=_LENGTH_EF,
                parameterized_with=[],
            )
            verifier.assert_pulse_pair(
                signals=[
                    "/logical_signal_groups/q0/drive_ef",
                    "/logical_signal_groups/q0/measure",
                ],
                indices=[0, 0],
                distance=delays[0][0],
            )
            verifier.assert_pulse_pair(
                signals=[
                    "/logical_signal_groups/q1/drive_ef",
                    "/logical_signal_groups/q1/measure",
                ],
                indices=[0, 0],
                distance=delays[1][0],
            )

    def test_pulse_measure(
        self,
        two_tunable_transmon_platform,
        delays,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the properties of measure pulses."""
        verifier = create_T1_verifier(
            two_tunable_transmon_platform,
            delays,
            _COUNT,
            transition,
            use_cal_traces,
            cal_states,
        )

        if transition == "ge":
            # Check for q0
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                length=_LENGTH_MEASURE,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                length=_LENGTH_MEASURE,
            )

            # Check for q1
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/measure",
                index=0,
                length=_LENGTH_MEASURE,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/acquire",
                index=0,
                length=_LENGTH_MEASURE,
            )
            verifier.assert_pulse_pair(
                signals=[
                    "/logical_signal_groups/q0/measure",
                    "/logical_signal_groups/q0/acquire",
                ],
                indices=[0, 0],
                start=0,
            )
            verifier.assert_pulse_pair(
                signals=[
                    "/logical_signal_groups/q1/measure",
                    "/logical_signal_groups/q1/acquire",
                ],
                indices=[0, 0],
                start=0,
            )


def test_invalid_averaging_mode(single_tunable_transmon_platform):
    [q0] = single_tunable_transmon_platform.qpu.qubits
    session = single_tunable_transmon_platform.session(do_emulation=True)
    options = lifetime_measurement.experiment_workflow.options()
    options.averaging_mode("sequential")
    options.use_cal_traces(True)
    options.do_analysis(False)

    with pytest.raises(ValueError) as err:
        lifetime_measurement.experiment_workflow(
            session=session,
            qubits=q0,
            qpu=single_tunable_transmon_platform.qpu,
            delays=np.linspace(0, 10e-6, 10),
            options=options,
        ).run()

    assert str(err.value) == (
        "'AveragingMode.SEQUENTIAL' (or {AveragingMode.SEQUENTIAL}) cannot be used "
        "with calibration traces because the calibration traces are added "
        "outside the sweep."
    )
