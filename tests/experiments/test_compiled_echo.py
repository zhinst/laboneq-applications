# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for the compiled Hahn-Echo experiment using the testing utilities
provided by the LabOne Q Applications Library.
"""

import numpy as np
import pytest

from laboneq_applications.experiments import (
    echo,
)
from laboneq_applications.testing import CompiledExperimentVerifier

_LENGTH_GE = 32e-9
_LENGTH_EF = 64e-9


def on_system_grid(time, system_grid=8):
    # time_ns is time in ns, system_grid is minimum stepsize in ns
    time_ns = time * 1e9

    remainder = round(time_ns % system_grid, 2)
    if remainder != 0:
        time_ns = time_ns + system_grid - remainder
    return round(time_ns * 1e-9, 12)


def create_echo_verifier(
    tunable_transmon_platform,
    delays,
    count,
    transition,
    use_cal_traces,
    cal_states,
    readout_lengths=None,
):
    """Create a CompiledExperimentVerifier for the echo experiment."""
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
    options = echo.experiment_workflow.options()
    options.count(count)
    options.transition(transition)
    options.use_cal_traces(use_cal_traces)
    options.cal_states(cal_states)
    options.do_analysis(False)  # TODO: fix tests to work with do_analysis=True

    res = echo.experiment_workflow(
        session=session,
        qubits=qubits,
        qpu=tunable_transmon_platform.qpu,
        delays=delays,
        options=options,
    ).run()
    return CompiledExperimentVerifier(res.tasks["compile_experiment"].output)


@pytest.mark.parametrize(
    "transition",
    ["ge", "ef"],
)
@pytest.mark.parametrize(
    "use_cal_traces",
    [True, False],
)
@pytest.mark.parametrize(
    "cal_states",
    ["ge", "ef", "gef"],
)
class TestEchoSingleQubit:
    def test_pulse_count(
        self,
        single_tunable_transmon_platform,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the number of drive pulses."""
        delays = [1e-6, 2e-6, 3e-6]
        count = 2
        verifier = create_echo_verifier(
            single_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,
        )

        if transition == "ge":
            expected_drive_count_ge = count * (3 * len(delays))
            expected_drive_count_ef = 0
        elif transition == "ef":
            expected_drive_count_ge = count * len(delays)
            expected_drive_count_ef = count * (3 * len(delays))

        if cal_states == "ge":
            expected_drive_count_ge += count * int(use_cal_traces)
        elif cal_states in ("ef", "gef"):
            expected_drive_count_ge += count * 2 * int(use_cal_traces)
            expected_drive_count_ef += count * int(use_cal_traces)

        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive",
            expected_drive_count_ge,
        )
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive_ef",
            expected_drive_count_ef,
        )

        # Note that with cal_state on, there are additional measure pulses
        expected_measure_count = count * (len(delays))
        if cal_states in ("ge", "ef"):
            expected_measure_count += count * 2 * int(use_cal_traces)
        elif cal_states == "gef":
            expected_measure_count += count * 3 * int(use_cal_traces)

        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/measure",
            expected_measure_count,
        )

        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/acquire",
            expected_measure_count,
        )

    def test_pulse_drive(
        self,
        single_tunable_transmon_platform,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the properties of drive pulses."""
        # In this test, all the qubit ge drive pulses have lengths of 51ns,
        # and all the ef pulses have lengths of 52ns.

        delays = [1.032e-6, 2.0560e-6, 3.08e-6]
        count = 2
        verifier = create_echo_verifier(
            single_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,
        )
        if transition == "ge":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=0,
                start=0e-6,
                end=_LENGTH_GE,
                parameterized_with=[  # empty for echo: pulses are constant
                ],
            )
            verifier.assert_pulse_pair(
                signals="/logical_signal_groups/q0/drive",
                indices=[0, 1],
                distance=delays[0] / 2,
            )
            verifier.assert_pulse_pair(
                signals="/logical_signal_groups/q0/drive",
                indices=[1, 2],
                distance=delays[0] / 2,
            )
        elif transition == "ef":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=0,
                start=_LENGTH_GE,
                end=_LENGTH_GE + _LENGTH_EF,
                parameterized_with=[],
            )
            verifier.assert_pulse_pair(
                signals="/logical_signal_groups/q0/drive_ef",
                indices=[0, 1],
                distance=delays[0] / 2,
            )
            verifier.assert_pulse_pair(
                signals="/logical_signal_groups/q0/drive_ef",
                indices=[1, 2],
                distance=delays[0] / 2,
            )

    def test_pulse_measure(
        self,
        single_tunable_transmon_platform,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the properties of measure pulses.

        Here, we assert the start, end, and the parameterization of the pulses.

        """

        delays = [1.032e-6, 2.0560e-6, 3.08e-6]
        count = 2

        verifier = create_echo_verifier(
            single_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,
        )

        delay = delays[0] / 2

        if transition == "ge":
            measure_pulse_start = on_system_grid(3 * _LENGTH_GE + 2 * delay)
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start=measure_pulse_start,
                end=measure_pulse_start + 2e-6,
                tolerance=1e-9,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start=measure_pulse_start,
                end=measure_pulse_start + 2e-6,
                tolerance=1e-9,
            )
        elif transition == "ef":
            measure_pulse_start = on_system_grid(
                _LENGTH_GE + 3 * _LENGTH_EF + 2 * delay
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start=measure_pulse_start,
                end=measure_pulse_start + 2e-6,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start=measure_pulse_start,
                end=measure_pulse_start + 2e-6,
            )


@pytest.mark.parametrize(
    ("transition", "readout_lengths"),
    [("ge", [1e-6, 1e-6]), ("ef", [100e-9, 200e-9])],
)
@pytest.mark.parametrize(
    "use_cal_traces",
    [True, False],
)
@pytest.mark.parametrize(
    "cal_states",
    ["ge", "ef", "gef"],
)
class TestEchoTwoQubits:
    def test_pulse_count(  # noqa: C901
        self,
        two_tunable_transmon_platform,
        transition,
        use_cal_traces,
        cal_states,
        readout_lengths,
    ):
        """Test the number of drive pulses.

        `two_tunable_transmon` is a pytest fixture that is automatically
        imported into the test function.

        """
        delays = [[1e-6, 2e-6, 3e-6], [1e-6, 2e-6, 3e-6]]
        count = 2
        verifier = create_echo_verifier(
            two_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,
            readout_lengths,
        )

        if transition == "ge":
            expected_drive_count_ge = count * (3 * len(delays[0]))
            expected_drive_count_ef = 0
        elif transition == "ef":
            expected_drive_count_ge = count * len(delays[0])
            expected_drive_count_ef = count * (3 * len(delays[0]))

        if cal_states == "ge":
            expected_drive_count_ge += count * int(use_cal_traces)
        elif cal_states in ("ef", "gef"):
            expected_drive_count_ge += count * 2 * int(use_cal_traces)
            expected_drive_count_ef += count * int(use_cal_traces)

        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive",
            expected_drive_count_ge,
        )
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive_ef",
            expected_drive_count_ef,
        )
        # Check for q1
        if transition == "ge":
            expected_drive_count_ge = count * (3 * len(delays[1]))
            expected_drive_count_ef = 0
        elif transition == "ef":
            expected_drive_count_ge = count * len(delays[1])
            expected_drive_count_ef = count * (3 * len(delays[1]))

        if cal_states == "ge":
            expected_drive_count_ge += count * int(use_cal_traces)
        elif cal_states in ("ef", "gef"):
            expected_drive_count_ge += count * 2 * int(use_cal_traces)
            expected_drive_count_ef += count * int(use_cal_traces)

        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q1/drive",
            expected_drive_count_ge,
        )
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q1/drive_ef",
            expected_drive_count_ef,
        )

        # Check for q0
        # Note that with cal_state on, there are additional measure pulses
        expected_measure_count = count * (len(delays[0]))
        if cal_states in ("ge", "ef"):
            expected_measure_count += count * 2 * int(use_cal_traces)
        elif cal_states == "gef":
            expected_measure_count += count * 3 * int(use_cal_traces)

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
        # Note that with cal_state on, there are additional measure pulses
        expected_measure_count = count * (len(delays[1]))
        if cal_states in ("ge", "ef"):
            expected_measure_count += count * 2 * int(use_cal_traces)
        elif cal_states == "gef":
            expected_measure_count += count * 3 * int(use_cal_traces)

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
        transition,
        use_cal_traces,
        cal_states,
        readout_lengths,
    ):
        """Test the properties of drive pulses."""

        # In this test, all the qubit ge drive pulses have lengths of 51ns,
        # and all the ef pulses have lengths of 52ns.

        delays = [[1.032e-6, 2.0560e-6, 3.08e-6], [1.032e-6, 2.0560e-6, 3.08e-6]]
        count = 2
        verifier = create_echo_verifier(
            two_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,
            readout_lengths,
        )
        if transition == "ge":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=0,
                start=0e-6,
                end=_LENGTH_GE,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive",
                index=0,
                start=0e-6,
                end=_LENGTH_GE,
                parameterized_with=[],
            )
            verifier.assert_pulse_pair(
                signals="/logical_signal_groups/q0/drive",
                indices=[0, 1],
                distance=delays[0][0] / 2,
            )
            verifier.assert_pulse_pair(
                signals="/logical_signal_groups/q0/drive",
                indices=[1, 2],
                distance=delays[0][0] / 2,
            )

            verifier.assert_pulse_pair(
                signals="/logical_signal_groups/q1/drive",
                indices=[0, 1],
                distance=delays[1][0] / 2,
            )
            verifier.assert_pulse_pair(
                signals="/logical_signal_groups/q1/drive",
                indices=[1, 2],
                distance=delays[1][0] / 2,
            )
        elif transition == "ef":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=0,
                start=_LENGTH_GE,
                end=_LENGTH_GE + _LENGTH_EF,
                parameterized_with=[],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive_ef",
                index=0,
                start=_LENGTH_GE,
                end=_LENGTH_GE + _LENGTH_EF,
                parameterized_with=[],
            )
            verifier.assert_pulse_pair(
                signals="/logical_signal_groups/q0/drive_ef",
                indices=[0, 1],
                distance=delays[0][0] / 2,
            )
            verifier.assert_pulse_pair(
                signals="/logical_signal_groups/q0/drive_ef",
                indices=[1, 2],
                distance=delays[0][0] / 2,
            )

            verifier.assert_pulse_pair(
                signals="/logical_signal_groups/q1/drive_ef",
                indices=[0, 1],
                distance=delays[1][0] / 2,
            )
            verifier.assert_pulse_pair(
                signals="/logical_signal_groups/q1/drive_ef",
                indices=[1, 2],
                distance=delays[1][0] / 2,
            )

    def test_pulse_measure(
        self,
        two_tunable_transmon_platform,
        transition,
        use_cal_traces,
        cal_states,
        readout_lengths,
    ):
        """Test the properties of measure pulses.

        Here, we can assert the start, end, and the parameterization of the pulses.

        """
        delays = [[1.032e-6, 2.0560e-6, 3.08e-6], [1.032e-6, 2.0560e-6, 3.08e-6]]
        count = 2

        verifier = create_echo_verifier(
            two_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,
            readout_lengths,
        )

        if transition == "ge":
            # Check for q0
            delay = delays[0][0] / 2
            measure_start_time = on_system_grid(3 * _LENGTH_GE + 2 * delay)
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start=measure_start_time,
                end=measure_start_time + readout_lengths[0],
                tolerance=1e-9,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start=measure_start_time,
                end=measure_start_time + 2e-6,
                tolerance=1e-9,
            )
            # Check for q1
            delay = delays[1][0] / 2
            measure_start_time = on_system_grid(3 * _LENGTH_GE + 2 * delay)
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/measure",
                index=0,
                start=measure_start_time,
                end=measure_start_time + readout_lengths[1],
                tolerance=1e-9,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/acquire",
                index=0,
                start=measure_start_time,
                end=measure_start_time + 2e-6,
                tolerance=1e-9,
            )
        elif transition == "ef":
            # Check for q0
            delay = delays[0][0] / 2
            measure_start_time = on_system_grid(_LENGTH_GE + 3 * _LENGTH_EF + 2 * delay)
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start=measure_start_time,
                end=measure_start_time + readout_lengths[0],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start=measure_start_time,
                end=measure_start_time + 2e-6,
            )
            # Check for q1
            delay = delays[1][0] / 2
            measure_start_time = on_system_grid(_LENGTH_GE + 3 * _LENGTH_EF + 2 * delay)
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/measure",
                index=0,
                start=measure_start_time,
                end=measure_start_time + readout_lengths[1],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/acquire",
                index=0,
                start=measure_start_time,
                end=measure_start_time + 2e-6,
                tolerance=1e-9,
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
    options = echo.experiment_workflow.options()
    options.transition(transition)
    options.cal_states(cal_states)
    options.active_reset(True)
    options.active_reset_states(active_reset_states)
    options.active_reset_repetitions(active_reset_repetitions)
    options.do_analysis(False)
    [q0] = single_tunable_transmon_platform.qpu.qubits
    delays = np.linspace(0, 1e-6, 11)
    workflow_result = echo.experiment_workflow(
        session=single_tunable_transmon_platform.session(do_emulation=True),
        qubits=q0,
        qpu=single_tunable_transmon_platform.qpu,
        delays=delays,
        options=options,
    ).run()

    exp = workflow_result.tasks["create_experiment"].output
    active_reset_section = exp.sections[0].children[0].children[0]
    assert active_reset_section.uid == "active_reset_q0_0"
    truth_len = len(q0.signals) + active_reset_repetitions
    assert len(active_reset_section.children) == truth_len

    data = workflow_result.output
    assert "active_reset" in data.q0
    shape_truth = (
        (len(delays), active_reset_repetitions)
        if active_reset_repetitions > 1
        else (len(delays),)
    )
    assert np.shape(data.q0.active_reset.result.data) == shape_truth
    for s in cal_states:
        cal_trace_data = data.q0.active_reset.cal_trace[s].data
        if active_reset_repetitions == 1:
            assert isinstance(cal_trace_data, np.complex128)
        else:
            assert len(cal_trace_data) == active_reset_repetitions


@pytest.mark.parametrize(
    ("transition", "cal_states", "active_reset_states"),
    [("ge", "ge", "ge"), ("ef", "ef", "gef")],
)
@pytest.mark.parametrize(
    "active_reset_repetitions",
    [1, 5],
)
def test_two_qubit_run_with_active_reset(
    two_tunable_transmon_platform,
    transition,
    cal_states,
    active_reset_states,
    active_reset_repetitions,
):
    options = echo.experiment_workflow.options()
    options.transition(transition)
    options.cal_states(cal_states)
    options.active_reset(True)
    options.active_reset_states(active_reset_states)
    options.active_reset_repetitions(active_reset_repetitions)
    options.do_analysis(False)
    qubits = two_tunable_transmon_platform.qpu.qubits
    delays = [np.linspace(0, 1e-6, 11), np.linspace(0, 5e-6, 11)]
    workflow_result = echo.experiment_workflow(
        session=two_tunable_transmon_platform.session(do_emulation=True),
        qubits=qubits,
        qpu=two_tunable_transmon_platform.qpu,
        delays=delays,
        options=options,
    ).run()

    exp = workflow_result.tasks["create_experiment"].output
    active_reset_section = exp.sections[0].children[0].children[0]
    assert active_reset_section.uid == "active_reset_q0_q1_0"
    truth_len = len(qubits[0].signals) * len(qubits) + active_reset_repetitions
    assert len(active_reset_section.children) == truth_len

    data = workflow_result.output
    assert "active_reset" in data.q0
    assert "active_reset" in data.q1
    shape_truth = (
        (len(delays[0]), active_reset_repetitions)
        if active_reset_repetitions > 1
        else (len(delays[0]),)
    )
    assert np.shape(data.q0.active_reset.result.data) == shape_truth
    assert np.shape(data.q1.active_reset.result.data) == shape_truth
    for s in cal_states:
        for q in qubits:
            cal_trace_data = data[q.uid].active_reset.cal_trace[s].data
            if active_reset_repetitions == 1:
                assert isinstance(cal_trace_data, np.complex128)
            else:
                assert len(cal_trace_data) == active_reset_repetitions


def test_invalid_averaging_mode(single_tunable_transmon_platform):
    [q0] = single_tunable_transmon_platform.qpu.qubits
    session = single_tunable_transmon_platform.session(do_emulation=True)
    options = echo.experiment_workflow.options()
    options.averaging_mode("sequential")
    options.use_cal_traces(True)
    options.do_analysis(False)

    with pytest.raises(ValueError) as err:
        echo.experiment_workflow(
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
