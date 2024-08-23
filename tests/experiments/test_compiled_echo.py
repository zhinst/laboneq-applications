"""Tests for the compiled Hahn-Echo experiment using the testing utilities
provided by the LabOne Q Applications Library.
"""

import pytest

from laboneq_applications.experiments import (
    echo,
)
from laboneq_applications.testing import CompiledExperimentVerifier


def on_system_grid(time, system_grid = 8):
    # time_ns is time in ns, system_grid is minimum stepsize in ns
    time_ns = time *1e9

    remainder = round(time_ns % system_grid, 2)
    if remainder != 0.0:
        time_ns = time_ns + system_grid - remainder
    return round(time_ns*1e-9, 12)

def create_echo_verifier(
    tunable_transmon_platform,
    delays,
    count,
    transition,
    use_cal_traces,
    cal_states,
):
    """Create a CompiledExperimentVerifier for the echo experiment."""
    qubits = tunable_transmon_platform.qpu.qubits
    if len(qubits) == 1:
        qubits = qubits[0]
    session = tunable_transmon_platform.session(do_emulation=True)
    options = echo.options()
    options.create_experiment.count = count

    options.create_experiment.transition = transition
    options.create_experiment.use_cal_traces = use_cal_traces
    options.create_experiment.cal_states = cal_states

    res = echo.experiment_workflow(
        session=session,
        qubits=qubits,
        qpu=tunable_transmon_platform.qpu,
        delays=delays,
        options=options,
    ).run()
    return CompiledExperimentVerifier(res.tasks["compile_experiment"].output)


### Single-Qubit Tests ###


# use pytest.mark.parametrize to generate test cases for
# all combinations of the parameters.
@pytest.mark.parametrize(
    "delays",
    [
        [1e-6, 2e-6],
        [1e-6, 2e-6, 3e-6],
    ],
)
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
@pytest.mark.parametrize(
    "cal_states",
    ["ge", "ef", "gef"],
)


class TestEchoSingleQubit:
    def test_pulse_count_drive(
        self,
        single_tunable_transmon_platform,
        delays,
        count,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the number of drive pulses.

        `single_tunable_transmon` is a pytest fixture that is automatically
        imported into the test function.

        """
        # create a verifier for the experiment

        verifier = create_echo_verifier(
            single_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,
        )

        # The signal names can be looked up in device_setup,
        # but typically they are in the form of
        # /logical_signal_groups/q{i}/drive(measure/acquire/drive_ef)
        # echo: x90 y180 x90
        # Note that with cal_state on, there is 1 additional drive pulse.

        # drive count from echo experiment
        if transition == "ge":
            expected_drive_count_ge = count * (3 * len(delays))
            expected_drive_count_ef = 0
        elif transition == "ef":
            expected_drive_count_ge = count * len(delays)
            expected_drive_count_ef = count * (3 * len(delays))

        if cal_states == "ge":
            expected_drive_count_ge += count * int(use_cal_traces)
        elif cal_states in ("ef","gef"):
            expected_drive_count_ge += count * 2*int(use_cal_traces)
            expected_drive_count_ef += count * int(use_cal_traces)

        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive",
            expected_drive_count_ge,
        )
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive_ef",
            expected_drive_count_ef,
        )

    def test_pulse_count_measure_acquire(
        self,
        single_tunable_transmon_platform,
        delays,
        count,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the number of measure and acquire pulses."""

        verifier = create_echo_verifier(
            single_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,
        )
        # Note that with cal_state on, there are additional measure pulses
        expected_measure_count = count * (len(delays))
        if cal_states in ("ge","ef"):
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

    def test_pulse_drive(
        self,
        single_tunable_transmon_platform,
        delays,
        count,
        transition,
        use_cal_traces,
        cal_states,
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


        verifier = create_echo_verifier(
            single_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,
        )
        if transition == "ge":
            # ge pulses
            # Here, we give an example of verifying the first drive pulse of q0
            # More pulses should be tested in a similar way

            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=0,
                start=0e-6,
                end=51e-9,
                parameterized_with=[ # empty for echo: pulses are constant
                ],
            )
        elif transition == "ef":

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
        delays,
        count,
        transition,
        use_cal_traces,
        cal_states,
    ):
        """Test the properties of measure pulses.

        Here, we assert the start, end, and the parameterization of the pulses.

        """

        verifier = create_echo_verifier(
            single_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,
        )

        delay = delays[0]/2

        if transition == "ge":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start= on_system_grid(51e-9 + delay +
                    51e-9 + delay + 51e-9),
                end= on_system_grid(51e-9 + delay +
                    51e-9 + delay + 51e-9)  + 2e-6,
                tolerance = 1e-9,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start= on_system_grid(51e-9 + delay +
                    51e-9 + delay + 51e-9),
                end= on_system_grid(51e-9 + delay +
                    51e-9 + delay + 51e-9)  + 2e-6,
                tolerance = 1e-9,
            )
        elif transition == "ef":

            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start= 56e-9 + on_system_grid(52e-9)+on_system_grid(delay) +
                    on_system_grid(52e-9) + on_system_grid(delay)
                    + on_system_grid(52e-9),
                end= 56e-9 + on_system_grid(52e-9)+on_system_grid(delay) +
                    on_system_grid(52e-9) + on_system_grid(delay)
                    + on_system_grid(52e-9) + 2e-6, # why extra 8 ns?
                tolerance = 1e-9,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start= 56e-9 + on_system_grid(52e-9)+on_system_grid(delay) +
                    on_system_grid(52e-9) + on_system_grid(delay)
                    + on_system_grid(52e-9),
                end= 56e-9 + on_system_grid(52e-9)+on_system_grid(delay) +
                    on_system_grid(52e-9) + on_system_grid(delay)
                    + on_system_grid(52e-9) + 2e-6,
                tolerance = 1e-9,
            )


# use pytest.mark.parametrize to generate test cases for
# all combinations of the parameters.
@pytest.mark.parametrize(
    "delays",
    [
        [[1e-6, 2e-6],[1e-6, 2e-6]],
        [[1e-6, 2e-6, 3e-6],[1e-6, 2e-6, 3e-6]],
    ],
)
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
@pytest.mark.parametrize(
    "cal_states",
    ["ge", "ef", "gef"],
)

class TestEchoTwoQubits:
    def test_pulse_count_drive(
        self,
        two_tunable_transmon_platform,
        delays,
        count,
        transition,
        use_cal_traces,
        cal_states,

    ):
        """Test the number of drive pulses.

        `two_tunable_transmon` is a pytest fixture that is automatically
        imported into the test function.

        """
        # create a verifier for the experiment
        verifier = create_echo_verifier(
            two_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,

        )

        # The signal names can be looked up in device_setup,
        # but typically they are in the form of
        # /logical_signal_groups/q{i}/drive(measure/acquire/drive_ef)
        # Note that with cal_state on, there are additional drive pulse.
        # drive count from echo experiment
        # Check for q0
        if transition == "ge":
            expected_drive_count_ge = count * (3 * len(delays[0]))
            expected_drive_count_ef = 0
        elif transition == "ef":
            expected_drive_count_ge = count * len(delays[0])
            expected_drive_count_ef = count * (3 * len(delays[0]))

        if cal_states == "ge":
            expected_drive_count_ge += count * int(use_cal_traces)
        elif cal_states in ("ef","gef"):
            expected_drive_count_ge += count * 2*int(use_cal_traces)
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
        elif cal_states in ("ef","gef"):
            expected_drive_count_ge += count * 2*int(use_cal_traces)
            expected_drive_count_ef += count * int(use_cal_traces)

        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q1/drive",
            expected_drive_count_ge,
        )
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q1/drive_ef",
            expected_drive_count_ef,
        )

    def test_pulse_count_measure_acquire(
        self,
        two_tunable_transmon_platform,
        delays,
        count,
        transition,
        use_cal_traces,
        cal_states,

    ):
        """Test the number of measure and acquire pulses."""

        verifier = create_echo_verifier(
            two_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,

        )
        # Check for q0
        # Note that with cal_state on, there are additional measure pulses
        expected_measure_count = count * (len(delays[0]))
        if cal_states in ("ge","ef"):
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
        if cal_states in ("ge","ef"):
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
        delays,
        count,
        transition,
        use_cal_traces,
        cal_states,

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

        verifier = create_echo_verifier(
            two_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,

        )
        if transition == "ge":
            # Here, we give an example of verifying the first drive pulse of q0
            # More pulses should be tested in a similar way
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=0,
                start=0e-6,
                end=51e-9,
                parameterized_with=[
                ],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive",
                index=0,
                start=0e-6,
                end=51e-9,
                parameterized_with=[
                ],
            )
        elif transition == "ef":

            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=0,
                start=56e-9,
                end=56e-9 + 52e-9,
                parameterized_with=[
                ],
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/drive_ef",
                index=0,
                start=56e-9,
                end=56e-9 + 52e-9,
                parameterized_with=[
                ],
            )

    def test_pulse_measure(
        self,
        two_tunable_transmon_platform,
        delays,
        count,
        transition,
        use_cal_traces,
        cal_states,

    ):
        """Test the properties of measure pulses.

        Here, we can assert the start, end, and the parameterization of the pulses.

        """

        verifier = create_echo_verifier(
            two_tunable_transmon_platform,
            delays,
            count,
            transition,
            use_cal_traces,
            cal_states,

        )

        if transition == "ge":
            # Check for q0
            delay = delays[0][0]/2
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start= on_system_grid(51e-9 + delay +
                    51e-9 + delay + 51e-9),
                end= on_system_grid(51e-9 + delay +
                    51e-9 + delay + 51e-9) + 2e-6,
                tolerance = 1e-9,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start= on_system_grid(51e-9 + delay +
                    51e-9 + delay + 51e-9),
                end= on_system_grid(51e-9 + delay +
                    51e-9 + delay + 51e-9) + 2e-6,
                tolerance = 1e-9,
            )
            delay = delays[1][0]/2
            # Check for q1
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/measure",
                index=0,
                start= on_system_grid(51e-9 + delay +
                    51e-9 + delay + 51e-9),
                end= on_system_grid(51e-9 + delay +
                    51e-9 + delay + 51e-9) + 2e-6,
                tolerance = 1e-9,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/acquire",
                index=0,
                start= on_system_grid(51e-9 + delay +
                    51e-9 + delay + 51e-9),
                end= on_system_grid(51e-9 + delay +
                    51e-9 + delay + 51e-9) + 2e-6,
                tolerance = 1e-9,
            )
        elif transition == "ef":
            # Check for q0
            delay = delays[0][0]/2
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start= 56e-9 + on_system_grid(52e-9)+on_system_grid(delay) +
                    on_system_grid(52e-9) + on_system_grid(delay)
                    + on_system_grid(52e-9),
                end= 56e-9 + on_system_grid(52e-9)+on_system_grid(delay) +
                    on_system_grid(52e-9) + on_system_grid(delay)
                    + on_system_grid(52e-9) + 2e-6,
                tolerance = 1e-9,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0,
                start= 56e-9 + on_system_grid(52e-9)+on_system_grid(delay) +
                    on_system_grid(52e-9) + on_system_grid(delay)
                    + on_system_grid(52e-9),
                end= 56e-9 + on_system_grid(52e-9)+on_system_grid(delay) +
                    on_system_grid(52e-9) + on_system_grid(delay)
                    + on_system_grid(52e-9) + 2e-6,
                tolerance = 1e-9,
            )
            # Check for q1
            delay = delays[1][0]/2
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/measure",
                index=0,
                start= 56e-9 + on_system_grid(52e-9)+on_system_grid(delay) +
                    on_system_grid(52e-9) + on_system_grid(delay)
                    + on_system_grid(52e-9),
                end= 56e-9 + on_system_grid(52e-9)+on_system_grid(delay) +
                    on_system_grid(52e-9) + on_system_grid(delay)
                    + on_system_grid(52e-9) + 2e-6,
                tolerance = 1e-9,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q1/acquire",
                index=0,
                start= 56e-9 + on_system_grid(52e-9)+on_system_grid(delay) +
                    on_system_grid(52e-9) + on_system_grid(delay)
                    + on_system_grid(52e-9),
                end= 56e-9 + on_system_grid(52e-9)+on_system_grid(delay) +
                    on_system_grid(52e-9) + on_system_grid(delay)
                    + on_system_grid(52e-9) + 2e-6,
                tolerance = 1e-9,
            )
