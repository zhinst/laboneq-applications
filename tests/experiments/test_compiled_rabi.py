"""Tests for the compiled amplitude_rabi experiment using testing utilities
provided by the LabOne Q applications.
These tests could be used as references for others to write tests for their
contributions.
"""

import pytest
from laboneq.simple import Session

from laboneq_applications.experiments import amplitude_rabi
from laboneq_applications.qpu_types.tunable_transmon import TunableTransmonOperations
from laboneq_applications.testing import CompiledExperimentVerifier


def create_rabi_verifier(
    single_tunable_transmon,
    amplitudes,
    count,
    transition,
    use_cal_traces,
):
    """Create a CompiledExperimentVerifier for the amplitude rabi experiment."""
    session = Session(single_tunable_transmon.setup)
    session.connect(do_emulation=True)
    qop = TunableTransmonOperations()
    [q0] = single_tunable_transmon.qubits
    options = amplitude_rabi.options()
    options.create_experiment.count = count
    options.create_experiment.transition = transition
    options.create_experiment.use_cal_traces = use_cal_traces
    res = amplitude_rabi.run(
        session=session,
        qop=qop,
        qubits=q0,
        amplitudes=amplitudes,
        options=options,
    )
    return CompiledExperimentVerifier(res.tasks["compile_experiment"].output)


@pytest.mark.parametrize(
    "amplitudes",
    [
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3, 0.4],
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
class TestAmplitudeRabiSingleQubit:
    def test_pulse_count_drive(
        self,
        single_tunable_transmon,
        amplitudes,
        count,
        transition,
        use_cal_traces,
    ):
        verifier = create_rabi_verifier(
            single_tunable_transmon,
            amplitudes,
            count,
            transition,
            use_cal_traces,
        )

        # with cal_state on, there is 1 additional drive pulse
        expected_drive_count = count * (len(amplitudes) + int(use_cal_traces))
        verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/drive",
            expected_drive_count,
        )

        if transition == "ef":
            expected_drive_count = count * len(amplitudes)
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/drive_ef",
                expected_drive_count,
            )

    def test_pulse_count_measure_acquire(
        self,
        single_tunable_transmon,
        amplitudes,
        count,
        transition,
        use_cal_traces,
    ):
        verifier = create_rabi_verifier(
            single_tunable_transmon,
            amplitudes,
            count,
            transition,
            use_cal_traces,
        )
        # with cal_state on, there are 2 additional measure pulses
        expected_measure_count = count * (len(amplitudes) + 2 * int(use_cal_traces))
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
        single_tunable_transmon,
        amplitudes,
        count,
        transition,
        use_cal_traces,
    ):
        verifier = create_rabi_verifier(
            single_tunable_transmon,
            amplitudes,
            count,
            transition,
            use_cal_traces,
        )
        if transition == "eg":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive",
                index=0,
                start=0e-6,
                end=51e-9,
                parameterized_with=[],
            )
        elif transition == "ef":
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/drive_ef",
                index=0,
                start=56e-9,
                end=56e-9 + 52e-9,
                parameterized_with=["amplitude_q0"],
            )

    def test_pulse_measure(
        self,
        single_tunable_transmon,
        amplitudes,
        count,
        transition,
        use_cal_traces,
    ):
        verifier = create_rabi_verifier(
            single_tunable_transmon,
            amplitudes,
            count,
            transition,
            use_cal_traces,
        )

        if transition == "ge":
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
        elif transition == "ef":
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
