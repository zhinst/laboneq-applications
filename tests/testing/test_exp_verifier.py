# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ExperimentVerifier class.

Here, to test the ExperimentVerifier class, we need a sample L1Q experiment that
has been compiled.

This can be obtained from a saved workflow result.

However, this way of testing is very fragile, as L1Q deserialization is not guaranteed
to be stable across different versions.

Instead, we choose the lesser devil between two by coupling these tests to
parts belonging to the library itself (local) rather than the
L1Q (global).

To this end, we create a compiled experiment using a well-tested and presumably
stable workflow amplitude_rabi.
"""

import numpy as np
import pytest
from laboneq.dsl.session import Session

from laboneq_applications.experiments import amplitude_rabi
from laboneq_applications.testing.experiment_verifier import (
    CompiledExperimentVerifier,
    _Pulse,
    _PulseExtractorPSV,
)


@pytest.fixture()
def rabi_compiled(single_tunable_transmon_platform):
    session = Session(single_tunable_transmon_platform.setup)
    session.connect(do_emulation=True)
    [q0] = single_tunable_transmon_platform.qpu.qubits
    amplitudes = [0.1, 0.2, 0.3]
    options = amplitude_rabi.experiment_workflow.options()
    options.count(2)
    options.do_analysis(value=False)  # TODO: fix tests to work with do_analysis=True
    res = amplitude_rabi.experiment_workflow(
        session=session,
        qubits=q0,
        qpu=single_tunable_transmon_platform.qpu,
        amplitudes=amplitudes,
        options=options,
    ).run()
    return res.tasks["compile_experiment"].output


@pytest.fixture()
def rabi_pulse_extractor(rabi_compiled):
    return _PulseExtractorPSV(rabi_compiled, max_events=5000)


@pytest.fixture()
def rabi_exp_verifier(rabi_compiled) -> CompiledExperimentVerifier:
    return CompiledExperimentVerifier(rabi_compiled)


class TestPulseExtractor:
    def test_create_p_extract(self, rabi_compiled):
        p_extract = _PulseExtractorPSV(rabi_compiled, max_events=5432)
        assert p_extract is not None
        assert p_extract.max_events == 5432

    def test_get_number_of_pulses(self, rabi_pulse_extractor):
        assert (
            rabi_pulse_extractor.get_pulse_count("/logical_signal_groups/q0/drive") == 8
        )
        assert (
            rabi_pulse_extractor.get_pulse_count("/logical_signal_groups/q0/acquire")
            == 10
        )
        assert (
            rabi_pulse_extractor.get_pulse_count("/logical_signal_groups/q0/measure")
            == 10
        )

    def test_get_pulse(self, rabi_pulse_extractor):
        pulse = rabi_pulse_extractor.get_pulse("/logical_signal_groups/q0/drive", 1)
        truth = _Pulse(
            start=3.061e-6,
            end=3.112e-6,
            parameterized_with=["amplitude_q0"],
        )
        np.testing.assert_allclose(pulse.start, truth.start, atol=1e-12)
        np.testing.assert_allclose(pulse.end, truth.end, atol=1e-12)
        assert pulse.parameterized_with == truth.parameterized_with

        # amplitude_rabi has cal state measurement at the end of every averaging
        pulse = rabi_pulse_extractor.get_pulse("/logical_signal_groups/q0/drive", 3)
        # This is not ideal to hardcode values for pulse timing here
        # but too much effort to calculate them to test only the pulse extractor
        truth = _Pulse(start=12.173e-6, end=12.224e-6, parameterized_with=[])
        np.testing.assert_allclose(pulse.start, truth.start, atol=1e-12)
        np.testing.assert_allclose(pulse.end, truth.end, atol=1e-12)
        assert pulse.parameterized_with == truth.parameterized_with

        # amplitude_rabi has cal state measurement at the end of every averaging
        pulse = rabi_pulse_extractor.get_pulse("/logical_signal_groups/q0/measure", 0)
        truth = _Pulse(start=56e-9, end=56e-9 + 2e-6, parameterized_with=[])
        np.testing.assert_allclose(pulse.start, truth.start, atol=1e-12)
        np.testing.assert_allclose(pulse.end, truth.end, atol=1e-12)
        assert pulse.parameterized_with == truth.parameterized_with

    def test_raise_error_for_invalid_pulse_number(self, rabi_pulse_extractor):
        with pytest.raises(ValueError, match="Pulse number out of range"):
            rabi_pulse_extractor.get_pulse(
                "/logical_signal_groups/q0/drive",
                1000,
            )


class TestExperimentVerifier:
    def test_init(self, rabi_compiled):
        verifier = CompiledExperimentVerifier(rabi_compiled)
        assert isinstance(verifier.pulse_extractor, _PulseExtractorPSV)

    def test_assert_number_of_pulses(self, rabi_exp_verifier):
        rabi_exp_verifier.assert_number_of_pulses("/logical_signal_groups/q0/drive", 8)
        rabi_exp_verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/acquire",
            10,
        )
        rabi_exp_verifier.assert_number_of_pulses(
            "/logical_signal_groups/q0/measure",
            10,
        )

    def test_assert_wrong_number_of_pulse(self, rabi_exp_verifier):
        signal = "/logical_signal_groups/q0/drive"
        pulse_number = 10000
        actual_pulse_number = rabi_exp_verifier.pulse_extractor.get_pulse_count(signal)
        expected_err = (
            f"Number of pulses mismatch for signal {signal} "
            f"expected {actual_pulse_number} got {pulse_number}"
        )

        with pytest.raises(
            AssertionError,
            match=expected_err,
        ):
            rabi_exp_verifier.assert_number_of_pulses(signal, pulse_number)

    def test_assert_pulse(self, rabi_exp_verifier):
        rabi_exp_verifier.assert_pulse(
            "/logical_signal_groups/q0/drive",
            3,
            12.173e-6,
            12.224e-6,
            [],
        )
        rabi_exp_verifier.assert_pulse(
            "/logical_signal_groups/q0/drive",
            3,
            12.173e-6,
        )

    def test_assert_wrong_pulse(self, rabi_exp_verifier):
        signal = "/logical_signal_groups/q0/drive"
        pulse_number = 3
        start = 1234
        end = 5678
        parameterized_with = ["amplitude_q0"]
        with pytest.raises(
            AssertionError,
        ):
            rabi_exp_verifier.assert_pulse(
                signal,
                pulse_number,
                start,
                end,
                parameterized_with,
            )

    def test_assert_pulse_pair(self, rabi_exp_verifier):
        signal = "/logical_signal_groups/q0/drive"
        rabi_exp_verifier.assert_pulse_pair(
            signal,
            [0, 1],
            start=3.056e-6,
            end=3.056e-6,
        )

        signal_drive = "/logical_signal_groups/q0/drive"
        signal_measure = "/logical_signal_groups/q0/measure"
        rabi_exp_verifier.assert_pulse_pair(
            (signal_drive, signal_measure),
            (0, 0),
            start=51e-9,
        )

        rabi_exp_verifier.assert_pulse_pair(
            (signal_drive, signal_measure),
            (0, 0),
            distance=0.0,
        )

        with pytest.raises(ValueError, match="Indices must be a tuple of two integers"):
            rabi_exp_verifier.assert_pulse_pair(
                (signal_measure, signal_drive),
                (1, 2, 3),
                start=56e-9,
            )
