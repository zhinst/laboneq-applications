"""Tests for the compiled resonator spectroscopy experiment using the testing utilities
provided by the LabOne Q Applications Library.
"""

import pytest

from laboneq_applications.experiments import (
    resonator_spectroscopy,
)
from laboneq_applications.testing import CompiledExperimentVerifier


def create_res_spectroscopy_verifier(
    tunable_transmon_platform,
    frequencies,
    count,
    use_cw,
    spectroscopy_reset_delay,
):
    """Create a CompiledExperimentVerifier for the resonator spectroscopy experiment."""
    qubits = tunable_transmon_platform.qpu.qubits
    # for this specific experiment, I force just one qubit by default
    qubit = qubits[0]
    session = tunable_transmon_platform.session(do_emulation=True)
    options = resonator_spectroscopy.options()
    options.create_experiment.count = count
    options.create_experiment.use_cw = use_cw
    options.create_experiment.spectroscopy_reset_delay = spectroscopy_reset_delay
    res = resonator_spectroscopy.experiment_workflow(
        session=session,
        qubit=qubit,
        qpu=tunable_transmon_platform.qpu,
        frequencies=frequencies,
        options=options,
    ).run()
    return CompiledExperimentVerifier(res.tasks["compile_experiment"].output)


### Single-Qubit Tests ###


# use pytest.mark.parametrize to generate test cases for
# all combinations of the parameters.
@pytest.mark.parametrize(
    "frequencies",
    [
        [1.7e9, 2e9, 2.3e9],
        [1e9, 2e9, 3e9],
    ],
)
@pytest.mark.parametrize(
    "count",
    [2, 4],
)
@pytest.mark.parametrize(
    "use_cw",
    [True, False],
)
@pytest.mark.parametrize(
    "spectroscopy_reset_delay",
    [1e-6, 100e-6],
)
class TestResonatorSpectroscopySingleQubit:
    def test_pulse_count_measure_acquire(
        self,
        single_tunable_transmon_platform,
        frequencies,
        count,
        use_cw,
        spectroscopy_reset_delay,
    ):
        """Test the number of measure and acquire pulses."""
        verifier = create_res_spectroscopy_verifier(
            single_tunable_transmon_platform,
            frequencies,
            count,
            use_cw,
            spectroscopy_reset_delay,
        )
        expected_measure_count = count * len(frequencies)
        expected_acquire_count = expected_measure_count
        if use_cw:
            expected_measure_count = 0

        if True:
            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/measure",
                expected_measure_count,
            )

            verifier.assert_number_of_pulses(
                "/logical_signal_groups/q0/acquire",
                expected_acquire_count,
            )

    def test_pulse_measure(
        self,
        single_tunable_transmon_platform,
        frequencies,
        count,
        use_cw,
        spectroscopy_reset_delay,
    ):
        """Test the properties of measure pulses.

        Here, we assert the start, end, and the parameterization of the pulses.

        """
        verifier = create_res_spectroscopy_verifier(
            single_tunable_transmon_platform,
            frequencies,
            count,
            use_cw,
            spectroscopy_reset_delay,
        )
        if not use_cw:
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,
                start=88e-9,
                end=2.088e-6,
            )
        verifier.assert_pulse(
            signal="/logical_signal_groups/q0/acquire",
            index=0,
            start=88e-9,
            end=2.088e-6,
        )
