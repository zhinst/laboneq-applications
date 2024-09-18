"""Tests for the compiled fine_amplitude experiment using the testing utilities
provided by the LabOne Q Applications Library. in an abstract way
"""

import numpy as np
import pytest

from laboneq_applications.experiments import (
    fine_amplitude,
)

from laboneq_applications.testing import CompiledExperimentVerifier

def on_system_grid(time, system_grid=8):
    # time_ns is time in ns, system_grid is minimum stepsize in ns
    time_ns = time * 1e9

    remainder = round(time_ns % system_grid, 2)
    if remainder != 0.0:
        time_ns = time_ns + system_grid - remainder
    return round(time_ns * 1e-9, 12)


def create_fine_amplitude_verifier(
    tunable_transmon_platform,
    iterations,
    count,
    transition, 
    use_cal_traces,  
    cal_states,
):
    """Create a CompiledExperimentVerifier for the Fine Amplitude experiment."""
    qubits = tunable_transmon_platform.qpu.qubits
    if len(qubits) == 1:
        qubits = qubits[0]
    session = tunable_transmon_platform.session(do_emulation=True)
    options = fine_amplitude.options()
    options.create_experiment.count = count
    options.create_experiment.transition = transition
    options.create_experiment.use_cal_traces = use_cal_traces
    options.create_experiment.cal_states = cal_states
    options.do_analysis = False  # TODO: fix tests to work with do_analysis=True

    res = fine_amplitude.experiment_workflow(
        session=session,
        qubits=qubits,
        qpu=tunable_transmon_platform.qpu,
        iterations=iterations,
        options=options,
    ).run()

    return CompiledExperimentVerifier(res.tasks["compile_experiment"].output)


### Single-Qubit Tests ###
@pytest.mark.parametrize(
    "iterations",
    [
        np.arange(0, 2, 1),
        # np.arange(0, 4, 2),
        # np.arange(1, 4, 1),
    ],
)
@pytest.mark.parametrize(
    "count",
    [2, 5],
)
@pytest.mark.parametrize(  
    "transition, cal_states",
    [("ge","ge"), ("ef","ef")],
)
@pytest.mark.parametrize( 
    "use_cal_traces",
    [True, False],
)

class TestFineAmplitudeSingleQubit:
    """Test for fine-amplitude on a single qubit"""
    def test_pulse_count_drive(
        self,
        single_tunable_transmon_platform,
        iterations,
        count,
        transition, 
        use_cal_traces, 
        cal_states,
    ):
        """Test the number of drive pulses.

        `single_tunable_transmon` is a pytest fixture that is automatically
        imported into the test function.
        
        """
        verifier = create_fine_amplitude_verifier(
            single_tunable_transmon_platform,
            iterations,
            count,
            transition,
            use_cal_traces, 
            cal_states,
        )

    
        # only x180 is applied on the experiment, it will be varied when we do fine_amp on x90
        # this should be discussed whether putting x90 in the same module or not

        if transition == "ge":
            expected_drive_count_ge = count * np.sum(iterations)
            expected_drive_count_ef = 0
        elif transition == "ef":
            expected_drive_count_ge = count * len(iterations)
            expected_drive_count_ef = count * np.sum(iterations)

        if cal_states == "ge":
            expected_drive_count_ge += count * int(use_cal_traces)
        elif cal_states in "ef":
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

    def test_pulse_count_measure_acquire(
        self,
        single_tunable_transmon_platform,
        iterations,
        count,
        transition, 
        use_cal_traces, 
        cal_states
    ):
        """Test the number of measure and acquire pulses."""

        verifier = create_fine_amplitude_verifier(
            single_tunable_transmon_platform,
            iterations,
            count,
            transition, 
            use_cal_traces, 
            cal_states
        )

        expected_measure_count = count * (len(iterations))
        if cal_states in ("ge", "ef"):
            expected_measure_count += count * 2 * int(use_cal_traces)

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
        iterations,
        count,
        transition,  
        use_cal_traces, 
        cal_states,
    ):
        """Test the properties of drive pulses."""
        verifier = create_fine_amplitude_verifier(
            single_tunable_transmon_platform,
            iterations,
            count,
            transition, 
            use_cal_traces,  
            cal_states,
        )
        # qubits = single_tunable_transmon_platform.qpu.qubits
        # ge_drive_length = qubits[0].parameters.ge_drive_length
        ge_drive_length = 51e-9

        if iterations[0] !=0:
            if transition == "ge":
                verifier.assert_pulse(
                    signal="/logical_signal_groups/q0/drive",
                    index=0,  
                    start=0e-6,  
                    end=ge_drive_length,#ge_drive_length*iterations[0],
                    parameterized_with=[  # empty for fine_amp: pulses are constant
                    ], 
                )
            elif transition == "ef":
                verifier.assert_pulse(
                    signal="/logical_signal_groups/q0/drive_ef",
                    index=0,
                    start=on_system_grid(ge_drive_length),
                    end=on_system_grid(ge_drive_length)+ge_drive_length,
                    parameterized_with=[
                    ], 
                )

    def test_pulse_measure(
        self,
        single_tunable_transmon_platform,
        iterations,
        count,
        transition,
        use_cal_traces,
        cal_states
    ):
        """Test the properties of measure pulses.

        Here, we assert the start, end, and the parameterization of the pulses.

        """
  
        verifier = create_fine_amplitude_verifier(
            single_tunable_transmon_platform,
            iterations,
            count,
            transition, 
            use_cal_traces, 
            cal_states,
        )
        # qubits = single_tunable_transmon_platform.qpu.qubits
        # ge_drive_length = qubits[0].parameters.ge_drive_length
        # readout_length = qubits[0].parameters.readout_length
        # readout_integration_length = qubits[0].parameters.readout_integration_length
        # readout_integration_delay  = qubits[0].parameters.readout_integration_delay
        ge_drive_length = 51e-9
        readout_length = 2e-6


        if iterations[0] != 0:
            measure_start =  on_system_grid(ge_drive_length)
            measure_end = measure_start + readout_length
            integration_start = measure_start
            integration_end = measure_end

            if transition == "ef":
                measure_start += on_system_grid(ge_drive_length)
                measure_end   += on_system_grid(ge_drive_length)
                integration_start += on_system_grid(ge_drive_length)
                integration_end   += on_system_grid(ge_drive_length)
            
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/measure",
                index=0,  
                start=measure_start, 
                end=measure_end,
            )
            verifier.assert_pulse(
                signal="/logical_signal_groups/q0/acquire",
                index=0, 
                start=integration_start, 
                end=integration_end,  
            )


# @pytest.mark.parametrize(  # For new experiments: replace with relevant name and values
#     "iterations",
#     [
#         [[0,1,2,3], [0,2,4,6]],
#         [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]],
#     ],
# )
# @pytest.mark.parametrize(
#     "count",
#     [2, 4],
# )
# @pytest.mark.parametrize(  # For new experiments: keep or remove as needed
#     "transition",
#     ["ge", "ef"],
# )
# @pytest.mark.parametrize(  # For new experiments: keep or remove as needed
#     "use_cal_traces",
#     [True, False],
# )
