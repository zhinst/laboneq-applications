# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for tasks that generate rabi experiments."""

from collections.abc import Sequence

import numpy as np
import pytest
from laboneq.simple import dsl

from laboneq_applications.experiments import amplitude_rabi
from laboneq_applications.experiments.options import TuneupExperimentOptions

import tests.helpers.dsl as tsl


def reserve_ops(q):
    return [
        tsl.reserve_op(signal=f"{q.uid}/drive"),
        tsl.reserve_op(signal=f"{q.uid}/drive_ef"),
        tsl.reserve_op(signal=f"{q.uid}/measure"),
        tsl.reserve_op(signal=f"{q.uid}/acquire"),
        tsl.reserve_op(signal=f"{q.uid}/flux"),
    ]


def reference_rabi_exp(qubits, count, amplitudes, transition):
    if not isinstance(amplitudes[0], Sequence) and not isinstance(
        amplitudes[0],
        np.ndarray,
    ):
        amplitudes = [amplitudes]
    exp = tsl.experiment()
    acq = tsl.acquire_loop_rt(count=count)
    exp.children(acq)
    sweep_parameters = [
        tsl.sweep_parameter(
            uid=f"amplitude_{q.uid}", values=amplitudes[i], axis_name=f"{q.uid}"
        )
        for i, q in enumerate(qubits)
    ]
    measure_sections = []
    for q in qubits:
        measure_sections += [
            tsl.section(uid=f"measure_{q.uid}_0").children(
                reserve_ops(q),
                tsl.play_pulse_op(),
                tsl.acquire_op(),
            ),
            tsl.section(uid=f"passive_reset_{q.uid}_0").children(
                reserve_ops(q),
                tsl.delay_op(),
            ),
        ]
    if transition == "ge":
        x180_sections = [
            tsl.section(uid=f"x180_{q.uid}_0").children(
                reserve_ops(q),
                tsl.play_pulse_op(length=q.transition_parameters("ge")[1]["length"]),
            )
            for q in qubits
        ]
        acq.children(
            tsl.sweep(uid="rabi_amp_sweep_0", parameters=sweep_parameters).children(
                tsl.section(uid="main_0").children(
                    tsl.section(uid="main_drive_0").children(x180_sections),
                    tsl.section(uid="main_measure_0").children(measure_sections),
                ),
            )
        )
    elif transition == "ef":
        x180_sections = []
        for q in qubits:
            x180_sections += [
                tsl.section(uid=f"x180_{q.uid}_0").children(
                    reserve_ops(q),
                    tsl.play_pulse_op(
                        length=q.transition_parameters("ge")[1]["length"]
                    ),
                ),
                tsl.section(uid=f"x180_{q.uid}_1").children(
                    reserve_ops(q),
                    tsl.play_pulse_op(
                        length=q.transition_parameters("ef")[1]["length"]
                    ),
                ),
            ]
        acq.children(
            tsl.sweep(uid="rabi_amp_sweep_0", parameters=sweep_parameters).children(
                tsl.section(uid="main_0").children(
                    tsl.section(uid="main_drive_0").children(x180_sections),
                    tsl.section(uid="main_measure_0").children(measure_sections),
                ),
            )
        )
    return exp


def test_update_qubits(two_tunable_transmon_platform):
    qpu = two_tunable_transmon_platform.qpu

    np.testing.assert_almost_equal(
        qpu.quantum_elements[0].parameters.ge_drive_amplitude_pi, 0.8
    )
    np.testing.assert_almost_equal(
        qpu.quantum_elements[0].parameters.resonance_frequency_ge, 6.5e9
    )
    np.testing.assert_almost_equal(
        qpu.quantum_elements[1].parameters.ge_drive_amplitude_pi2, 0.41
    )
    np.testing.assert_almost_equal(
        qpu.quantum_elements[1].parameters.resonance_frequency_ef, 6.31e9
    )

    qubit_parameters = {
        "q0": {"ge_drive_amplitude_pi": 0.345, "resonance_frequency_ge": 6.61e9},
        "q1": {"ge_drive_amplitude_pi2": 0.2355, "resonance_frequency_ef": 6.01e9},
    }
    amplitude_rabi.update_qpu(qpu, qubit_parameters)

    np.testing.assert_almost_equal(
        qpu.quantum_elements[0].parameters.ge_drive_amplitude_pi, 0.345
    )
    np.testing.assert_almost_equal(
        qpu.quantum_elements[0].parameters.resonance_frequency_ge, 6.61e9
    )
    np.testing.assert_almost_equal(
        qpu.quantum_elements[1].parameters.ge_drive_amplitude_pi2, 0.2355
    )
    np.testing.assert_almost_equal(
        qpu.quantum_elements[1].parameters.resonance_frequency_ef, 6.01e9
    )


class TestWorkflow:
    def test_create_and_run(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        amplitudes = np.linspace(0, 1, 21)
        options = amplitude_rabi.experiment_workflow.options()
        options.count(10)
        options.transition("ge")

        result = amplitude_rabi.experiment_workflow(
            session=single_tunable_transmon_platform.session(do_emulation=True),
            qpu=single_tunable_transmon_platform.qpu,
            qubits=q0,
            amplitudes=amplitudes,
            options=options,
        ).run()

        assert len(result.tasks) == 6

        exp = result.tasks["create_experiment"].output
        assert exp.uid == "create_experiment"

        compiled_exp = result.tasks["compile_experiment"].output
        assert compiled_exp.experiment.uid == "create_experiment"
        assert compiled_exp.device_setup.uid == "tunable_transmons_1"

        exp_result = result.tasks["run_experiment"].output
        np.testing.assert_array_almost_equal(
            exp_result[dsl.handles.result_handle(q0.uid)].axis,
            [np.linspace(0, 1, 21)],
        )
        np.testing.assert_almost_equal(
            exp_result[dsl.handles.calibration_trace_handle(q0.uid, state="g")].data,
            4.2 + 2.1j,
        )
        np.testing.assert_almost_equal(
            exp_result[dsl.handles.calibration_trace_handle(q0.uid, state="e")].data,
            4.2 + 2.2j,
        )
        traces = exp_result[dsl.handles.calibration_trace_handle(q0.uid)]
        assert len(traces) == 2

    def test_create_and_run_no_analysis(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        amplitudes = np.linspace(0, 1, 21)
        options = amplitude_rabi.experiment_workflow.options()
        options.count(10)
        options.transition("ge")
        options.do_analysis(False)

        result = amplitude_rabi.experiment_workflow(
            session=single_tunable_transmon_platform.session(do_emulation=True),
            qpu=single_tunable_transmon_platform.qpu,
            qubits=q0,
            amplitudes=amplitudes,
            options=options,
        ).run()

        assert len(result.tasks) == 5

    def test_create_and_run_update(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        amplitudes = np.linspace(0, 1, 21)
        options = amplitude_rabi.experiment_workflow.options()
        options.count(10)
        options.transition("ge")
        options.do_analysis(True)

        result = amplitude_rabi.experiment_workflow(
            session=single_tunable_transmon_platform.session(do_emulation=True),
            qpu=single_tunable_transmon_platform.qpu,
            qubits=q0,
            amplitudes=amplitudes,
            options=options,
        ).run()

        assert len(result.tasks) == 6

    def test_create_and_run_two_qubits(self, two_tunable_transmon_platform):
        [q0, q1] = two_tunable_transmon_platform.qpu.quantum_elements
        amplitudes = [np.linspace(0, 1, 21), np.linspace(0, 0.5, 21)]
        options = amplitude_rabi.experiment_workflow.options()
        options.count(10)
        options.transition("ge")

        result = amplitude_rabi.experiment_workflow(
            session=two_tunable_transmon_platform.session(do_emulation=True),
            qpu=two_tunable_transmon_platform.qpu,
            qubits=[q0, q1],
            amplitudes=amplitudes,
            options=options,
        ).run()

        assert len(result.tasks) == 6

        exp = result.tasks["create_experiment"].output
        assert exp.uid == "create_experiment"

        compiled_exp = result.tasks["compile_experiment"].output
        assert compiled_exp.experiment.uid == "create_experiment"
        assert compiled_exp.device_setup.uid == "tunable_transmons_2"

        exp_result = result.tasks["run_experiment"].output
        np.testing.assert_array_almost_equal(
            exp_result[dsl.handles.result_handle(q0.uid)].axis[0][0],
            np.linspace(0, 1, 21),
        )
        np.testing.assert_array_almost_equal(
            exp_result[dsl.handles.result_handle(q1.uid)].axis[0][1],
            np.linspace(0, 0.5, 21),
        )
        np.testing.assert_almost_equal(
            exp_result[dsl.handles.calibration_trace_handle(q0.uid, state="g")].data,
            4.2 + 2.1j,
        )
        np.testing.assert_almost_equal(
            exp_result[dsl.handles.calibration_trace_handle(q0.uid, state="e")].data,
            4.2 + 2.2j,
        )

        np.testing.assert_almost_equal(
            exp_result[dsl.handles.calibration_trace_handle(q1.uid, state="g")].data,
            4.3 + 2.1j,
        )
        np.testing.assert_almost_equal(
            exp_result[dsl.handles.calibration_trace_handle(q1.uid, state="e")].data,
            4.3 + 2.2j,
        )

        traces = exp_result[dsl.handles.calibration_trace_handle(q0.uid)]
        assert len(traces) == 2

        traces = exp_result[dsl.handles.calibration_trace_handle(q0.uid)]
        assert len(traces) == 2


@pytest.mark.parametrize("transition", ["ge", "ef"])
@pytest.mark.parametrize("count", [10, 12])
class TestAmplitudeRabiSingleQubit:
    @pytest.fixture
    def platform(self, single_tunable_transmon_platform):
        return single_tunable_transmon_platform

    @pytest.fixture
    def amplitude(self):
        return np.linspace(0, 1, 21)

    @pytest.fixture
    def options(self, transition, count):
        options = TuneupExperimentOptions(
            count=count, transition=transition, cal_states=transition
        )
        options.use_cal_traces = False

        return options

    @pytest.fixture
    def qpu(self, platform):
        return platform.qpu

    @pytest.fixture
    def q0(self, qpu):
        return qpu.quantum_elements[0]

    def test_create_exp_single_qubit(
        self,
        platform,
        amplitude,
        options,
        qpu,
        q0,
    ):
        exp = amplitude_rabi.create_experiment(
            qpu,
            q0,
            amplitude,
            options=options,
        )
        assert exp == reference_rabi_exp(
            [q0],
            options.count,
            amplitude,
            options.transition,
        )
        session = platform.session(do_emulation=True)
        session.compile(exp)

    def test_invalid_input_raises_error(self, options, qpu, q0):
        with pytest.raises(ValueError):
            amplitude_rabi.create_experiment(
                qpu,
                q0,
                [[0.1, 0.5], [0.1, 0.5]],
                options=options,
            )

        with pytest.raises(ValueError):
            amplitude_rabi.create_experiment(
                qpu,
                [q0],
                [0.1, 0.5],
                options=options,
            )
        with pytest.raises(ValueError):
            amplitude_rabi.create_experiment(
                qpu,
                q0,
                [0.1, None, 0.5],
                options=options,
            )

    def test_amplitude_is_nparray(self, amplitude, options, qpu, q0):
        exp = amplitude_rabi.create_experiment(
            qpu,
            q0,
            np.array(amplitude),
            options=options,
        )
        assert exp == reference_rabi_exp(
            [q0],
            options.count,
            np.array(amplitude),
            options.transition,
        )


@pytest.mark.parametrize("transition", ["ge", "ef"])
@pytest.mark.parametrize("count", [10, 12])
class TestAmplitudeRabiTwoQubit:
    @pytest.fixture
    def platform(self, two_tunable_transmon_platform):
        return two_tunable_transmon_platform

    @pytest.fixture
    def options(self, transition, count):
        options = TuneupExperimentOptions(
            count=count, transition=transition, cal_states=transition
        )
        options.use_cal_traces = False

        return options

    @pytest.fixture
    def qpu(self, platform):
        return platform.qpu

    @pytest.fixture
    def amplitudes(self):
        return [np.linspace(0, 1, 21), np.linspace(0, 0.5, 21)]

    @pytest.fixture
    def qubits(self, qpu):
        return qpu.quantum_elements

    def test_run_standalone(self, platform, amplitudes, options, qpu, qubits):
        exp = amplitude_rabi.create_experiment(
            qpu,
            qubits,
            amplitudes,
            options=options,
        )
        assert exp == reference_rabi_exp(
            qubits,
            options.count,
            amplitudes,
            options.transition,
        )
        session = platform.session(do_emulation=True)
        session.compile(exp)

    def test_invalid_input_raises_error(self, qpu, qubits, options):
        with pytest.raises(ValueError):
            amplitude_rabi.create_experiment(
                qpu,
                qubits,
                [0.1, 0.5],
                options=options,
            )

        with pytest.raises(ValueError):
            amplitude_rabi.create_experiment(
                qpu,
                qubits,
                [[0.1, 0.5]],
                options=options,
            )
        with pytest.raises(ValueError):
            amplitude_rabi.create_experiment(
                qpu,
                qubits,
                [[0.1, 0.5], [0.1, None]],
                options=options,
            )

    def test_amplitude_is_nparray(self, qpu, qubits, options):
        exp = amplitude_rabi.create_experiment(
            qpu,
            qubits,
            [np.array([0, 1, 2]), np.array([0, 1, 2])],
            options=options,
        )
        assert exp == reference_rabi_exp(
            qubits,
            options.count,
            [np.array([0, 1, 2]), np.array([0, 1, 2])],
            options.transition,
        )
