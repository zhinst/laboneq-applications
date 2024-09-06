"""Tests for tasks that generate rabi experiments."""

from collections.abc import Sequence

import numpy as np
import pytest

from laboneq_applications.experiments import amplitude_rabi
from laboneq_applications.experiments.options import TuneupExperimentOptions

import tests.helpers.dsl as tsl


def reserve_ops(q):
    return [
        tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/drive"),
        tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/drive_ef"),
        tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/measure"),
        tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/acquire"),
        tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/flux"),
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
    for i, q in enumerate(qubits):
        sweep_parameter = tsl.sweep_parameter(
            uid=f"amplitude_{q.uid}",
            values=amplitudes[i],
        )
        x180_ge_length = q.transition_parameters("ge")[1]["length"]
        x180_ef_length = q.transition_parameters("ef")[1]["length"]
        if transition == "ge":
            acq.children(
                tsl.sweep(uid=f"amps_{q.uid}_0", parameters=[sweep_parameter]).children(
                    tsl.section(uid=f"prepare_state_{q.uid}_0").children(
                        reserve_ops(q),
                    ),
                    tsl.section(uid=f"x180_{q.uid}_0").children(
                        reserve_ops(q),
                        tsl.play_pulse_op(length=x180_ge_length),
                    ),
                    tsl.section(uid=f"measure_{q.uid}_0").children(
                        reserve_ops(q),
                        tsl.play_pulse_op(),
                        tsl.acquire_op(),
                    ),
                    tsl.section(uid=f"passive_reset_{q.uid}_0").children(
                        reserve_ops(q),
                        tsl.delay_op(),
                    ),
                ),
            )
            acq.children(
                tsl.section(uid=f"cal_{q.uid}_0").children(
                    tsl.section(uid=f"prepare_state_{q.uid}_1").children(
                        reserve_ops(q),
                    ),
                    tsl.section(uid=f"measure_{q.uid}_1").children(
                        reserve_ops(q),
                        tsl.play_pulse_op(),
                        tsl.acquire_op(),
                    ),
                    tsl.section(uid=f"passive_reset_{q.uid}_1").children(
                        reserve_ops(q),
                        tsl.delay_op(),
                    ),
                    tsl.section(uid=f"prepare_state_{q.uid}_2").children(
                        reserve_ops(q),
                        tsl.section(uid=f"x180_{q.uid}_1").children(
                            reserve_ops(q),
                            tsl.play_pulse_op(),
                        ),
                    ),
                    tsl.section(uid=f"measure_{q.uid}_2").children(
                        reserve_ops(q),
                        tsl.play_pulse_op(),
                        tsl.acquire_op(),
                    ),
                    tsl.section(uid=f"passive_reset_{q.uid}_2").children(
                        reserve_ops(q),
                        tsl.delay_op(),
                    ),
                ),
            )
        elif transition == "ef":
            acq.children(
                tsl.sweep(uid=f"amps_{q.uid}_0", parameters=[sweep_parameter]).children(
                    tsl.section(uid=f"prepare_state_{q.uid}_0").children(
                        reserve_ops(q),
                        tsl.section(uid=f"x180_{q.uid}_0").children(
                            reserve_ops(q),
                            tsl.play_pulse_op(length=x180_ge_length),
                        ),
                    ),
                    tsl.section(uid=f"x180_{q.uid}_1").children(
                        reserve_ops(q),
                        tsl.play_pulse_op(length=x180_ef_length),
                    ),
                    tsl.section(uid=f"measure_{q.uid}_0").children(
                        reserve_ops(q),
                        tsl.play_pulse_op(),
                        tsl.acquire_op(),
                    ),
                    tsl.section(uid=f"passive_reset_{q.uid}_0").children(
                        reserve_ops(q),
                        tsl.delay_op(),
                    ),
                ),
            )
            acq.children(
                tsl.section(uid=f"cal_{q.uid}_0").children(
                    tsl.section(uid=f"prepare_state_{q.uid}_1").children(
                        reserve_ops(q),
                        tsl.section(uid=f"x180_{q.uid}_2").children(
                            reserve_ops(q),
                            tsl.play_pulse_op(length=x180_ge_length),
                        ),
                    ),
                    tsl.section(uid=f"measure_{q.uid}_1").children(
                        reserve_ops(q),
                        tsl.play_pulse_op(),
                        tsl.acquire_op(),
                    ),
                    tsl.section(uid=f"passive_reset_{q.uid}_1").children(
                        reserve_ops(q),
                        tsl.delay_op(),
                    ),
                    tsl.section(uid=f"prepare_state_{q.uid}_2").children(
                        reserve_ops(q),
                        tsl.section(uid=f"x180_{q.uid}_3").children(
                            reserve_ops(q),
                            tsl.play_pulse_op(),
                        ),
                        tsl.section(uid=f"x180_{q.uid}_4").children(
                            reserve_ops(q),
                            tsl.play_pulse_op(),
                        ),
                    ),
                    tsl.section(uid=f"measure_{q.uid}_2").children(
                        reserve_ops(q),
                        tsl.play_pulse_op(),
                        tsl.acquire_op(),
                    ),
                    tsl.section(uid=f"passive_reset_{q.uid}_2").children(
                        reserve_ops(q),
                        tsl.delay_op(),
                    ),
                ),
            )

    return exp


def test_update_qubits(two_tunable_transmon_platform):
    qpu = two_tunable_transmon_platform.qpu

    np.testing.assert_almost_equal(qpu.qubits[0].parameters.ge_drive_amplitude_pi, 0.8)
    np.testing.assert_almost_equal(
        qpu.qubits[0].parameters.resonance_frequency_ge, 1.6e9
    )
    np.testing.assert_almost_equal(
        qpu.qubits[1].parameters.ge_drive_amplitude_pi2, 0.41
    )
    np.testing.assert_almost_equal(
        qpu.qubits[1].parameters.resonance_frequency_ef, 1.71e9
    )

    qubit_parameters = {
        "q0": {"ge_drive_amplitude_pi": 0.345, "resonance_frequency_ge": 6.61e9},
        "q1": {"ge_drive_amplitude_pi2": 0.2355, "resonance_frequency_ef": 6.01e9},
    }
    amplitude_rabi.update_qubits(qpu, qubit_parameters)

    np.testing.assert_almost_equal(
        qpu.qubits[0].parameters.ge_drive_amplitude_pi, 0.345
    )
    np.testing.assert_almost_equal(
        qpu.qubits[0].parameters.resonance_frequency_ge, 6.61e9
    )
    np.testing.assert_almost_equal(
        qpu.qubits[1].parameters.ge_drive_amplitude_pi2, 0.2355
    )
    np.testing.assert_almost_equal(
        qpu.qubits[1].parameters.resonance_frequency_ef, 6.01e9
    )


class TestWorkflow:
    def test_create_and_run(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        amplitudes = np.linspace(0, 1, 21)
        options = amplitude_rabi.options()
        options.create_experiment.count = 10
        options.create_experiment.transition = "ge"

        result = amplitude_rabi.experiment_workflow(
            session=single_tunable_transmon_platform.session(do_emulation=True),
            qpu=single_tunable_transmon_platform.qpu,
            qubits=q0,
            amplitudes=amplitudes,
            options=options,
        ).run()

        assert len(result.tasks) == 4

        exp = result.tasks["create_experiment"].output
        assert exp.uid == "create_experiment"

        compiled_exp = result.tasks["compile_experiment"].output
        assert compiled_exp.experiment.uid == "create_experiment"
        assert compiled_exp.device_setup.uid == "tunable_transmons_1"

        exp_result = result.tasks["run_experiment"].output
        np.testing.assert_array_almost_equal(
            exp_result.result.q0.axis,
            [np.linspace(0, 1, 21)],
        )
        np.testing.assert_almost_equal(exp_result.cal_trace.q0.g.data, 4.2 + 2.1j)
        np.testing.assert_almost_equal(exp_result.cal_trace.q0.e.data, 4.2 + 2.2j)
        traces = exp_result.cal_trace.q0
        assert len(traces) == 2

    def test_create_and_run_no_analysis(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        amplitudes = np.linspace(0, 1, 21)
        options = amplitude_rabi.options()
        options.create_experiment.count = 10
        options.create_experiment.transition = "ge"
        options.do_analysis = False

        result = amplitude_rabi.experiment_workflow(
            session=single_tunable_transmon_platform.session(do_emulation=True),
            qpu=single_tunable_transmon_platform.qpu,
            qubits=q0,
            amplitudes=amplitudes,
            options=options,
        ).run()

        assert len(result.tasks) == 3

    def test_create_and_run_update(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        amplitudes = np.linspace(0, 1, 21)
        options = amplitude_rabi.options()
        options.create_experiment.count = 10
        options.create_experiment.transition = "ge"
        options.update = True

        result = amplitude_rabi.experiment_workflow(
            session=single_tunable_transmon_platform.session(do_emulation=True),
            qpu=single_tunable_transmon_platform.qpu,
            qubits=q0,
            amplitudes=amplitudes,
            options=options,
        ).run()

        assert len(result.tasks) == 5

    def test_create_and_run_two_qubits(self, two_tunable_transmon_platform):
        [q0, q1] = two_tunable_transmon_platform.qpu.qubits
        amplitudes = [np.linspace(0, 1, 21), np.linspace(0, 0.5, 21)]
        options = amplitude_rabi.options()
        options.create_experiment.count = 10
        options.create_experiment.transition = "ge"

        result = amplitude_rabi.experiment_workflow(
            session=two_tunable_transmon_platform.session(do_emulation=True),
            qpu=two_tunable_transmon_platform.qpu,
            qubits=[q0, q1],
            amplitudes=amplitudes,
            options=options,
        ).run()

        assert len(result.tasks) == 4

        exp = result.tasks["create_experiment"].output
        assert exp.uid == "create_experiment"

        compiled_exp = result.tasks["compile_experiment"].output
        assert compiled_exp.experiment.uid == "create_experiment"
        assert compiled_exp.device_setup.uid == "tunable_transmons_2"

        exp_result = result.tasks["run_experiment"].output
        np.testing.assert_array_almost_equal(
            exp_result.result.q0.axis,
            [np.linspace(0, 1, 21)],
        )
        np.testing.assert_array_almost_equal(
            exp_result.result.q1.axis,
            [np.linspace(0, 0.5, 21)],
        )
        np.testing.assert_almost_equal(exp_result.cal_trace.q0.g.data, 4.2 + 2.1j)
        np.testing.assert_almost_equal(exp_result.cal_trace.q0.e.data, 4.2 + 2.2j)

        np.testing.assert_almost_equal(exp_result.cal_trace.q1.g.data, 4.3 + 2.1j)
        np.testing.assert_almost_equal(exp_result.cal_trace.q1.e.data, 4.3 + 2.2j)

        traces = exp_result.cal_trace.q0
        assert len(traces) == 2

        traces = exp_result.cal_trace.q1
        assert len(traces) == 2


@pytest.mark.parametrize("transition", ["ge", "ef"])
@pytest.mark.parametrize("count", [10, 12])
class TestAmplitudeRabiSingleQubit:
    @pytest.fixture(autouse=True)
    def _setup(self, single_tunable_transmon_platform, transition, count):
        self.platform = single_tunable_transmon_platform
        self.qpu = self.platform.qpu
        [self.q0] = self.qpu.qubits
        self.amplitude = np.linspace(0, 1, 21)
        self.options = TuneupExperimentOptions(count=count, transition=transition)

    def test_create_exp_single_qubit(self):
        exp = amplitude_rabi.create_experiment(
            self.qpu,
            self.q0,
            self.amplitude,
            options=self.options,
        )
        assert exp == reference_rabi_exp(
            [self.q0],
            self.options.count,
            self.amplitude,
            self.options.transition,
        )
        session = self.platform.session(do_emulation=True)
        session.compile(exp)

    def test_invalid_input_raises_error(self):
        with pytest.raises(ValueError):
            amplitude_rabi.create_experiment(
                self.qpu,
                self.q0,
                [[0.1, 0.5], [0.1, 0.5]],
                options=self.options,
            )

        with pytest.raises(ValueError):
            amplitude_rabi.create_experiment(
                self.qpu,
                [self.q0],
                [0.1, 0.5],
                options=self.options,
            )
        with pytest.raises(ValueError):
            amplitude_rabi.create_experiment(
                self.qpu,
                self.q0,
                [0.1, None, 0.5],
                options=self.options,
            )

    def test_amplitude_is_nparray(self):
        exp = amplitude_rabi.create_experiment(
            self.qpu,
            self.q0,
            np.array(self.amplitude),
            options=self.options,
        )
        assert exp == reference_rabi_exp(
            [self.q0],
            self.options.count,
            np.array(self.amplitude),
            self.options.transition,
        )


@pytest.mark.parametrize("transition", ["ge", "ef"])
@pytest.mark.parametrize("count", [10, 12])
class TestAmplitudeRabiTwoQubit:
    @pytest.fixture(autouse=True)
    def _setup(self, two_tunable_transmon_platform, transition, count):
        self.platform = two_tunable_transmon_platform
        self.qpu = self.platform.qpu
        self.q0, self.q1 = self.qpu.qubits
        self.amplitudes = [np.linspace(0, 1, 21), np.linspace(0, 0.5, 21)]
        self.options = TuneupExperimentOptions(count=count, transition=transition)

    def test_run_standalone(self):
        exp = amplitude_rabi.create_experiment(
            self.qpu,
            [self.q0, self.q1],
            self.amplitudes,
            options=self.options,
        )
        assert exp == reference_rabi_exp(
            [self.q0, self.q1],
            self.options.count,
            self.amplitudes,
            self.options.transition,
        )
        session = self.platform.session(do_emulation=True)
        session.compile(exp)

    def test_invalid_input_raises_error(self):
        with pytest.raises(ValueError):
            amplitude_rabi.create_experiment(
                self.qpu,
                [self.q0, self.q1],
                [0.1, 0.5],
                options=self.options,
            )

        with pytest.raises(ValueError):
            amplitude_rabi.create_experiment(
                self.qpu,
                [self.q0, self.q1],
                [[0.1, 0.5]],
                options=self.options,
            )
        with pytest.raises(ValueError):
            amplitude_rabi.create_experiment(
                self.qpu,
                [self.q0, self.q1],
                [[0.1, 0.5], [0.1, None]],
                options=self.options,
            )

    def test_amplitude_is_nparray(self):
        exp = amplitude_rabi.create_experiment(
            self.qpu,
            [self.q0, self.q1],
            [np.array([0, 1, 2]), np.array([0, 1, 2])],
            options=self.options,
        )
        assert exp == reference_rabi_exp(
            [self.q0, self.q1],
            self.options.count,
            [np.array([0, 1, 2]), np.array([0, 1, 2])],
            self.options.transition,
        )
