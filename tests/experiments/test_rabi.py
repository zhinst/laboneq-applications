"""Tests for tasks that generate rabi experiments."""

from collections.abc import Sequence

import numpy as np
import pytest

import tests.helpers.dsl as tsl
from laboneq_applications.core.options import TuneupExperimentOptions
from laboneq_applications.experiments import amplitude_rabi


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


class TestTaskbook:
    def test_create_and_run(self, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        amplitudes = [0.1, 0.2]
        options = amplitude_rabi.options()
        options.create_experiment.count = 10
        options.create_experiment.transition = "ge"

        result = amplitude_rabi.run(
            session=single_tunable_transmon.session(do_emulation=True),
            qpu=single_tunable_transmon,
            qubits=q0,
            amplitudes=amplitudes,
            options=options,
        )

        assert len(result.tasks) == 3

        exp = result.tasks[0].output
        assert exp.uid == "create_experiment"

        compiled_exp = result.tasks[1].output
        assert compiled_exp.experiment.uid == "create_experiment"
        assert compiled_exp.device_setup.uid == "tunable_transmons_1"

        exp_result = result.tasks[2].output
        np.testing.assert_array_almost_equal(
            exp_result.result.q0.axis,
            [[0.1, 0.2]],
        )
        np.testing.assert_almost_equal(exp_result.cal_trace.q0.g.data, 4.2 + 0.2j)
        np.testing.assert_almost_equal(exp_result.cal_trace.q0.e.data, 4.2 + 0.3j)
        traces = exp_result.cal_trace.q0
        assert len(traces) == 2


@pytest.mark.parametrize("transition", ["ge", "ef"])
@pytest.mark.parametrize("count", [10, 12])
class TestAmplitudeRabiSingleQubit:
    @pytest.fixture(autouse=True)
    def _setup(self, single_tunable_transmon, transition, count):
        self.qpu = single_tunable_transmon
        [self.q0] = self.qpu.qubits
        self.amplitude = [0.1, 0.5, 1]
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
        session = self.qpu.session(do_emulation=True)
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
    def _setup(self, two_tunable_transmon, transition, count):
        self.qpu = two_tunable_transmon
        self.q0, self.q1 = self.qpu.qubits
        self.amplitudes = [[0.1, 0.5, 1], [0.1, 0.5, 1]]
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
        session = self.qpu.session(do_emulation=True)
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
