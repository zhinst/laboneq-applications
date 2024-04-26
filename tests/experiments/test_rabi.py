"""Tests for tasks that generate rabi experiments."""

from collections.abc import Sequence

import pytest

import tests.helpers.dsl as tsl
from laboneq_applications.experiments import amplitude_rabi
from laboneq_applications.qpu_types.tunable_transmon import TunableTransmonOperations
from laboneq_applications.workflow.workflow import Workflow


def reserve_ops(q):
    return [
        tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/drive"),
        tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/drive_ef"),
        tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/measure"),
        tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/acquire"),
        tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/flux"),
    ]


def reference_rabi_exp(qubits, count, amplitudes, transition):
    if not isinstance(amplitudes[0], Sequence):
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
                    tsl.section(uid=f"prep_{q.uid}_0").children(
                        reserve_ops(q),
                        tsl.section(uid=f"reset_{q.uid}_0").children(
                            reserve_ops(q),
                            tsl.delay_op(),
                        ),
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
                ),
            )
            acq.children(
                tsl.section(uid=f"cal_{q.uid}_0").children(
                    tsl.section(uid=f"prep_{q.uid}_1").children(
                        reserve_ops(q),
                        tsl.section(uid=f"reset_{q.uid}_1").children(
                            reserve_ops(q),
                            tsl.delay_op(),
                        ),
                    ),
                    tsl.section(uid=f"measure_{q.uid}_1").children(
                        reserve_ops(q),
                        tsl.play_pulse_op(),
                        tsl.acquire_op(),
                    ),
                    tsl.section(uid=f"prep_{q.uid}_2").children(
                        reserve_ops(q),
                        tsl.section(uid=f"reset_{q.uid}_2").children(
                            reserve_ops(q),
                            tsl.delay_op(),
                        ),
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
                ),
            )
        elif transition == "ef":
            acq.children(
                tsl.sweep(uid=f"amps_{q.uid}_0", parameters=[sweep_parameter]).children(
                    tsl.section(uid=f"prep_{q.uid}_0").children(
                        reserve_ops(q),
                        tsl.section(uid=f"reset_{q.uid}_0").children(
                            reserve_ops(q),
                            tsl.delay_op(),
                        ),
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
                ),
            )
            acq.children(
                tsl.section(uid=f"cal_{q.uid}_0").children(
                    tsl.section(uid=f"prep_{q.uid}_1").children(
                        reserve_ops(q),
                        tsl.section(uid=f"reset_{q.uid}_1").children(
                            reserve_ops(q),
                            tsl.delay_op(),
                        ),
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
                    tsl.section(uid=f"prep_{q.uid}_2").children(
                        reserve_ops(q),
                        tsl.section(uid=f"reset_{q.uid}_2").children(
                            reserve_ops(q),
                            tsl.delay_op(),
                        ),
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
                ),
            )

    return exp


@pytest.mark.parametrize("transition", ["ge", "ef"])
@pytest.mark.parametrize("count", [10, 12])
class TestAmplitudeRabiSingleQubit:
    @pytest.fixture(autouse=True)
    def _setup(self, single_tunable_transmon, transition, count):
        self.q0 = single_tunable_transmon.qubits[0]
        self.amplitude = [0.1, 0.5, 1]
        self.count = count
        self.transition = transition
        self.qop = TunableTransmonOperations()

    def test_run_standalone_single_qubit_passed(self):
        exp = amplitude_rabi(
            self.qop,
            self.q0,
            self.amplitude,
            self.count,
            self.transition,
        )
        assert exp == reference_rabi_exp(
            [self.q0],
            self.count,
            self.amplitude,
            self.transition,
        )

    def test_run_task(self):
        exp = amplitude_rabi(
            self.qop,
            self.q0,
            self.amplitude,
            self.count,
            self.transition,
        )
        with Workflow() as wf:
            amplitude_rabi(
                self.qop,
                self.q0,
                self.amplitude,
                self.count,
                self.transition,
            )
        assert wf.run().tasklog == {"amplitude_rabi": [exp]}

    def test_invalid_input_raises_error(self):
        with pytest.raises(ValueError):
            amplitude_rabi(
                self.qop,
                self.q0,
                [[0.1, 0.5], [0.1, 0.5]],
                self.count,
                self.transition,
            )

        with pytest.raises(ValueError):
            amplitude_rabi(
                self.qop,
                [self.q0],
                [0.1, 0.5],
                self.count,
                self.transition,
            )
        with pytest.raises(ValueError):
            amplitude_rabi(
                self.qop,
                self.q0,
                [0.1, None, 0.5],
                self.count,
                self.transition,
            )


@pytest.mark.parametrize("transition", ["ge", "ef"])
@pytest.mark.parametrize("count", [10, 12])
class TestAmplitudeRabiTwoQubit:
    @pytest.fixture(autouse=True)
    def _setup(self, two_tunable_transmon, transition, count):
        self.q0, self.q1 = two_tunable_transmon.qubits
        self.amplitudes = [[0.1, 0.5, 1], [0.1, 0.5, 1]]
        self.count = count
        self.transition = transition
        self.qop = TunableTransmonOperations()

    def test_run_standalone(self):
        exp = amplitude_rabi(
            self.qop,
            [self.q0, self.q1],
            self.amplitudes,
            self.count,
            self.transition,
        )
        assert exp == reference_rabi_exp(
            [self.q0, self.q1],
            self.count,
            self.amplitudes,
            self.transition,
        )

    def test_run_task(self):
        exp = amplitude_rabi(
            self.qop,
            [self.q0, self.q1],
            self.amplitudes,
            self.count,
            self.transition,
        )
        with Workflow() as wf:
            amplitude_rabi(
                self.qop,
                [self.q0, self.q1],
                self.amplitudes,
                self.count,
                self.transition,
            )
        assert wf.run().tasklog == {"amplitude_rabi": [exp]}

    def test_invalid_input_raises_error(self):
        with pytest.raises(ValueError):
            amplitude_rabi(
                self.qop,
                [self.q0, self.q1],
                [0.1, 0.5],
                self.count,
                self.transition,
            )

        with pytest.raises(ValueError):
            amplitude_rabi(
                self.qop,
                [self.q0, self.q1],
                [[0.1, 0.5]],
                self.count,
                self.transition,
            )
        with pytest.raises(ValueError):
            amplitude_rabi(
                self.qop,
                [self.q0, self.q1],
                [[0.1, 0.5], [0.1, None]],
                self.count,
                self.transition,
            )
