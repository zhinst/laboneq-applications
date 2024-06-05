"""Tests for laboneq_applications.core.build_experiment."""

import numpy as np
import pytest
from laboneq.simple import (
    SweepParameter,
    Transmon,
)

import tests.helpers.dsl as tsl
from laboneq_applications import dsl
from laboneq_applications.core.build_experiment import (
    ExperimentBuilder,
    build,
)
from laboneq_applications.core.quantum_operations import (
    QuantumOperations,
    quantum_operation,
)


class DummyOperations(QuantumOperations):
    """Dummy operations for testing."""

    QUBIT_TYPE = Transmon

    @quantum_operation
    def x(self, q, amplitude):
        """A dummy quantum operation."""
        pulse = dsl.pulse_library.const()
        dsl.play(q.signals["drive"], pulse, amplitude=amplitude)


@pytest.fixture()
def dummy_ops():
    return DummyOperations()


@pytest.fixture()
def dummy_q():
    return Transmon(
        uid="q0",
        signals={
            "drive": "/lsg/q0/drive",
            "measure": "/lsg/q0/measure",
            "acquire": "/lsg/q0/acquire",
        },
    )


def reserve_ops_for_dummy_q():
    return [
        tsl.reserve_op(signal="/lsg/q0/drive"),
        tsl.reserve_op(signal="/lsg/q0/measure"),
        tsl.reserve_op(signal="/lsg/q0/acquire"),
    ]


def simple_exp(qop, q, amplitudes, shots, prep=None):
    with dsl.acquire_loop_rt(count=shots):
        with dsl.sweep(parameter=SweepParameter(values=amplitudes)) as amplitude:
            if prep:
                prep(q)
            qop.x(q, amplitude)


def simple_exp_with_cal(qop, q, amplitudes, shots, acquire_range):
    with dsl.acquire_loop_rt(count=shots):
        with dsl.sweep(parameter=SweepParameter(values=amplitudes)) as amplitude:
            qop.x(q, amplitude)

    calibration = dsl.experiment_calibration()
    calibration[q.signals["acquire"]].range = 12


class TestExperimentBuilder:
    def test_create(self):
        builder = ExperimentBuilder(simple_exp)
        assert builder.exp_func is simple_exp
        assert builder.name == "simple_exp"

    def test_create_with_name(self):
        builder = ExperimentBuilder(simple_exp, name="custom_name")
        assert builder.exp_func is simple_exp
        assert builder.name == "custom_name"

    def test_call(self, dummy_ops, dummy_q):
        builder = ExperimentBuilder(simple_exp)
        amplitudes = np.array([0.1, 0.2, 0.3])
        shots = 5
        exp = builder(dummy_ops, dummy_q, amplitudes, shots=shots)

        assert exp == tsl.experiment(uid="simple_exp").children(
            tsl.acquire_loop_rt().children(
                tsl.sweep(parameters=[tsl.sweep_parameter(values=amplitudes)]).children(
                    tsl.section(uid="x_q0_0").children(
                        reserve_ops_for_dummy_q(),
                        tsl.play_pulse_op(
                            signal="/lsg/q0/drive",
                            amplitude=tsl.sweep_parameter(values=amplitudes),
                        ),
                    ),
                ),
            ),
        )

    def test_calibration(self, dummy_ops, dummy_q):
        builder = ExperimentBuilder(simple_exp)
        amplitudes = np.array([0.1, 0.2, 0.3])
        shots = 5
        exp = builder(dummy_ops, dummy_q, amplitudes, shots=shots)

        assert exp.get_calibration() == tsl.calibration(
            calibration_items={
                "/lsg/q0/acquire": tsl.signal_calibration(range=10),
                "/lsg/q0/drive": tsl.signal_calibration(range=10),
                "/lsg/q0/measure": tsl.signal_calibration(range=5),
            },
        )

    def test_setting_calibration_inside_experiment(self, dummy_ops, dummy_q):
        builder = ExperimentBuilder(simple_exp_with_cal)
        amplitudes = np.array([0.1, 0.2, 0.3])
        shots = 5
        acquire_range = 12
        exp = builder(
            dummy_ops,
            dummy_q,
            amplitudes,
            shots=shots,
            acquire_range=acquire_range,
        )

        assert exp.get_calibration() == tsl.calibration(
            calibration_items={
                "/lsg/q0/acquire": tsl.signal_calibration(range=12),
                "/lsg/q0/drive": tsl.signal_calibration(range=10),
                "/lsg/q0/measure": tsl.signal_calibration(range=5),
            },
        )


class TestQubitExperiment:
    def test_decorator(self, dummy_ops, dummy_q):
        @dsl.qubit_experiment
        def exp_1(qop, q, amplitude):
            qop.x(q, amplitude)

        exp = exp_1(dummy_ops, dummy_q, amplitude=0.1)

        assert exp == tsl.experiment(uid="exp_1").children(
            tsl.section(uid="x_q0_0").children(
                reserve_ops_for_dummy_q(),
                tsl.play_pulse_op(signal="/lsg/q0/drive", amplitude=0.1),
            ),
        )

    def test_partial_decorator(self, dummy_ops, dummy_q):
        @dsl.qubit_experiment(name="custom_name")
        def exp_2(qop, q, amplitude):
            qop.x(q, amplitude)

        exp = exp_2(dummy_ops, dummy_q, amplitude=0.2)

        assert exp == tsl.experiment(uid="custom_name").children(
            tsl.section(uid="x_q0_0").children(
                reserve_ops_for_dummy_q(),
                tsl.play_pulse_op(signal="/lsg/q0/drive", amplitude=0.2),
            ),
        )


class TestBuild:
    def test_build(self, dummy_ops, dummy_q):
        amplitudes = np.array([0.1, 0.2, 0.3])
        shots = 5
        exp = build(simple_exp, dummy_ops, dummy_q, amplitudes, shots=shots)

        assert exp == tsl.experiment(uid="simple_exp").children(
            tsl.acquire_loop_rt().children(
                tsl.sweep(parameters=[tsl.sweep_parameter(values=amplitudes)]).children(
                    tsl.section(uid="x_q0_0").children(
                        reserve_ops_for_dummy_q(),
                        tsl.play_pulse_op(
                            signal="/lsg/q0/drive",
                            amplitude=tsl.sweep_parameter(values=amplitudes),
                        ),
                    ),
                ),
            ),
        )
