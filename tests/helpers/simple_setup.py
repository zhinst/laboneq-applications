"""Helper fixtures for a simple experiment setup."""

import pytest
from laboneq.dsl.calibration import Oscillator, SignalCalibration
from laboneq.dsl.experiment import Experiment, ExperimentSignal, pulse_library
from laboneq.dsl.session import Session


@pytest.fixture()
def simple_device_setup(single_tunable_transmon):
    device_setup = single_tunable_transmon.setup
    device_setup.logical_signal_by_uid("q0/measure").calibration = SignalCalibration(
        local_oscillator=Oscillator(frequency=5e9),
        oscillator=Oscillator(frequency=100e6),
    )
    device_setup.logical_signal_by_uid("q0/acquire").calibration = SignalCalibration(
        local_oscillator=Oscillator(frequency=5e9),
        oscillator=Oscillator(frequency=100e6),
    )
    return device_setup


@pytest.fixture()
def simple_experiment(simple_device_setup):
    exp = Experiment(signals=[ExperimentSignal("measure"), ExperimentSignal("acquire")])
    exp.map_signal("measure", simple_device_setup.logical_signal_by_uid("q0/measure"))
    exp.map_signal("acquire", simple_device_setup.logical_signal_by_uid("q0/acquire"))
    with exp.acquire_loop_rt(count=4), exp.section():
        exp.play("measure", pulse_library.const())
        exp.acquire("acquire", "ac0", length=100e-9)
        exp.delay("measure", 100e-9)
    return exp


@pytest.fixture()
def simple_session(simple_device_setup):
    session = Session(simple_device_setup)
    session.connect(do_emulation=True)
    return session
