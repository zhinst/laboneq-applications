""" Collection of smoke tests for the experiments in the tuneup module.
"""
from laboneq.simple import *

from laboneq_library.automatic_tuneup.tuneup.experiment import (
    AmplitudeRabi,
    ParallelResSpecCW,
    PulsedQubitSpecBiasSweep,
    PulsedQubitSpectroscopy,
    Ramsey,
    ReadoutSpectroscopyCWBiasSweep,
    ResonatorCWSpec,
    ResonatorPulsedSpec,
)


def test_rspec_exp(qubit_configs, session):
    exp_cls = ResonatorCWSpec(qubit_configs)
    exp = exp_cls.exp
    compiled_exp = session.compile(exp)
    session.run(compiled_exp)


def test_parallel_rspec_exp(qubit_configs, session):
    exp_cls = ParallelResSpecCW(
        qubit_configs, exp_settings={"integration_time": 100e-6, "num_averages": 2**5}
    )
    exp = exp_cls.exp
    compiled_exp = session.compile(exp)
    # show_pulse_sheet("test_parallel_rspec",compiled_exp)
    session.run(compiled_exp)


def test_rspec_flux_bias_sweep_exp(qubit_configs, session, set_bias_dc):
    ext_calls = set_bias_dc
    session.register_user_function(ext_calls)
    qubit_configs[0].parameter.flux = [LinearSweepParameter(start=-5, stop=5, count=11)]
    exp_cls = ReadoutSpectroscopyCWBiasSweep(
        qubit_configs,
        exp_settings={"integration_time": 100e-6, "num_averages": 2**5},
        ext_calls=ext_calls,
    )
    exp = exp_cls.exp
    compiled_exp = session.compile(exp)
    session.run(compiled_exp)


def test_resonator_pulsed_spec_exp(qubit_configs, session):
    exp_cls = ResonatorPulsedSpec(qubit_configs)
    exp = exp_cls.exp
    compiled_exp = session.compile(exp)
    session.run(compiled_exp)


def test_qubit_pulsed_spec_exp(qubit_configs, session):
    exp_cls = PulsedQubitSpectroscopy(qubit_configs)
    exp = exp_cls.exp
    compiled_exp = session.compile(exp)
    session.run(compiled_exp)


def test_pulsed_qubit_spec_flux_bias_sweep_exp(qubit_configs, session, set_bias_dc):
    ext_calls = set_bias_dc
    session.register_user_function(ext_calls)
    qubit_configs[0].parameter.flux = [LinearSweepParameter(start=-5, stop=5, count=11)]
    exp_cls = PulsedQubitSpecBiasSweep(
        qubit_configs,
        exp_settings={"integration_time": 100e-6, "num_averages": 2**5},
        ext_calls=ext_calls,
    )
    exp = exp_cls.exp
    compiled_exp = session.compile(exp)
    session.run(compiled_exp)


def test_amplitude_rabi_exp(qubit_configs, session):
    qubit_configs[0].parameter.amplitude = [
        LinearSweepParameter(start=0, stop=1, count=11)
    ]
    exp_cls = AmplitudeRabi(
        qubit_configs,
        exp_settings={"integration_time": 100e-6, "num_averages": 2**5},
    )
    exp = exp_cls.exp
    compiled = session.compile(exp)
    session.run(compiled)


def test_ramsey_exp(qubit_configs, session):
    qubit_configs[0].parameter.delay = [
        LinearSweepParameter(start=0, stop=1e-6, count=11)
    ]
    exp_cls = Ramsey(
        qubit_configs,
        exp_settings={"integration_time": 100e-6, "num_averages": 2**5},
    )
    exp = exp_cls.exp
    compiled = session.compile(exp)
    show_pulse_sheet("ramsey", compiled)
    session.run(compiled)
