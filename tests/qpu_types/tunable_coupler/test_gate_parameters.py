# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for laboneq_applications.qpu_types.tunable_coupler.gate_parameters."""

from laboneq.dsl.quantum import QuantumParameters

from laboneq_applications.qpu_types.tunable_coupler import CzParameters


class TestCzParameters:
    def test_create(self):
        p = CzParameters()
        assert isinstance(p, QuantumParameters)
        assert p.coupler_pulse == {
            "function": "modulated_flux",
            "amplitude": 0.0,
            "length": 1e-6,
            "frequency": 225e6,
        }
        assert p.control_angle == 0.0
        assert p.target_angle == 0.0

    def test_custom_parameters(self):
        p = CzParameters(
            coupler_pulse={
                "function": "modulated_flux",
                "amplitude": 0.7,
                "length": 500e-9,
                "frequency": 300e6,
            }
        )
        assert p.coupler_pulse["amplitude"] == 0.7
        p = CzParameters(
            coupler_pulse={
                "function": "modulated_flux",
                "amplitude": 0.0,
                "length": 200e-9,
                "frequency": 225e6,
            }
        )
        assert p.coupler_pulse["length"] == 200e-9
        p = CzParameters(
            coupler_pulse={
                "function": "modulated_flux",
                "amplitude": 0.0,
                "length": 1e-6,
                "frequency": 350e6,
            }
        )
        assert p.coupler_pulse["frequency"] == 350e6
        p = CzParameters(control_angle=1.23)
        assert p.control_angle == 1.23
        p = CzParameters(target_angle=0.45)
        assert p.target_angle == 0.45

    def test_coupler_pulse_instances_are_independent(self):
        p1 = CzParameters()
        p2 = CzParameters()
        p1.coupler_pulse["amplitude"] = 0.9
        assert p2.coupler_pulse["amplitude"] == 0.0
