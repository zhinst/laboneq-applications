# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for laboneq_applications.qpu_types.tunable_coupler.pulses."""

import numpy as np
import pytest
from laboneq.dsl.experiment.pulse import PulseFunctional
from laboneq.simple import dsl

# Importing the module registers modulated_flux in the pulse library.
from laboneq_applications.qpu_types.tunable_coupler.pulses import modulated_flux

# The raw sampler function (operating on numpy arrays) is stored separately
# from the PulseFunctional factory returned by the decorator.
_sampler = dsl.pulse_library._pulse_samplers["modulated_flux"]


class TestModulatedFlux:
    def test_factory(self):
        assert "modulated_flux" in dsl.pulse_library._pulse_samplers

        pulse = modulated_flux(uid="test", length=1e-6, amplitude=0.5, frequency=225e6)
        assert isinstance(pulse, PulseFunctional)
        assert pulse.function == "modulated_flux"
        assert pulse.length == 1e-6
        assert pulse.amplitude == 0.5
        assert pulse.pulse_parameters == {"frequency": 225e6}

        default_pulse = modulated_flux(uid="test")
        assert default_pulse.length == 100e-9
        assert default_pulse.amplitude == 1.0

    @pytest.mark.parametrize("width", [1e-6, 2e-6])
    def test_width_must_be_less_than_length(self, width):
        x = np.linspace(-1, 1, 100)
        with pytest.raises(ValueError, match="smaller than the total length"):
            _sampler(x, width=width, length=1e-6)

    def test_zero_boundaries(self):
        x = np.linspace(-1, 1, 10001)
        length = 1e-6
        result_off = _sampler(
            x, frequency=150e6, length=length, sigma=2.0, zero_boundaries=False
        )
        result_on = _sampler(
            x, frequency=150e6, length=length, sigma=2.0, zero_boundaries=True
        )
        # The two modes produce different waveforms ...
        assert not np.allclose(result_off, result_on)
        # ... and zero_boundaries pulls the edge envelope down.
        n_edge = 50
        assert np.max(np.abs(result_on[:n_edge])) <= np.max(np.abs(result_off[:n_edge]))

    def test_flat_top_is_fully_modulated(self):
        """At the centre sample, envelope == 1, so value == sin(2π f t_centre)."""
        x = np.linspace(-1, 1, 10001)
        length = 1e-6
        # t_centre = 0.5 µs; choose f so 2π f t_centre = 250.5π = 250π + π/2 → sin = 1
        frequency = 250.5e6
        result = _sampler(x, frequency=frequency, width=0.5 * length, length=length)
        expected = np.sin(2 * np.pi * frequency * 0.5 * length)
        assert abs(result[len(x) // 2] - expected) < 1e-6
