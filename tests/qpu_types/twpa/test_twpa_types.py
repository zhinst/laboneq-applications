# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for laboneq_applications.qpu_types.twpa.twpa_types."""

import pytest
from laboneq.dsl.calibration import CancellationSource
from laboneq.simple import Calibration

from laboneq_applications.qpu_types.twpa.twpa_types import TWPA, TWPAParameters


@pytest.fixture
def twpa() -> TWPA:
    return TWPA("twpa0", signals={"acquire": "acquire", "measure": "measure"})


class TestTWPAParameters:
    def test_create(self):
        twpa_param = TWPAParameters()
        assert twpa_param.probe_frequency is None
        assert twpa_param.readout_lo_frequency is None
        assert twpa_param.readout_pulse == {"function": "const"}
        assert twpa_param.readout_amplitude == 0.5
        assert twpa_param.readout_length == 5e-6
        assert twpa_param.readout_range_out == 5
        assert twpa_param.readout_range_in == 10
        assert twpa_param.readout_integration_delay == 20e-9

        assert twpa_param.voltage_bias == 0
        assert twpa_param.electric_delay == 0
        assert twpa_param.pump_frequency is None
        assert twpa_param.pump_power is None
        assert twpa_param.probe_power is None
        assert twpa_param.cancellation_attenuation is None
        assert twpa_param.cancellation_phase is None
        assert twpa_param.cancellation_source is CancellationSource.EXTERNAL
        assert twpa_param.pump_on
        assert not twpa_param.cancellation_on
        assert twpa_param.pump_filter_on
        assert not twpa_param.probe_on
        assert twpa_param.alc_on

    def test_readout_frequency(self):
        twpa_param = TWPAParameters()
        assert twpa_param.readout_frequency is None

        twpa_param.probe_frequency = 6.1e9
        twpa_param.readout_lo_frequency = 6e9

        assert twpa_param.readout_frequency == 0.1e9


class TestTWPA:
    def test_create(self):
        twpa = TWPA("twpa0", signals={"acquire": "acquire", "measure": "measure"})
        assert isinstance(twpa.parameters, TWPAParameters)
        assert twpa.REQUIRED_SIGNALS == ("acquire", "measure")

    def test_readout_parameters(self, twpa):
        measure_line, params = twpa.readout_parameters()
        assert measure_line == "measure"
        assert params["length"] == 5e-6
        assert params["amplitude"] == 0.5

    def test_calibration(self, twpa):
        calibration = twpa.calibration()
        assert isinstance(calibration, Calibration)

        assert calibration["acquire"].amplifier_pump.pump_frequency is None
        assert calibration["acquire"].amplifier_pump.pump_power is None
        assert calibration["acquire"].amplifier_pump.pump_on
        assert calibration["acquire"].amplifier_pump.pump_filter_on
        assert not calibration["acquire"].amplifier_pump.cancellation_on

        assert calibration["acquire"].amplifier_pump.cancellation_phase is None
        assert calibration["acquire"].amplifier_pump.cancellation_attenuation is None
        assert (
            calibration["acquire"].amplifier_pump.cancellation_source
            == CancellationSource.EXTERNAL
        )
        assert (
            calibration["acquire"].amplifier_pump.cancellation_source_frequency is None
        )
        assert calibration["acquire"].amplifier_pump.alc_on
        assert not calibration["acquire"].amplifier_pump.probe_on
        assert calibration["acquire"].amplifier_pump.probe_frequency is None
        assert calibration["acquire"].amplifier_pump.probe_power is None

        assert calibration["acquire"].range == 10
        assert calibration["acquire"].local_oscillator is None
        assert calibration["acquire"].oscillator is None
        assert calibration["acquire"].port_delay == 20e-9

        assert calibration["measure"].range == 5
        assert calibration["measure"].local_oscillator is None
        assert calibration["measure"].oscillator is None
        assert calibration["measure"].amplitude == 0.5

    def test_update(self, twpa):
        # `update` method is tested more thoroughly in QuantumElement
        twpa.update(readout_lo_frequency=6.5e9)
        assert twpa.parameters.readout_lo_frequency == 6.5e9

    def test_replace(self, twpa):
        # `replace` method is tested more thoroughly in QuantumElement
        new_twpa = twpa.replace(readout_lo_frequency=6.5e9)
        assert isinstance(new_twpa, TWPA)
        assert new_twpa.parameters.readout_lo_frequency == 6.5e9
