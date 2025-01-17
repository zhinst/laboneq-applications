# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Traveling wave parametric amplifier (TWPA) parameters and elements."""

from __future__ import annotations

import attrs
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration import (
    AmplifierPump,
    Calibration,
    CancellationSource,
    Oscillator,
    SignalCalibration,
)
from laboneq.dsl.enums import ModulationType
from laboneq.simple import (
    QuantumElement,
    QuantumParameters,
)


@classformatter
@attrs.define
class TWPAParameters(QuantumParameters):
    """Parameters for the TWPA."""

    # QA related parameters
    # Probe frequency by default it set to readout resonator frequency
    probe_frequency: float | None = None
    # local oscillator frequency for the readout lines.
    readout_lo_frequency: float = None
    # readout pulse
    readout_pulse: dict = attrs.field(
        factory=lambda: {
            "function": "const",
        },
    )
    # readout output amplitude setting, defaults to 0.5.
    readout_amplitude: float | None = 0.5
    # readout output length setting, defaults to 5e-6.
    readout_length: float | None = 5e-6
    # readout output power setting, defaults to 5 dBm.
    readout_range_out: float | None = 5
    # readout input power setting, defaults to 10 dBm.
    readout_range_in: float | None = 10
    # set integration delay
    readout_integration_delay: float | None = 20e-9

    # TWPA parameters

    # voltage bias, defaults to 0 V
    voltage_bias: float | None = 0
    # set electric delay in second
    electric_delay: float | None = 0
    # Pump frequency for ParamAmplifier
    pump_frequency: float | None = None
    # Pump Power for ParamAmplifier
    pump_power: float | None = None
    # Probe power
    probe_power: float | None = None
    # Cancellation tone attenuation
    cancellation_attenuation: float | None = None
    # Cancellation tone phaseshift
    cancellation_phase: float | None = None
    # Cancellation Source
    cancellation_source: CancellationSource | None = CancellationSource.EXTERNAL
    # Flag for turning on the pump
    pump_on: bool = True
    # Flag for the cancellation state
    cancellation_on: bool = False
    # Flag for the filter state
    pump_filter_on: bool = True
    # Flag for the probe state
    probe_on: bool = False
    # Flag for the alc state
    alc_on: bool = True

    @property
    def readout_frequency(self) -> float | None:
        """Readout baseband frequency."""
        try:
            return self.probe_frequency - self.readout_lo_frequency
        except TypeError:
            return None


@classformatter
@attrs.define
class TWPA(QuantumElement):
    """A class for Traveling Wave Parametric Amplifiers."""

    PARAMETERS_TYPE = TWPAParameters
    REQUIRED_SIGNALS = (
        "acquire",
        "measure",
    )

    def readout_parameters(self) -> tuple[str, dict]:
        """Return the measure line and the readout parameters.

        Returns:
           line:
               The measure line of the qubit.
           params:
               The readout parameters.
        """
        param_keys = ["amplitude", "length", "pulse"]
        params = {k: getattr(self.parameters, f"readout_{k}") for k in param_keys}
        return "measure", params

    def calibration(self) -> Calibration:
        """Generate calibration from the parameters and attached signal lines.

        Returns:
            calibration:
                Prefilled calibration object from TWPA parameters.
        """
        readout_lo = None
        if self.parameters.readout_lo_frequency is not None:
            readout_lo = Oscillator(
                uid=f"{self.uid}_readout_local_osc",
                frequency=self.parameters.readout_lo_frequency,
            )
        if self.parameters.probe_frequency is not None:
            readout_oscillator = Oscillator(
                uid=f"{self.uid}_readout_acquire_osc",
                frequency=self.parameters.probe_frequency,
                modulation_type=ModulationType.AUTO,
            )

        calibration_items = {}

        # Apply calibration to the measure signal
        sig_cal = SignalCalibration()
        if self.parameters.probe_frequency is not None:
            sig_cal.oscillator = readout_oscillator
        sig_cal.local_oscillator = readout_lo
        sig_cal.range = self.parameters.readout_range_out
        sig_cal.amplitude = self.parameters.readout_amplitude
        calibration_items[self.signals["measure"]] = sig_cal

        # Apply calibration to the acquire signal
        sig_cal = SignalCalibration()
        if self.parameters.probe_frequency is not None:
            sig_cal.oscillator = readout_oscillator
        sig_cal.local_oscillator = readout_lo
        sig_cal.range = self.parameters.readout_range_in
        sig_cal.port_delay = self.parameters.readout_integration_delay
        sig_cal.amplifier_pump = AmplifierPump(
            pump_frequency=self.parameters.pump_frequency,
            pump_power=self.parameters.pump_power,
            pump_on=self.parameters.pump_on,
            pump_filter_on=self.parameters.pump_filter_on,
            cancellation_on=self.parameters.cancellation_on,
            cancellation_phase=self.parameters.cancellation_phase,
            cancellation_attenuation=self.parameters.cancellation_attenuation,
            cancellation_source=self.parameters.cancellation_source,
            cancellation_source_frequency=self.parameters.pump_frequency,
            alc_on=self.parameters.alc_on,
            probe_on=self.parameters.probe_on,
            probe_frequency=self.parameters.probe_frequency,
            probe_power=self.parameters.probe_power,
        )
        calibration_items[self.signals["acquire"]] = sig_cal
        return Calibration(calibration_items)
