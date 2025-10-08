# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tunable coupler parameters and elements."""

from __future__ import annotations

from typing import ClassVar

import attrs
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration import (
    Calibration,
    SignalCalibration,
)
from laboneq.dsl.quantum import (
    QuantumElement,
    QuantumParameters,
)


@classformatter
@attrs.define(kw_only=True)
class TunableCouplerParameters(QuantumParameters):
    """Tunable coupler parameters.

    Attributes:
        gate_parameters:
            Dictionary of parameters specific to each implemented gate. The dictionary
            keys are the names of the gates, and the dictionary values are the
            corresponding parameter dictionaries. Typical parameters include: "pulse",
            "amplitude", "length", and "frequency". By default, only the iSWAP gate
            parameters are initialized.
        dc_slot:
            Slot number on the DC source used for applying a DC voltage to the coupler.
        flux_offset_voltage:
            Offset voltage for flux control line - defaults to 0 volts.
        dc_voltage_parking:
            Coupler DC parking voltage.
    """

    # gate parameters

    gate_parameters: dict[str, dict[str, object]] = attrs.field(
        factory=lambda: {
            "iswap": {
                "pulse": {"function": "gaussian_square"},
                "amplitude": None,
                "length": None,
                "frequency": None,
            }
        }
    )

    # flux parameters

    dc_slot: int | None = 0
    dc_voltage_parking: float | None = None
    flux_offset_voltage: float = 0.0


@classformatter
@attrs.define
class TunableCoupler(QuantumElement):
    """Tunable coupler."""

    PARAMETERS_TYPE = TunableCouplerParameters
    REQUIRED_SIGNALS = ("flux",)
    SIGNAL_ALIASES: ClassVar = {"flux_line": "flux"}

    def gate_parameters(self, gate: str) -> tuple[str, dict]:
        """Returns the flux line and the parameters of the gate.

        Arguments:
            gate: The name of the gate.

        Returns:
           line:
               The flux line of the coupler.
           params:
               The gate parameters.
        """
        return "flux", self.parameters.gate_parameters[gate]

    def calibration(self) -> Calibration:
        """Returns the calibration for the tunable coupler.

        Returns:
            Prefilled calibration object from coupler parameters.
        """
        calibration = {}
        # define the flux signal calibration:
        sig_cal = SignalCalibration()
        if self.parameters.flux_offset_voltage is not None:
            sig_cal.voltage_offset = self.parameters.flux_offset_voltage
        calibration[self.signals["flux"]] = sig_cal

        return Calibration(calibration)
