# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Gate-edge parameters for tunable-coupler-mediated two-qubit gates.

These are stored on the topology edge (rather than on the qubit or coupler) so
that each pair of qubits can have its own gate calibration.
"""

from __future__ import annotations

import attrs
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.quantum import QuantumParameters


def _default_coupler_pulse() -> dict:
    return {
        "function": "modulated_flux",
        "amplitude": 0.0,
        "length": 1e-6,
        "frequency": 225e6,
    }


@classformatter
@attrs.define(kw_only=True)
class CzParameters(QuantumParameters):
    """Edge parameters for a CZ gate driven via a tunable coupler.

    Attributes:
        coupler_pulse:
            Pulse definition for the flux drive on the tunable coupler. Includes
            the pulse ``function`` name (defaults to ``"modulated_flux"``),
            ``amplitude``, ``length`` and parametric-resonance ``frequency``.
        control_angle:
            Single-qubit Rz angle applied to the control qubit after the
            flux pulse to correct its acquired dynamic phase.
        target_angle:
            Single-qubit Rz angle applied to the target qubit after the
            flux pulse to correct its acquired dynamic phase.
    """

    coupler_pulse: dict = attrs.field(factory=_default_coupler_pulse)
    control_angle: float = 0.0
    target_angle: float = 0.0
