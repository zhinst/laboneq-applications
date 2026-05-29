# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Quantum operations for two-qubit gates driven by tunable couplers.

`TunableCouplerOperations` extends `TunableTransmonOperations` with a CZ
gate that drives a flux pulse on a tunable coupler that connects the two
qubits via a ``"cz"`` topology edge. The edge must carry a [CzParameters][]
instance and reference the [TunableCoupler][] as the edge's
``quantum_element``.
"""

from __future__ import annotations

from laboneq.dsl.parameter import SweepParameter
from laboneq.simple import dsl

from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonOperations,
    TunableTransmonQubit,
)

# Re-import to register the modulated_flux pulse functional in the pulse
# library when this module is imported.
from . import pulses  # noqa: F401


class TunableCouplerOperations(TunableTransmonOperations):
    """Tunable-transmon operations plus a tunable-coupler-mediated CZ gate.

    Adds the ``cz`` operation on top of all the single-qubit operations in
    [TunableTransmonOperations][]. The qubit type is unchanged
    (`TunableTransmonQubit`); the coupler is read from the topology edge
    between the two qubits.
    """

    @dsl.quantum_operation(broadcast=False)
    def cz(
        self,
        q0: TunableTransmonQubit,
        q1: TunableTransmonQubit,
        tc_amplitude: float | SweepParameter | None = None,
        length: float | SweepParameter | None = None,
        frequency: float | SweepParameter | None = None,
        control_angle: float | SweepParameter | None = None,
        target_angle: float | SweepParameter | None = None,
    ) -> None:
        """CZ gate via a flux pulse on the tunable coupler between ``q0`` and ``q1``.

        Plays a flux pulse on the tunable coupler whose parameters are read
        from the ``("cz", q0.uid, q1.uid)`` topology edge ([CzParameters][]),
        then applies single-qubit Rz phase corrections on both qubits.

        Arguments:
            q0:
                The control qubit.
            q1:
                The target qubit.
            tc_amplitude:
                Override for the coupler-pulse amplitude. Defaults to the
                edge's ``coupler_pulse.amplitude``.
            length:
                Override for the coupler-pulse length. Defaults to the
                edge's ``coupler_pulse.length``.
            frequency:
                Override for the coupler-pulse modulation frequency. Defaults
                to the edge's ``coupler_pulse.frequency``.
            control_angle:
                Override for the post-pulse Rz angle on ``q0``. Defaults to
                the edge's ``control_angle``.
            target_angle:
                Override for the post-pulse Rz angle on ``q1``. Defaults to
                the edge's ``target_angle``.
        """
        edge = self.qpu.topology["cz", q0.uid, q1.uid]
        coupler = edge.quantum_element
        parameters = edge.parameters

        tc_params = parameters.coupler_pulse.copy()

        if tc_amplitude is None:
            tc_amplitude = tc_params["amplitude"]
        if length is None:
            length = tc_params["length"]
        if control_angle is None:
            control_angle = parameters.control_angle
        if target_angle is None:
            target_angle = parameters.target_angle

        if frequency is not None:
            tc_params["frequency"] = frequency

        fp_tc = dsl.create_pulse(
            {k: v for k, v in tc_params.items() if k not in ["length", "amplitude"]},
            name="pulse_coupler",
        )

        dsl.play(
            signal=coupler.signals["flux"],
            pulse=fp_tc,
            length=length,
            amplitude=tc_amplitude,
        )
        self.rz.omit_section(q0, angle=control_angle)
        self.rz.omit_section(q1, angle=target_angle)
