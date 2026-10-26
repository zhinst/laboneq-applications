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
        q0_amplitude: float | SweepParameter | None = None,
    ) -> None:
        """CZ gate via flux pulses on the coupler (and optionally the qubit).

        The gate parameters are read from the ``("cz", q0.uid, q1.uid)``
        topology edge ([CzParameters][]). Two flavors are supported,
        selected by the edge parameters:

        - If the edge's ``control_pulse`` is `None` (the default), a
          parametric flux pulse is played on the tunable coupler only.
        - If the edge's ``control_pulse`` is set, the parametric flux pulse
          is played on the flux line of the qubit with the higher ge
          resonance frequency, with the coupler pulse played simultaneously
          as a bias.

        In both cases the gate ends with single-qubit Rz phase corrections
        on both qubits.

        Arguments:
            q0:
                The control qubit.
            q1:
                The target qubit.
            tc_amplitude:
                Override for the coupler-pulse amplitude. Defaults to the
                edge's ``coupler_pulse.amplitude``.
            length:
                Override for the flux-pulse length. Defaults to the
                edge's ``coupler_pulse.length``.
            frequency:
                Override for the parametric-drive modulation frequency.
                Applied to the coupler pulse, or to the qubit pulse when the
                edge's ``control_pulse`` is set.
            control_angle:
                Override for the post-pulse Rz angle on ``q0``. Defaults to
                the edge's ``control_angle``.
            target_angle:
                Override for the post-pulse Rz angle on ``q1``. Defaults to
                the edge's ``target_angle``.
            q0_amplitude:
                Override for the qubit-pulse amplitude (only used when the
                edge's ``control_pulse`` is set). Defaults to the edge's
                ``control_pulse.amplitude``.
        """
        edge = self.qpu.topology["cz", q0.uid, q1.uid]
        coupler = edge.quantum_element
        parameters = edge.parameters

        tc_params = parameters.coupler_pulse.copy()
        q0_params = (
            parameters.control_pulse.copy()
            if parameters.control_pulse is not None
            else None
        )

        if tc_amplitude is None:
            tc_amplitude = tc_params["amplitude"]
        if length is None:
            length = tc_params["length"]
        if control_angle is None:
            control_angle = parameters.control_angle
        if target_angle is None:
            target_angle = parameters.target_angle

        if frequency is not None:
            if q0_params is not None:
                q0_params["frequency"] = frequency
            else:
                tc_params["frequency"] = frequency

        fp_tc = dsl.create_pulse(
            {k: v for k, v in tc_params.items() if k not in ["length", "amplitude"]},
            name="pulse_coupler",
        )

        if q0_params is not None:
            # qubit-driven flavor: parametric drive on the qubit with the
            # higher ge frequency, coupler pulse as a simultaneous bias
            if q0_amplitude is None:
                q0_amplitude = q0_params["amplitude"]

            qubit_flux = (
                q0
                if q0.parameters.resonance_frequency_ge
                > q1.parameters.resonance_frequency_ge
                else q1
            )
            fp_q0 = dsl.create_pulse(
                {
                    k: v
                    for k, v in q0_params.items()
                    if k not in ["length", "amplitude"]
                },
                name="pulse_qubit",
            )
            flux_line, _ = qubit_flux.flux_parameters()
            dsl.play(
                signal=qubit_flux.signals[flux_line],
                pulse=fp_q0,
                length=length,
                amplitude=q0_amplitude,
            )

        dsl.play(
            signal=coupler.signals["flux"],
            pulse=fp_tc,
            length=length,
            amplitude=tc_amplitude,
        )
        self.rz.omit_section(q0, angle=control_angle)
        self.rz.omit_section(q1, angle=target_angle)

    @dsl.quantum_operation(broadcast=False)
    def iswap(
        self,
        q0: TunableTransmonQubit,
        q1: TunableTransmonQubit,
        amplitude_fp_q1: float | SweepParameter | None = None,
        amplitude_fp_tc: float | SweepParameter | None = None,
        length: float | SweepParameter | None = None,
    ) -> None:
        """ISWAP gate via parametric resonance mediated by a tunable coupler.

        The gate parameters are read from the ``("iswap", q0.uid, q1.uid)``
        topology edge ([IswapParameters][]).

        Arguments:
            q0:
                The target qubit.
            q1:
                The control qubit.
            amplitude_fp_q1:
                Override for the fast-flux pulse amplitude on the
                control-qubit flux line. Defaults to the edge's
                ``control_pulse.amplitude``.
            amplitude_fp_tc:
                Override for the fast-flux pulse amplitude on the coupler
                flux line. Defaults to the edge's ``coupler_pulse.amplitude``.
            length:
                Override for the length of both fast-flux pulses. Defaults
                to the edge's ``coupler_pulse.length``.
        """
        edge = self.qpu.topology["iswap", q0.uid, q1.uid]
        coupler = edge.quantum_element
        parameters = edge.parameters

        q1_params = parameters.control_pulse
        tc_params = parameters.coupler_pulse

        if amplitude_fp_q1 is None:
            amplitude_fp_q1 = q1_params["amplitude"]
        if amplitude_fp_tc is None:
            amplitude_fp_tc = tc_params["amplitude"]
        if length is None:
            length = tc_params["length"]

        # Note that here the other parameters of the control puslse
        # and the coupler pulse (except 'funtion') are silently dropped here.
        # This may not be the desired behavior.
        fp_q1 = dsl.create_pulse(
            {"function": q1_params["function"]}, name="ff_pulse_q1"
        )
        fp_tc = dsl.create_pulse(
            {"function": tc_params["function"]}, name="ff_pulse_coupler"
        )

        flux_line, _ = q1.flux_parameters()
        dsl.play(
            signal=q1.signals[flux_line],
            pulse=fp_q1,
            length=length,
            amplitude=amplitude_fp_q1,
        )
        dsl.play(
            signal=coupler.signals["flux"],
            pulse=fp_tc,
            length=length,
            amplitude=amplitude_fp_tc,
        )

    @dsl.quantum_operation(broadcast=False)
    def discharge_coupler(
        self,
        q0: TunableTransmonQubit,
        q1: TunableTransmonQubit,
        amplitude: float | SweepParameter | None = None,
        length: float | SweepParameter | None = None,
        tag: str = "cz",
    ) -> None:
        """Discharge the coupler mediating the two qubits passed.

        Plays the coupler fast-flux pulse with inverted amplitude.

        Arguments:
            q0:
                The target qubit.
            q1:
                The control qubit.
            amplitude:
                Amplitude of the discharge pulse played on the coupler flux
                line (before inversion). Defaults to the edge's
                ``coupler_pulse.amplitude``.
            length:
                Length of the discharge pulse. Defaults to the edge's
                ``coupler_pulse.length``.
            tag:
                The topology-edge tag used to look up the coupler and its
                pulse parameters. Default: "cz".
        """
        edge = self.qpu.topology[tag, q0.uid, q1.uid]
        coupler = edge.quantum_element
        parameters = edge.parameters

        tc_params = parameters.coupler_pulse

        if amplitude is None:
            amplitude = tc_params["amplitude"]
        if length is None:
            length = tc_params["length"]

        fp_tc = dsl.create_pulse(
            {"function": tc_params["function"]}, name="ff_pulse_coupler"
        )

        dsl.play(
            signal=coupler.signals["flux"],
            pulse=fp_tc,
            length=length,
            amplitude=-1 * amplitude,
        )
