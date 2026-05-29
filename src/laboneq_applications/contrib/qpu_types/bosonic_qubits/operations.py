# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Bosonic qubit operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from laboneq.dsl.calibration import Calibration, Oscillator
from laboneq.dsl.enums import ModulationType
from laboneq.dsl.parameter import SweepParameter
from laboneq.simple import SectionAlignment, dsl

from laboneq_applications.typing import QuantumElements

from .qubit_types import BosonicQubit


class BosonicQubitOperations(dsl.QuantumOperations):
    """Operations for BosonicQubits.

    Operations on the ancilla transmon are calibrated for the case
    where the memory cavity is in the vacuum state (|0⟩).
    """

    QUBIT_TYPES = BosonicQubit

    # common angles used by rx, ry and rz.
    _PI = np.pi
    _PI_BY_2 = np.pi / 2

    @dsl.quantum_operation
    def barrier(self, q: BosonicQubit) -> None:
        """Add a barrier on all the qubit signals.

        Arguments:
            q:
                The qubit to block on.

        Note:
            A barrier returns an empty section that
            reserves all the qubit signals. The
            signals are reserved via `@dsl.quantum_operation` so
            the implementation of this operations is just
            `pass`.
        """

    @dsl.quantum_operation
    def delay(self, q: BosonicQubit, time: float) -> None:
        """Add a delay on the transmon drive signal.

        Arguments:
            q:
                The qubit to delay on.
            time:
                The duration of the delay in seconds.
        """
        signal_line, _ = q.selective_transmon_Rx_parameters(0)
        dsl.delay(q.signals[signal_line], time=time)

    @dsl.quantum_operation
    def set_frequency(
        self,
        q: BosonicQubit,
        frequency: float | SweepParameter,
        *,
        target: Literal["transmon", "memory", "readout"] = "transmon",
        n: int = 0,
        rf: bool = True,
        calibration: Calibration | None = None,
    ) -> None:
        """Sets the frequency of the given qubit drive or readout line.

        Arguments:
            q:
                The qubit to set the frequency of.
            frequency:
                The frequency to set in Hz.
                By default the frequency specified is the RF frequency.
                The oscillator frequency may be set directly instead
                by passing `rf=False`.
            target:
                The target line to set the frequency of. One of
                "transmon" (ancilla transmon drive), "memory"
                (memory cavity drive), or "readout". Defaults to
                "transmon".
            n:
                The cavity photon number. Only used when
                target="transmon". Determines which selective
                signal line is targeted. Default: 0.
            rf:
                If True, set the RF frequency.
                If False, set the oscillator frequency directly.
                The default is to set the RF frequency.
            calibration:
                The experiment calibration to update.
                By default, the calibration from the currently active
                experiment context is used.

        Raises:
            ValueError:
                If the target is not "transmon", "memory" or
                "readout".
            RuntimeError:
                If there is an attempt to call `set_frequency`
                more than once on the same signal.
        """
        if target == "transmon":
            signal_line, _ = q.selective_transmon_Rx_parameters(n)
            lo_frequency = q.parameters.transmon_lo_frequency
        elif target == "memory":
            signal_line, _ = q.memory_drive_parameters()
            lo_frequency = q.parameters.memory_lo_frequency
        elif target == "readout":
            signal_line, _ = q.readout_parameters()
            lo_frequency = q.parameters.readout_lo_frequency
        else:
            raise ValueError(
                f"Target must be 'transmon', 'memory' or 'readout', not {target!r}.",
            )

        if rf:
            frequency -= lo_frequency

        if calibration is None:
            calibration = dsl.experiment_calibration()
        signal_calibration = calibration[q.signals[signal_line]]
        oscillator = signal_calibration.oscillator

        if oscillator is None:
            oscillator = signal_calibration.oscillator = Oscillator(
                frequency=frequency,
            )
        if getattr(oscillator, "_set_frequency", False):
            raise RuntimeError(
                f"Frequency of qubit {q.uid} {signal_line} line "
                f"was set multiple times using the "
                f"set_frequency operation.",
            )

        oscillator._set_frequency = True
        oscillator.frequency = frequency
        if target == "readout":
            oscillator.modulation_type = ModulationType.HARDWARE

    @dsl.quantum_operation
    def set_readout_amplitude(
        self,
        q: BosonicQubit,
        amplitude: float | SweepParameter,
        *,
        calibration: Calibration | None = None,
    ) -> None:
        """Sets the readout amplitude of the given qubit's measure line.

        Arguments:
            q:
                The qubit to set the readout amplitude of.
            amplitude:
                The amplitude to set for the measure line
                in units from 0 (no power) to 1 (full scale).
            calibration:
                The experiment calibration to update (see the note below).
                By default, the calibration from the currently active
                experiment context is used. If no experiment context is
                active, the calibration object may be passed explicitly.

        Raises:
            RuntimeError:
                If there is an attempt to call `set_readout_amplitude` more than
                once on the same signal.

        Notes:
            Currently `set_readout_amplitude` is implemented by setting the
            amplitude of the measure line signal in the experiment calibration.
            This has two important consequences:

            * Each experiment may only set one amplitude per readout line,
                although this may be a parameter sweep.

            * The set readout amplitude or sweep applies for the whole experiment
                regardless of where in the experiment the amplitude is set.

            This will be improved in a future release.
        """
        if calibration is None:
            calibration = dsl.experiment_calibration()
        measure_line, _ = q.readout_parameters()
        signal_calibration = calibration[q.signals[measure_line]]

        if getattr(calibration, "_set_readout_amplitude", False):
            raise RuntimeError(
                f"Readout amplitude of qubit {q.uid}"
                f" measure line was set multiple times"
                f" using the set_readout_amplitude operation.",
            )

        calibration._set_readout_amplitude = True
        signal_calibration.amplitude = amplitude

    @dsl.quantum_operation
    def measure(
        self,
        q: BosonicQubit,
        handle: str,
        readout_pulse: dict | None = None,
        kernel_pulses: list[dict]
        | Literal["default", "continuous", "optimal"]
        | None = None,
    ) -> None:
        """Perform a measurement on the qubit.

        The measurement is performed via the readout resonator, which
        dispersively reads out the ancilla transmon state.

        Arguments:
            q:
                The qubit to measure.
            handle:
                The handle to store the acquisition results in.
            readout_pulse:
                A dictionary of overrides for the readout pulse parameters.

                The dictionary may contain sweep parameters for the pulse
                parameters other than `function`.

                If the `function` parameter is different to the one
                specified for the qubit, then this override dictionary
                completely replaces the existing pulse parameters.

                Otherwise the values override or extend the existing ones.
            kernel_pulses:
                Custom definitions for the kernel pulses, passed as a list of
                pulse dictionaries, or one of the values "default", "optimal",
                or "continuous".

                If not specified or `None`, the qubit's
                `readout_integration_kernels_type` is used to select one
                of "default", "optimal" or "continuous".

                If `"default"` is passed, a constant integration
                kernel of length equal to the qubit's
                `readout_integration_length` parameter is used.

                If `"optimal"` is passed, the kernels specified by
                the qubit's `readout_integration_kernels` parameter are
                used.

                If `"continuous"` is passed, the hardware integrates for the
                entire integration length, weighting all samples equally.

                If a list of dictionaries is passed, each dictionary must
                completely specify a kernel pulse and its parameters.
        """
        measure_line, ro_params = q.readout_parameters()
        acquire_line, ro_int_params = q.readout_integration_parameters()
        ro_pulse = dsl.create_pulse(
            ro_params["pulse"], readout_pulse, name="readout_pulse"
        )

        kernels = q.get_integration_kernels(kernel_pulses)

        dsl.measure(
            measure_signal=q.signals[measure_line],
            measure_pulse_amplitude=ro_params["amplitude"],
            measure_pulse_length=ro_params["length"],
            measure_pulse=ro_pulse,
            handle=handle,
            acquire_signal=q.signals[acquire_line],
            integration_kernel=kernels,
            integration_length=ro_int_params["length"],
            reset_delay=None,
        )

    @dsl.quantum_operation
    def acquire(
        self,
        q: BosonicQubit,
        handle: str,
        kernel_pulses: list[dict]
        | Literal["default", "optimal", "continuous"]
        | None = None,
    ) -> None:
        """Perform an acquisition on the qubit.

        The acquire operation performs only an acquisition. If you wish to play
        a readout pulse and perform an acquisition, use the `measure` operation.

        Arguments:
            q:
                The qubit to measure.
            handle:
                The handle to store the acquisition results in.
            kernel_pulses:
                Custom definitions for the kernel pulses, passed as a list of
                pulse dictionaries, or one of the values "default", "optimal",
                or "continuous".

                If not specified or `None`, the qubit's
                `readout_integration_kernels_type` is used to select one
                of "default", "optimal" or "continuous".

                If `"default"` is passed, a constant integration
                kernel of length equal to the qubit's
                `readout_integration_length` parameter is used.

                If `"optimal"` is passed, the kernels specified by
                the qubit's `readout_integration_kernels` parameter are
                used.

                If `"continuous"` is passed, the hardware integrates for the
                entire integration length, weighting all samples equally.

                If a list of dictionaries is passed, each dictionary must
                completely specify a kernel pulse and its parameters.
        """
        acquire_line, ro_int_params = q.readout_integration_parameters()
        kernels = q.get_integration_kernels(kernel_pulses)

        dsl.acquire(
            signal=q.signals[acquire_line],
            handle=handle,
            kernel=kernels,
            length=ro_int_params["length"],
        )

    @dsl.quantum_operation
    def prepare_state(
        self,
        q: BosonicQubit,
        state: str = "g",
        reset: Literal["active", "passive"] | None = None,
    ) -> None:
        """Prepare the ancilla transmon in the given state.

        The ancilla transmon is assumed to be in the ground state, 'g'.
        If this is not the case pass `reset="passive"` or `reset="active"`
        to perform a passive or active reset operation before preparing
        the state.

        The returned section is right-aligned to ensure that there is no
        time gap between the end of the preparation pulses and the end
        of the section.

        Arguments:
            q:
                The qubit to prepare.
            state:
                The state to prepare. One of 'g' or 'e'.
            reset:
                If not None, perform the specified reset operation before
                preparing the state.
        """
        if reset is None:
            pass
        elif reset == "passive":
            self.passive_reset(q)
        elif reset == "active":
            raise NotImplementedError(
                "The active reset operation is not yet implemented."
            )
        else:
            raise ValueError(
                f"The reset parameter to prepare_state must be 'active',"
                f" 'passive', or None, not: {reset!r}",
            )

        if state == "g":
            pass
        elif state == "e":
            sec = self.x180(q)
            sec.alignment = SectionAlignment.RIGHT
        else:
            raise ValueError(f"Only states g and e can be prepared, not {state!r}")

    @dsl.quantum_operation
    def passive_reset(
        self,
        q: BosonicQubit,
        delay: float | SweepParameter | None = None,
    ) -> None:
        """Reset the qubit into the ground state using a long delay.

        Arguments:
            q:
                The qubit to reset.
            delay:
                The duration of the delay in seconds. Defaults
                to the qubit parameter `reset_delay_length`.
        """
        if delay is None:
            delay = q.parameters.reset_delay_length
        self.delay.omit_section(q, time=delay)

    # -----------------------------------------------------------------
    # Ancilla transmon rotations
    # -----------------------------------------------------------------
    # The following operations perform rotations on the ancilla transmon.
    # The drive is applied at the number-dependent frequency f_transmon(n)
    # = f_transmon(0) + n * chi, where n is the cavity photon number
    # specified by the n argument. The drive parameters are selected based
    # on this n value, so that the correct number-dependent transition is
    # driven.
    # -----------------------------------------------------------------

    @dsl.quantum_operation
    def rx(
        self,
        q: BosonicQubit,
        angle: float | SweepParameter | None,
        n: int = 0,
        amplitude: float | SweepParameter | None = None,
        phase: float = 0.0,
        increment_oscillator_phase: float | SweepParameter | None = None,
        length: float | SweepParameter | None = None,
        pulse: dict | None = None,
    ) -> None:
        """Rotate the ancilla transmon by the given angle about the X axis.

        The drive is applied at the transmon frequency conditioned on
        the memory cavity containing n photons:

            f_transmon(n) = f_transmon(0) + n * chi

        Arguments:
            q:
                The qubit to rotate.
            angle:
                The angle to rotate by in radians.
            n:
                The cavity photon number that determines which
                number-dependent transmon transition is driven.
                Default: 0 (memory in vacuum state).
            amplitude:
                The amplitude of the rotation pulse. By default this
                is determined by the angle and the pi pulse amplitude
                for photon number n by linear interpolation.
            phase:
                The phase of the rotation pulse in radians, applied
                as a baseband rotation of the waveform. Default: 0.0.
            increment_oscillator_phase:
                The phase increment on the baseband oscillator in
                radians. Default: None.
            length:
                The duration of the rotation pulse. By default this
                is determined by the qubit parameters.
            pulse:
                A dictionary of overrides for the pulse parameters.
        """
        drive_line, params = q.selective_transmon_Rx_parameters(n)

        if amplitude is None:
            amplitude = (angle / self._PI) * params["amplitude_pi"]
        if length is None:
            length = params["length"]

        rx_pulse = dsl.create_pulse(
            params["pulse"],
            pulse,
            name="rx_pulse",
        )

        dsl.play(
            q.signals[drive_line],
            amplitude=amplitude,
            phase=phase,
            increment_oscillator_phase=increment_oscillator_phase,
            length=length,
            pulse=rx_pulse,
        )

    @dsl.quantum_operation
    def x90(
        self,
        q: BosonicQubit,
        n: int = 0,
        amplitude: float | None = None,
        phase: float = 0.0,
        increment_oscillator_phase: float | SweepParameter | None = None,
        length: float | None = None,
        pulse: dict | None = None,
    ) -> None:
        """Rotate the ancilla transmon by 90 degrees about the X axis.

        Arguments:
            q:
                The qubit to rotate.
            n:
                The cavity photon number. Default: 0.
            amplitude:
                The amplitude of the rotation pulse. By default this
                is the pi/2 amplitude for photon number n.
            phase:
                The phase of the rotation pulse in radians. Default: 0.0.
            increment_oscillator_phase:
                Phase increment on the baseband oscillator. Default: None.
            length:
                The duration of the rotation pulse. Default from
                qubit parameters.
            pulse:
                A dictionary of overrides for the pulse parameters.
        """
        if amplitude is None:
            _, params = q.selective_transmon_Rx_parameters(n)
            amplitude = params["amplitude_pi2"]

        self.rx.omit_section(
            q,
            self._PI_BY_2,
            n=n,
            amplitude=amplitude,
            phase=phase,
            increment_oscillator_phase=increment_oscillator_phase,
            length=length,
            pulse=pulse,
        )

    @dsl.quantum_operation
    def x180(
        self,
        q: BosonicQubit,
        n: int = 0,
        amplitude: float | None = None,
        phase: float = 0.0,
        increment_oscillator_phase: float | SweepParameter | None = None,
        length: float | None = None,
        pulse: dict | None = None,
    ) -> None:
        """Rotate the ancilla transmon by 180 degrees about the X axis.

        Arguments:
            q:
                The qubit to rotate.
            n:
                The cavity photon number. Default: 0.
            amplitude:
                The amplitude of the rotation pulse. By default this
                is the pi amplitude for photon number n.
            phase:
                The phase of the rotation pulse in radians. Default: 0.0.
            increment_oscillator_phase:
                Phase increment on the baseband oscillator. Default: None.
            length:
                The duration of the rotation pulse. Default from
                qubit parameters.
            pulse:
                A dictionary of overrides for the pulse parameters.
        """
        if amplitude is None:
            _, params = q.selective_transmon_Rx_parameters(n)
            amplitude = params["amplitude_pi"]

        self.rx.omit_section(
            q,
            self._PI,
            n=n,
            amplitude=amplitude,
            phase=phase,
            increment_oscillator_phase=increment_oscillator_phase,
            length=length,
            pulse=pulse,
        )

    @dsl.quantum_operation
    def ry(
        self,
        q: BosonicQubit,
        angle: float | SweepParameter | None,
        n: int = 0,
        amplitude: float | SweepParameter | None = None,
        phase: float = _PI_BY_2,
        increment_oscillator_phase: float | SweepParameter | None = None,
        length: float | SweepParameter | None = None,
        pulse: dict | None = None,
    ) -> None:
        """Rotate the ancilla transmon by the given angle about the Y axis.

        Arguments:
            q:
                The qubit to rotate.
            angle:
                The angle to rotate by in radians.
            n:
                The cavity photon number. Default: 0.
            amplitude:
                The amplitude of the rotation pulse. By default this
                is determined by the angle and the pi pulse amplitude
                for photon number n.
            phase:
                The phase of the rotation pulse in radians.
                Default: pi/2.
            increment_oscillator_phase:
                Phase increment on the baseband oscillator. Default: None.
            length:
                The duration of the rotation pulse. Default from
                qubit parameters.
            pulse:
                A dictionary of overrides for the pulse parameters.
        """
        drive_line, params = q.selective_transmon_Rx_parameters(n)

        if amplitude is None:
            amplitude = (angle / self._PI) * params["amplitude_pi"]
        if length is None:
            length = params["length"]

        ry_pulse = dsl.create_pulse(
            params["pulse"],
            pulse,
            name="ry_pulse",
        )

        dsl.play(
            q.signals[drive_line],
            amplitude=amplitude,
            phase=phase,
            increment_oscillator_phase=increment_oscillator_phase,
            length=length,
            pulse=ry_pulse,
        )

    @dsl.quantum_operation
    def y90(
        self,
        q: BosonicQubit,
        n: int = 0,
        amplitude: float | None = None,
        phase: float = _PI_BY_2,
        increment_oscillator_phase: float | SweepParameter | None = None,
        length: float | None = None,
        pulse: dict | None = None,
    ) -> None:
        """Rotate the ancilla transmon by 90 degrees about the Y axis.

        Arguments:
            q:
                The qubit to rotate.
            n:
                The cavity photon number. Default: 0.
            amplitude:
                The amplitude of the rotation pulse. By default this
                is the pi/2 amplitude for photon number n.
            phase:
                The phase of the rotation pulse in radians.
                Default: pi/2.
            increment_oscillator_phase:
                Phase increment on the baseband oscillator. Default: None.
            length:
                The duration of the rotation pulse. Default from
                qubit parameters.
            pulse:
                A dictionary of overrides for the pulse parameters.
        """
        if amplitude is None:
            _, params = q.selective_transmon_Rx_parameters(n)
            amplitude = params["amplitude_pi2"]

        self.ry.omit_section(
            q,
            self._PI_BY_2,
            n=n,
            amplitude=amplitude,
            phase=phase,
            increment_oscillator_phase=increment_oscillator_phase,
            length=length,
            pulse=pulse,
        )

    @dsl.quantum_operation
    def y180(
        self,
        q: BosonicQubit,
        n: int = 0,
        amplitude: float | None = None,
        phase: float = _PI_BY_2,
        increment_oscillator_phase: float | SweepParameter | None = None,
        length: float | None = None,
        pulse: dict | None = None,
    ) -> None:
        """Rotate the ancilla transmon by 180 degrees about the Y axis.

        Arguments:
            q:
                The qubit to rotate.
            n:
                The cavity photon number. Default: 0.
            amplitude:
                The amplitude of the rotation pulse. By default this
                is the pi amplitude for photon number n.
            phase:
                The phase of the rotation pulse in radians.
                Default: pi/2.
            increment_oscillator_phase:
                Phase increment on the baseband oscillator. Default: None.
            length:
                The duration of the rotation pulse. Default from
                qubit parameters.
            pulse:
                A dictionary of overrides for the pulse parameters.
        """
        if amplitude is None:
            _, params = q.selective_transmon_Rx_parameters(n)
            amplitude = params["amplitude_pi"]

        self.ry.omit_section(
            q,
            self._PI,
            n=n,
            amplitude=amplitude,
            phase=phase,
            increment_oscillator_phase=increment_oscillator_phase,
            length=length,
            pulse=pulse,
        )

    @dsl.quantum_operation
    def rz(
        self,
        q: BosonicQubit,
        angle: float,
        n: int = 0,
    ) -> None:
        """Rotate the ancilla transmon by the given angle about the Z-axis.

        Arguments:
            q:
                The qubit to rotate.
            angle:
                The angle to rotate by in radians.
            n:
                The cavity photon number. Default: 0.
        """
        drive_line, _ = q.selective_transmon_Rx_parameters(n)

        dsl.play(
            signal=q.signals[drive_line],
            pulse=None,
            increment_oscillator_phase=angle,
        )

    @dsl.quantum_operation
    def z90(self, q: BosonicQubit, n: int = 0) -> None:
        """Rotate the ancilla transmon by 90 degrees about the Z-axis.

        Arguments:
            q:
                The qubit to rotate.
            n:
                The cavity photon number. Default: 0.
        """
        self.rz.omit_section(q, self._PI_BY_2, n=n)

    @dsl.quantum_operation
    def z180(self, q: BosonicQubit, n: int = 0) -> None:
        """Rotate the ancilla transmon by 180 degrees about the Z-axis.

        Arguments:
            q:
                The qubit to rotate.
            n:
                The cavity photon number. Default: 0.
        """
        self.rz.omit_section(q, self._PI, n=n)

    # -----------------------------------------------------------------
    # Memory cavity operations
    # -----------------------------------------------------------------

    @dsl.quantum_operation
    def displacement(
        self,
        q: BosonicQubit,
        amplitude: float | SweepParameter | None = None,
        phase: float | SweepParameter = 0.0,
        *,
        beta: float | SweepParameter | None = None,
        n_bar: float | None = None,
        length: float | SweepParameter | None = None,
        pulse: dict | None = None,
    ) -> None:
        """Play a displacement pulse D(beta) on the memory cavity.

        The displacement in phase space is:

            beta = amplitude * exp(i * phase)

        Exactly one of ``amplitude``, ``beta``, or ``n_bar`` must
        be provided:

        - ``amplitude``: raw hardware amplitude (no conversion).
        - ``beta``: displacement magnitude |beta|. Converted to
        hardware amplitude via the calibrated parameter
        ``displacement_amp_per_unit_beta``.
        - ``n_bar``: mean photon number of the target coherent
        state. Converted via |beta| = √n̄, then to hardware
        amplitude.

        Arguments:
            q:
                The qubit whose memory cavity to displace.
            amplitude:
                The displacement pulse amplitude in hardware units.
            phase:
                The displacement phase arg(beta) in radians.
                Default: 0.0.
            beta:
                The displacement magnitude |beta| in calibrated units.
                Requires ``displacement_amp_per_unit_beta`` to be
                set on the qubit parameters.
            n_bar:
                The target mean photon number of the coherent state.
                Requires ``displacement_amp_per_unit_beta`` to be
                set on the qubit parameters.
            length:
                Duration of the displacement pulse in seconds.
                Defaults to the qubit parameter
                ``memory_drive_length``.
            pulse:
                A dictionary of overrides for the displacement pulse
                parameters.

                The dictionary may contain sweep parameters for the
                pulse parameters other than ``function``.

                If the ``function`` parameter is different to the one
                specified for the qubit, then this override dictionary
                completely replaces the existing pulse parameters.

                Otherwise the values override or extend the existing
                ones.
        """
        # --- resolve amplitude from exactly one specification ---
        n_specified = sum(x is not None for x in [amplitude, beta, n_bar])
        if n_specified != 1:
            raise ValueError(
                "Exactly one of 'amplitude', 'beta', or 'n_bar' "
                "must be specified. "
                f"Got: amplitude={amplitude}, beta={beta}, n_bar={n_bar}."
            )

        if beta is not None:
            amp_per_beta = q.parameters.displacement_amp_per_unit_beta
            if amp_per_beta is None:
                raise ValueError(
                    f"Qubit {q.uid}: "
                    f"'displacement_amp_per_unit_beta' is not set. "
                    f"Run the displacement calibration first, or "
                    f"use 'amplitude' directly."
                )
            amplitude = beta * amp_per_beta

        elif n_bar is not None:
            amp_per_beta = q.parameters.displacement_amp_per_unit_beta
            if amp_per_beta is None:
                raise ValueError(
                    f"Qubit {q.uid}: "
                    f"'displacement_amp_per_unit_beta' is not set. "
                    f"Run the displacement calibration first, or "
                    f"use 'amplitude' directly."
                )
            amplitude = np.sqrt(n_bar) * amp_per_beta

        # --- rest is unchanged ---
        drive_line, params = q.memory_drive_parameters()

        if length is None:
            length = params["length"]

        disp_pulse = dsl.create_pulse(
            params["pulse"],
            pulse,
            name="displacement_pulse",
        )

        dsl.play(
            q.signals[drive_line],
            amplitude=amplitude,
            phase=phase,
            length=length,
            pulse=disp_pulse,
        )

    @dsl.quantum_operation
    def swap(
        self,
        q: BosonicQubit,
        length: float | None = None,
        amplitude: float | None = None,
        pulse: dict | None = None,
    ) -> None:
        """Apply the transmon ↔ memory SWAP via a DC flux pulse.

        Implements |e, n⟩ ↔ |g, n+1⟩ by tuning the transmon into
        resonance with the memory cavity for a duration ``length``.
        The pulse is emitted on the ``swap_drive`` LF line.

        If ``length`` or ``amplitude`` is None, the stored qubit
        parameter is used. Either (or both) can be a SweepParameter.
        """
        line, defaults = q.swap_parameters()

        if length is None:
            length = defaults["length"]
        if amplitude is None:
            amplitude = defaults["amplitude"]

        # Merge default envelope with any per-call overrides.
        pulse_dict = {**defaults["pulse"], **(pulse or {})}
        # The pulse dictionary requires a concrete (non-SweepParameter) length.
        # When length is a SweepParameter (near-time sweep), we use the stored
        # default as the envelope definition; dsl.play overrides the played length.
        envelope_length = (
            defaults["length"] if isinstance(length, SweepParameter) else length
        )
        pulse_dict.setdefault("length", envelope_length)

        swap_pulse = dsl.create_pulse(
            pulse_dict,
            name="swap_pulse",
        )

        dsl.play(
            signal=q.signals[line],
            pulse=swap_pulse,
            amplitude=amplitude,
            length=length,
        )

    @dsl.quantum_operation
    def snap(
        self,
        q: BosonicQubit,
        thetas: list[float],
        phase_tolerance: float = 1e-6,
    ) -> None:
        """Apply the SNAP (Selective Number-Arbitrary Phase) gate.

        Implements the unitary:

            S(θ⃗) = Σₙ exp(iθₙ) |n⟩⟨n|

        which applies an independent phase θₙ to each Fock state
        |n⟩ of the memory cavity.

        For each photon number n where θₙ ≠ 0, the gate is
        decomposed into two selective transmon π pulses at the
        number-dependent frequency f(n) = f₀ + n·χ:

            SNAP(θₙ) = Rx(π, n) · R_φ(π, n)

        where the drive phase of the second pulse is:

            φ = θₙ + snap_drive_phase_per_n[str(n)]

        The calibrated amplitudes and phase offsets are read from
        ``q.parameters.snap_drive_amp_per_n`` and
        ``q.parameters.snap_drive_phase_per_n``, populated by the
        SNAP calibration workflow.

        Fock states whose target phase is effectively zero
        (mod 2π, within ``phase_tolerance``) are skipped — no
        pulses are emitted for those states.

        Note:
            Pulses for different photon numbers are applied
            sequentially. Because each pair targets a distinct
            number-resolved transition frequency, the ordering
            does not affect the ideal gate action.

        Prerequisites:
            - SNAP amplitude calibration (Part 1) must have been
            run for all photon numbers present in ``thetas``.
            - SNAP phase calibration (Part 2) must have been run
            for all photon numbers present in ``thetas``.
            - The selective ``rx`` operation must accept ``n``,
            ``amplitude``, and ``phase`` keyword arguments.

        Arguments:
            q:
                The bosonic qubit to apply the SNAP gate on.
            thetas:
                List of phases (radians) indexed by Fock state
                number. ``thetas[n]`` is the phase applied to
                |n⟩. For example, ``[0.0, π, 0.0, π/2]`` applies
                phase π to |1⟩ and π/2 to |3⟩.
            phase_tolerance:
                Tolerance (radians) below which a phase is
                treated as zero and skipped. Default: 1e-6.

        Raises:
            ValueError:
                If ``snap_drive_amp_per_n`` or
                ``snap_drive_phase_per_n`` is missing a required
                photon number.

        Example::

            # Apply θ₀=0, θ₁=π, θ₂=0, θ₃=π/2
            qop.snap(q, thetas=[0.0, np.pi, 0.0, np.pi / 2])
        """
        for n, theta_n in enumerate(thetas):
            # Normalize to [0, 2π) and skip near-zero phases
            theta_mod = theta_n % (2 * np.pi)
            if theta_mod < phase_tolerance or (2 * np.pi - theta_mod) < phase_tolerance:
                continue

            if str(n) not in q.parameters.snap_drive_amp_per_n:
                raise ValueError(
                    f"Qubit {q.uid!r}: SNAP drive amplitude for photon number "
                    f"n={n} is not calibrated. Run snap_calibration (Part 1) to "
                    "populate snap_drive_amp_per_n."
                )
            if str(n) not in q.parameters.snap_drive_phase_per_n:
                raise ValueError(
                    f"Qubit {q.uid!r}: SNAP drive phase for photon number "
                    f"n={n} is not calibrated. Run snap_calibration (Part 2) to "
                    "populate snap_drive_phase_per_n."
                )
            amp_n = q.parameters.snap_drive_amp_per_n[str(n)]
            phase_offset_n = q.parameters.snap_drive_phase_per_n[str(n)]

            # First selective π pulse (reference phase = 0)
            self.rx.omit_section(
                q,
                angle=None,
                n=n,
                amplitude=amp_n,
            )

            # Second selective π pulse (encodes θₙ via drive phase)
            self.rx.omit_section(
                q,
                angle=None,
                n=n,
                amplitude=-amp_n,
                phase=theta_n + phase_offset_n,
            )

    # -----------------------------------------------------------------
    # Parity mapping
    # -----------------------------------------------------------------

    @dsl.quantum_operation
    def parity_mapping(
        self,
        q: BosonicQubit,
    ) -> None:
        """Map the memory cavity photon-number parity onto the transmon.

        Implements a Ramsey interferometry sequence that exploits the
        dispersive interaction between the memory cavity and the
        ancilla transmon:

            Ry(+pi/2) → wait(1 / (2|χ|)) → Ry(-pi/2)

        During the dispersive wait time t = 1 / (2|χ|), the transmon
        in |e⟩ acquires a photon-number-dependent phase:

            φ(n) = 2pi * chi * n * t = pi * n

        The closing Ry(-pi/2) pulse converts this phase into a
        population difference:

            Even photon number (⟨pi⟩ = +1) → transmon in |g⟩
            Odd  photon number (⟨pi⟩ = -1) → transmon in |e⟩

        The transmon state can then be read out using the standard
        ``measure`` operation to obtain the parity expectation value:

            ⟨pi⟩ = P(|g⟩) - P(|e⟩)

        Prerequisites:
            - The dispersive shift χ must be calibrated
            (qubit parameter ``chi``).
            - The ancilla transmon must be in |g⟩ before this
            operation is called.

        Arguments:
            q:
                The qubit whose memory cavity parity to map onto
                the transmon state.
        """
        if q.parameters.chi == 0.0:
            raise ValueError(
                f"Qubit {q.uid!r}: parity_mapping requires a non-zero dispersive "
                "shift chi. Calibrate chi before using this operation."
            )
        t_parity = 1.0 / (2.0 * abs(q.parameters.chi))

        # Ry(+pi/2): create transmon superposition (|g⟩ + |e⟩)/√2
        self.y90.omit_section(q)

        # Dispersive wait: cavity parity → transmon phase
        self.delay.omit_section(q, time=t_parity)

        # Ry(-pi/2): convert phase to population
        self.ry.omit_section(q, angle=-self._PI_BY_2)

    # -----------------------------------------------------------------
    # Wigner function measurement at a single phase-space point
    # -----------------------------------------------------------------

    @dsl.quantum_operation(broadcast=False)
    def wigner_point(
        self,
        q: BosonicQubit,
        beta_amplitude: float | SweepParameter,
        beta_phase: float | SweepParameter,
        handle: str,
        displacement_length: float | SweepParameter | None = None,
        displacement_pulse: dict | None = None,
        readout_pulse: dict | None = None,
        kernel_pulses: list[dict]
        | Literal["default", "continuous", "optimal"]
        | None = None,
    ) -> None:
        """Measure the Wigner function at a single phase-space point.

        Applies D(-beta) to the cavity, performs parity mapping, then measures
        the transmon. The result encodes W(beta) = (2/pi) * <parity>.

        Arguments:
            q:
                The bosonic qubit to measure.
            beta_amplitude:
                Amplitude of the displacement pulse (|beta|).
            beta_phase:
                Phase of the displacement in radians (arg(beta)).
            handle:
                Acquisition handle for the readout result.
            displacement_length:
                Duration of the displacement pulse. Defaults to the
                qubit parameter ``memory_drive_length``.
            displacement_pulse:
                Dictionary of pulse parameter overrides for the displacement.
            readout_pulse:
                Dictionary of pulse parameter overrides for the readout.
            kernel_pulses:
                Integration kernel pulses. See ``measure`` for valid values.
        """
        # D(-beta): displace cavity by -beta.
        # -beta has the same magnitude but opposite phase:
        #   arg(-beta) = arg(beta) + pi
        self.displacement(
            q,
            beta=beta_amplitude,
            phase=beta_phase + self._PI,
            length=displacement_length,
            pulse=displacement_pulse,
        )

        # Parity mapping: Ry(pi/2) → wait(1/(2|chi|)) → Ry(-pi/2)
        self.parity_mapping(q)

        # Measure ancilla transmon
        self.measure(
            q,
            handle=handle,
            readout_pulse=readout_pulse,
            kernel_pulses=kernel_pulses,
        )

    # -----------------------------------------------------------------
    # Spectroscopy operations
    # -----------------------------------------------------------------

    @dsl.quantum_operation
    def transmon_spectroscopy_drive(
        self,
        q: BosonicQubit,
        amplitude: float | SweepParameter | None = None,
        phase: float = 0.0,
        length: float | SweepParameter | None = None,
        pulse: dict | None = None,
    ) -> None:
        """Long pulse for ancilla transmon spectroscopy.

        Arguments:
            q:
                The qubit to apply the spectroscopy drive on.
            amplitude:
                The amplitude of the pulse. By default, the
                qubit parameter "transmon_spectroscopy_amplitude".
            phase:
                The phase of the pulse in radians. By default,
                this is 0.0.
            length:
                The duration of the pulse. By default, this
                is the qubit parameter "transmon_spectroscopy_length".
            pulse:
                A dictionary of overrides for the spectroscopy pulse
                parameters.

                The dictionary may contain sweep parameters for the pulse
                parameters other than `function`.

                If the `function` parameter is different to the one
                specified for the qubit, then this override dictionary
                completely replaces the existing pulse parameters.

                Otherwise, the values override or extend the existing ones.
        """
        spec_line, params = q.transmon_spectroscopy_parameters()
        if amplitude is None:
            amplitude = params["amplitude"]
        if length is None:
            length = params["length"]

        spectroscopy_pulse = dsl.create_pulse(
            params["pulse"], pulse, name="transmon_spectroscopy_pulse"
        )

        dsl.play(
            q.signals[spec_line],
            amplitude=amplitude,
            phase=phase,
            length=length,
            pulse=spectroscopy_pulse,
        )

    @dsl.quantum_operation
    def memory_spectroscopy_drive(
        self,
        q: BosonicQubit,
        amplitude: float | SweepParameter | None = None,
        phase: float = 0.0,
        length: float | SweepParameter | None = None,
        pulse: dict | None = None,
    ) -> None:
        """Long pulse for memory cavity spectroscopy.

        Arguments:
            q:
                The qubit to apply the memory spectroscopy drive on.
            amplitude:
                The amplitude of the pulse. By default, the
                qubit parameter "memory_spectroscopy_amplitude".
            phase:
                The phase of the pulse in radians. By default,
                this is 0.0.
            length:
                The duration of the pulse. By default, this
                is the qubit parameter "memory_spectroscopy_length".
            pulse:
                A dictionary of overrides for the spectroscopy pulse
                parameters.

                The dictionary may contain sweep parameters for the pulse
                parameters other than `function`.

                If the `function` parameter is different to the one
                specified for the qubit, then this override dictionary
                completely replaces the existing pulse parameters.

                Otherwise, the values override or extend the existing ones.
        """
        spec_line, params = q.memory_spectroscopy_parameters()
        if amplitude is None:
            amplitude = params["amplitude"]
        if length is None:
            length = params["length"]

        spectroscopy_pulse = dsl.create_pulse(
            params["pulse"], pulse, name="memory_spectroscopy_pulse"
        )

        dsl.play(
            q.signals[spec_line],
            amplitude=amplitude,
            phase=phase,
            length=length,
            pulse=spectroscopy_pulse,
        )

    # -----------------------------------------------------------------
    # Composite operations
    # -----------------------------------------------------------------

    @dsl.quantum_operation(broadcast=False)
    def ramsey(
        self,
        q: BosonicQubit,
        delay: SweepParameter | float,
        ramsey_phase: SweepParameter | float,
        echo_pulse: Literal["x180", "y180"] | None = None,
    ) -> None:
        """Performs a Ramsey operation on the ancilla transmon.

        This operation consists of the following steps:
        x90 - delay/2 - [x180] or [y180] - delay/2 - x90

        The transmon drive parameters used are calibrated for the
        memory cavity in |0⟩.

        Arguments:
            q:
                The qubit to rotate.
            delay:
                The duration between two rotations, excluding the
                echo pulse length if an echo pulse is included.
            ramsey_phase:
                The phase of the second x90 rotation,
                this will be applied as a phase increment for the second
                pulse.
            echo_pulse:
                The echo pulse to include. One of "x180", "y180", or None.

        Raises:
            ValueError:
                If the echo pulse is not None, "x180" or "y180".
        """
        if echo_pulse is not None and echo_pulse not in ("x180", "y180"):
            raise ValueError(
                f"Support only x180 or y180 for echo pulse, not {echo_pulse}"
            )

        with dsl.section(
            name=f"ramsey_{q.uid}",
            alignment=SectionAlignment.RIGHT,
        ):
            sec_x90_1 = self.x90(q)
            sec_x90_1.alignment = SectionAlignment.RIGHT
            if echo_pulse is not None:
                self.delay(q, time=delay / 2)
                sec_echo = self[echo_pulse](q)
                sec_echo.alignment = SectionAlignment.RIGHT
                self.delay(q, time=delay / 2)
            else:
                self.delay(q, time=delay)
            sec_x90_2 = self.x90(q, increment_oscillator_phase=ramsey_phase)
            sec_x90_2.alignment = SectionAlignment.RIGHT

    @dsl.quantum_operation(broadcast=False)
    def active_reset(
        self,
        qubits: QuantumElements,
        number_resets: int = 1,
        feedback_processing_delay: float = 0.0,
        handles: Sequence[str] | None = None,
        measure_section_length: float | None = None,
    ) -> None:
        """Reset the ancilla transmon into the ground state using active reset.

        Arguments:
            qubits:
                The qubits to reset.
            number_resets:
                The number of active reset rounds to apply.
            feedback_processing_delay:
                Feedback processing time.
                Default: 0.0
            handles:
                The handles to store the active-reset acquisition results in
                for each qubit.
            measure_section_length:
                The length of the measure section. If multiple qubits are
                passed, the measure section must have the same length for
                each qubit. Default: None.
        """
        if isinstance(qubits, BosonicQubit):
            qubits = [qubits]

        if handles is None:
            handles = [dsl.handles.active_reset_handle(q.uid) for q in qubits]
        if len(handles) != len(qubits):
            raise ValueError(
                f"Please provide a handle for each qubit. Currently, there are "
                f"{len(qubits)} qubits and {len(handles)} handles."
            )

        for nr in range(number_resets):
            with dsl.section(name=f"active_reset_rep_{nr}"):
                for qidx, q in enumerate(qubits):
                    sec = self.measure(q, handle=handles[qidx])
                    sec.length = measure_section_length
                    self.delay(q, feedback_processing_delay)
                    with dsl.match(name=f"match_{q.uid}", handle=handles[qidx]):
                        with dsl.case(name=f"case_{q.uid}_g", state=0):
                            pass
                        with dsl.case(name=f"case_{q.uid}_e", state=1):
                            self.x180.omit_section(q)

    @dsl.quantum_operation(broadcast=False)
    def calibration_traces(
        self,
        qubits: QuantumElements,
        states: str | tuple = "ge",
        active_reset: bool = False,  # noqa: FBT001, FBT002
        active_reset_repetitions: int = 1,
        feedback_processing_delay: float = 0.0,
        measure_section_length: float | None = None,
    ) -> None:
        """Add calibration-trace measurements.

        Arguments:
            qubits:
                The qubits to measure calibration traces for.
            states:
                The calibration states to prepare. Can be any combination of
                ("g", "e"). The same states are prepared for each qubit.
                Default: "ge"
            active_reset:
                Whether to use active reset to prepare the qubit in g
                before every calibration state preparation.
            active_reset_repetitions:
                The number of active reset rounds to apply.
            feedback_processing_delay:
                Feedback processing time.
                Default: 0.0
            measure_section_length:
                The length of the measure section. If multiple qubits are
                passed, the measure section must have the same length for
                each qubit. Default: None.
        """
        if isinstance(qubits, BosonicQubit):
            qubits = [qubits]
        for state in states:
            if active_reset:
                active_reset_handles = [
                    dsl.handles.active_reset_calibration_trace_handle(q.uid, state)
                    for q in qubits
                ]
                self.active_reset(
                    qubits,
                    number_resets=active_reset_repetitions,
                    feedback_processing_delay=feedback_processing_delay,
                    handles=active_reset_handles,
                    measure_section_length=measure_section_length,
                )

            with dsl.section(
                name=f"cal_{state}",
                alignment=SectionAlignment.RIGHT,
            ):
                with dsl.section(
                    name=f"cal_prep_{state}", alignment=SectionAlignment.RIGHT
                ):
                    for q in qubits:
                        self.prepare_state.omit_section(q, state=state)
                with dsl.section(
                    name=f"cal_measure_{state}", alignment=SectionAlignment.LEFT
                ):
                    for q in qubits:
                        sec = self.measure(
                            q, dsl.handles.calibration_trace_handle(q.uid, state)
                        )
                        sec.length = measure_section_length
                        self.passive_reset(q)

    @staticmethod
    def measure_section_length(
        qubits: BosonicQubit | Sequence[BosonicQubit],
    ) -> float:
        """The length of the measure section.

        The length returned is the maximum, over all supplied qubits, of the
        larger of the readout pulse and readout kernel lengths.

        Arguments:
            qubits:
                The qubits to consider for the section length determination.

        Returns:
            The length of a measure section that involves the supplied qubits.
        """
        if not isinstance(qubits, Sequence):
            qubits = [qubits]

        return max(
            [q.parameters.readout_integration_length for q in qubits]
            + [q.parameters.readout_length for q in qubits]
        )
