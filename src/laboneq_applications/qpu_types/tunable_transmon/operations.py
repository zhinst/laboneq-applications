# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tunable transmon operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from laboneq.dsl.calibration import Oscillator
from laboneq.dsl.enums import ModulationType
from laboneq.simple import SectionAlignment, dsl

from .qubit_types import TunableTransmonQubit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.dsl.calibration import Calibration
    from laboneq.dsl.parameter import SweepParameter

    from laboneq_applications.typing import Qubits

# TODO: Implement multistate 0-1-2 measurement operation

# TODO: Add rotate_xy gate that performs a rotation about an axis in the xy-plane.


class TunableTransmonOperations(dsl.QuantumOperations):
    """Operations for TunableTransmonQubits."""

    QUBIT_TYPES = TunableTransmonQubit

    # common angles used by rx, ry and rz.
    _PI = np.pi
    _PI_BY_2 = np.pi / 2

    @dsl.quantum_operation
    def barrier(self, q: TunableTransmonQubit) -> None:
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
    def delay(self, q: TunableTransmonQubit, time: float) -> None:
        """Add a delay on the qubit drive signal.

        Arguments:
            q:
                The qubit to delay on.
            time:
                The duration of the delay in seconds.
        """
        # Delaying on a single line is sufficient since the operation
        # section automatically reserves all lines.
        dsl.delay(q.signals["drive"], time=time)

    @dsl.quantum_operation
    def set_frequency(
        self,
        q: TunableTransmonQubit,
        frequency: float | SweepParameter,
        *,
        transition: str | None = None,
        readout: bool = False,
        rf: bool = True,
        calibration: Calibration | None = None,
    ) -> None:
        """Sets the frequency of the given qubit drive line or readout line.

        Arguments:
            q:
                The qubit to set the transition or readout frequency of.
            frequency:
                The frequency to set in Hz.
                By default the frequency specified is the RF frequency.
                The oscillator frequency may be set directly instead
                by passing `rf=False`.
            transition:
                The transition to rotate. By default this is "ge"
                (i.e. the 0-1 transition).
            readout:
                If true, the frequency of the readout line is set
                instead. Setting the readout frequency to a sweep parameter
                is only supported in spectroscopy mode. The LabOne Q compiler
                will raise an error in other modes.
            rf:
                If True, set the RF frequency of the transition.
                If False, set the oscillator frequency directly instead.
                The default is to set the RF frequency.
            calibration:
                The experiment calibration to update (see the note below).
                By default, the calibration from the currently active
                experiment context is used. If no experiment context is
                active, for example when using
                `@qubit_experiment(context=False)`, the calibration
                object may be passed explicitly.

        Raises:
            RuntimeError:
                If there is an attempt to call `set_frequency` more than
                once on the same signal. See notes below for details.

        Notes:
            Currently `set_frequency` is implemented by setting the
            appropriate oscillator frequencies in the experiment calibration.
            This has two important consequences:

            * Each experiment may only set one frequency per signal line,
              although this may be a parameter sweep.

            * The set frequency or sweep applies for the whole experiment
              regardless of where in the experiment the frequency is set.

            This will be improved in a future release.
        """
        if readout:
            signal_line = "measure"
            lo_frequency = q.parameters.readout_lo_frequency
        else:
            signal_line, _ = q.transition_parameters(transition)
            lo_frequency = q.parameters.drive_lo_frequency

        if rf:
            # This subtraction works for both numbers and SweepParameters
            frequency -= lo_frequency

        if calibration is None:
            calibration = dsl.experiment_calibration()
        signal_calibration = calibration[q.signals[signal_line]]
        oscillator = signal_calibration.oscillator

        if oscillator is None:
            oscillator = signal_calibration.oscillator = Oscillator(frequency=frequency)
        if getattr(oscillator, "_set_frequency", False):
            # We mark the oscillator with a _set_frequency attribute to ensure that
            # set_frequency isn't performed on the same oscillator twice. Ideally
            # LabOne Q would provide a set_frequency DSL method that removes the
            # need for setting the frequency on the experiment calibration.
            raise RuntimeError(
                f"Frequency of qubit {q.uid} {signal_line} line was set multiple times"
                f" using the set_frequency operation.",
            )

        oscillator._set_frequency = True
        oscillator.frequency = frequency
        if signal_line == "measure":
            # LabOne Q does not support software modulation of measurement
            # signal sweeps because it results in multiple readout waveforms
            # on the same readout signal. Ideally the LabOne Q compiler would
            # sort this out for us when the modulation type is AUTO, but currently
            # it does not.
            oscillator.modulation_type = ModulationType.HARDWARE

    @dsl.quantum_operation
    def set_readout_amplitude(
        self,
        q: TunableTransmonQubit,
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
                active, for example when using
                `@qubit_experiment(context=False)`, the calibration
                object may be passed explicitly.

        Raises:
            RuntimeError:
                If there is an attempt to call `set_readout_amplitude` more than
                once on the same signal. See notes below for details.

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
        signal_calibration = calibration[q.signals["measure"]]

        if getattr(calibration, "_set_readout_amplitude", False):
            # We mark the oscillator with a _set_readout_amplitude attribute to ensure
            # that set_readout_amplitude isn't performed on the same signal twice.
            # Ideally LabOne Q DSL provide a more direct method that removes the
            # need for setting amplitude on the experiment calibration.
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
        q: TunableTransmonQubit,
        handle: str,
        readout_pulse: dict | None = None,
        kernel_pulses: list[dict] | Literal["default"] | None = None,
    ) -> None:
        """Perform a measurement on the qubit.

        The measurement operation plays a readout pulse and performs an acquisition.
        If you wish to perform only an acquisition, use the `acquire` operation.

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
                A list of dictionaries describing the pulse parameters for
                the integration kernels, or "default" or `None`.

                If a list of dictionaries is past, each dictionary must
                completely specify a kernel pulse and its parameters (i.e.
                they must include the `function` parameter and all of its
                arguments).

                If the string "default" is passed, a constant integration
                kernel of length equal to the qubit's
                `readout_integration_length` parameter is used.

                If not specified or `None`, the kernels specified in the
                qubit's `readout_integration_kernels` are used.
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
        q: TunableTransmonQubit,
        handle: str,
        kernel_pulses: list[dict] | Literal["default"] | None = None,
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
                A list of dictionaries describing the pulse parameters for
                the integration kernels, or "default" or `None`.

                If a list of dictionaries is past, each dictionary must
                completely specify a kernel pulse and its parameters (i.e.
                they must include the `function` parameter and all of its
                arguments).

                If the string "default" is passed, a constant integration
                kernel of length equal to the qubit's
                `readout_integration_length` parameter is used.

                If not specified or `None`, the kernels specified in the
                qubit's `readout_integration_kernels` are used.
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
        q: TunableTransmonQubit,
        state: str = "g",
        reset: Literal["active", "passive"] | None = None,
    ) -> None:
        """Prepare a qubit in the given state.

        The qubit is assumed to be in the ground state, 'g'. If this is not
        the case pass `reset="passive"` or `reset="active"` to perform a
        passive or active reset operation before preparing the state.

        The returned section is right-aligned to ensure that there is not time gap
        between the end of the preparation pulses and the end of the section.

        Arguments:
            q:
                The qubit to prepare.
            state:
                The state to prepapre. One of 'g', 'e' or 'f'.
            reset:
                If not None, perform the specified reset operation before
                preparing the state.
        """
        if reset is None:
            pass
        elif reset == "passive":
            self.passive_reset(q)
        elif reset == "active":
            raise ValueError("The active reset operation is not yet implemented.")
        else:
            raise ValueError(
                f"The reset parameter to prepare_state must be 'active',"
                f" 'passive', or None, not: {reset!r}",
            )

        if state == "g":
            pass
        elif state == "e":
            sec = self.x180(q, transition="ge")
            sec.alignment = SectionAlignment.RIGHT
        elif state == "f":
            sec = self.x180(q, transition="ge")
            sec.alignment = SectionAlignment.RIGHT
            sec = self.x180(q, transition="ef")
            sec.alignment = SectionAlignment.RIGHT
        else:
            raise ValueError(f"Only states g, e and f can be prepared, not {state!r}")

    @dsl.quantum_operation
    def passive_reset(
        self,
        q: TunableTransmonQubit,
        delay: float | SweepParameter | None = None,
    ) -> None:
        """Reset a qubit into the ground state, 'g', using a long delay.

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

    @dsl.quantum_operation
    def rx(
        self,
        q: TunableTransmonQubit,
        angle: float | SweepParameter | None,
        transition: str | None = None,
        amplitude: float | SweepParameter | None = None,
        phase: float = 0.0,
        length: float | SweepParameter | None = None,
        pulse: dict | None = None,
    ) -> None:
        """Rotate the qubit by the given angle (in radians) about the X axis.

        Arguments:
            q:
                The qubit to rotate.
            angle:
                The angle to rotate by in radians.
            transition:
                The transition to rotate. By default this is "ge"
                (i.e. the 0-1 transition).
            amplitude:
                The amplitude of the rotation pulse. By default this
                is determined by the angle and the π pulse amplitude
                qubit parameter "amplitude_pi" by linear interpolation.
            phase:
                The phase of the rotation pulse in radians. By default
                this is 0.0.
            length:
                The duration of the rotation pulse. By default this
                is determined by the qubit parameters.
            pulse:
                A dictionary of overrides for the qubit pulse parameters.

                The dictionary may contain sweep parameters for the pulse
                parameters other than `function`.

                If the `function` parameter is different to the one
                specified for the qubit, then this override dictionary
                completely replaces the existing pulse parameters.

                Otherwise the values override or extend the existing ones.
        """
        drive_line, params = q.transition_parameters(transition)

        if drive_line == "drive_ef":
            section = dsl.active_section()
            section.on_system_grid = True

        if amplitude is None:
            # always do a linear scaling with respect to the pi pulse amplitude from the
            # qubit
            amplitude = (angle / self._PI) * params["amplitude_pi"]
        if length is None:
            length = params["length"]

        rx_pulse = dsl.create_pulse(params["pulse"], pulse, name="rx_pulse")

        dsl.play(
            q.signals[drive_line],
            amplitude=amplitude,
            phase=phase,
            length=length,
            pulse=rx_pulse,
        )

    @dsl.quantum_operation
    def x90(
        self,
        q: TunableTransmonQubit,
        transition: str | None = None,
        amplitude: float | None = None,
        phase: float = 0.0,
        length: float | None = None,
        pulse: dict | None = None,
    ) -> None:
        """Rotate the qubit by 90 degrees about the X axis.

        This implementation calls `rx(q, π / 2, ...)`.

        Arguments:
            q:
                The qubit to rotate.
            transition:
                The transition to rotate. By default this is "ge"
                (i.e. the 0-1 transition).
            amplitude:
                The amplitude of the rotation pulse. By default this
                is determined from the qubit parameter "amplitude_pi2".
            phase:
                The phase of the rotation pulse in radians. By default
                this is 0.0.
            length:
                The duration of the rotation pulse. By default this
                is determined by the qubit parameters.
            pulse:
                A dictionary of overrides for the qubit pulse parameters.

                The dictionary may contain sweep parameters for the pulse
                parameters other than `function`.

                If the `function` parameter is different to the one
                specified for the qubit, then this override dictionary
                completely replaces the existing pulse parameters.

                Otherwise the values override or extend the existing ones.
        """
        if amplitude is None:
            _, params = q.transition_parameters(transition)
            amplitude = params["amplitude_pi2"]

        self.rx.omit_section(
            q,
            self._PI_BY_2,
            transition=transition,
            amplitude=amplitude,
            phase=phase,
            length=length,
            pulse=pulse,
        )

    @dsl.quantum_operation
    def x180(
        self,
        q: TunableTransmonQubit,
        transition: str | None = None,
        amplitude: float | None = None,
        phase: float = 0.0,
        length: float | None = None,
        pulse: dict | None = None,
    ) -> None:
        """Rotate the qubit by 180 degrees about the X axis.

        This implementation calls `rx(q, π, ...)`.

        Arguments:
            q:
                The qubit to rotate.
            transition:
                The transition to rotate. By default this is "ge"
                (i.e. the 0-1 transition).
            amplitude:
                The amplitude of the rotation pulse. By default this
                is determined from the qubit parameter "amplitude_pi".
            phase:
                The phase of the rotation pulse in radians. By default
                this is 0.0.
            length:
                The duration of the rotation pulse. By default this
                is determined by the qubit parameters.
            pulse:
                A dictionary of overrides for the qubit pulse parameters.

                The dictionary may contain sweep parameters for the pulse
                parameters other than `function`.

                If the `function` parameter is different to the one
                specified for the qubit, then this override dictionary
                completely replaces the existing pulse parameters.

                Otherwise the values override or extend the existing ones.
        """
        if amplitude is None:
            _, params = q.transition_parameters(transition)
            amplitude = params["amplitude_pi"]

        self.rx.omit_section(
            q,
            self._PI,
            transition=transition,
            amplitude=amplitude,
            phase=phase,
            length=length,
            pulse=pulse,
        )

    @dsl.quantum_operation
    def ry(
        self,
        q: TunableTransmonQubit,
        angle: float | SweepParameter | None,
        transition: str | None = None,
        amplitude: float | SweepParameter | None = None,
        phase: float = _PI_BY_2,
        length: float | SweepParameter | None = None,
        pulse: dict | None = None,
    ) -> None:
        """Rotate the qubit by the given angle (in radians) about the Y axis.

        Arguments:
            q:
                The qubit to rotate.
            angle:
                The angle to rotate by in radians.
            transition:
                The transition to rotate. By default this is "ge"
                (i.e. the 0-1 transition).
            amplitude:
                The amplitude of the rotation pulse. By default this
                is determined by the angle and the π pulse amplitude
                qubit parameter "amplitude_pi" by linear interpolation.
            phase:
                The phase of the rotation pulse in radians. By default
                this is `π / 2`.
            length:
                The duration of the rotation pulse. By default this
                is determined by the qubit parameters.
            pulse:
                A dictionary of overrides for the qubit pulse parameters.

                The dictionary may contain sweep parameters for the pulse
                parameters other than `function`.

                If the `function` parameter is different to the one
                specified for the qubit, then this override dictionary
                completely replaces the existing pulse parameters.

                Otherwise the values override or extend the existing ones.
        """
        drive_line, params = q.transition_parameters(transition)

        if drive_line == "drive_ef":
            section = dsl.active_section()
            section.on_system_grid = True

        if amplitude is None:
            # always do a linear scaling with respect to the pi pulse amplitude from the
            # qubit
            amplitude = (angle / self._PI) * params["amplitude_pi"]
        if length is None:
            length = params["length"]

        ry_pulse = dsl.create_pulse(params["pulse"], pulse, name="ry_pulse")

        dsl.play(
            q.signals[drive_line],
            amplitude=amplitude,
            phase=phase,
            length=length,
            pulse=ry_pulse,
        )

    @dsl.quantum_operation
    def y90(
        self,
        q: TunableTransmonQubit,
        transition: str | None = None,
        amplitude: float | None = None,
        phase: float = _PI_BY_2,
        length: float | None = None,
        pulse: dict | None = None,
    ) -> None:
        """Rotate the qubit by 90 degrees about the Y axis.

        This implementation calls `ry(q, π / 2, ...)`.

        Arguments:
            q:
                The qubit to rotate.
            transition:
                The transition to rotate. By default this is "ge"
                (i.e. the 0-1 transition).
            amplitude:
                The amplitude of the rotation pulse. By default this
                is determined from the qubit parameter "amplitude_pi2".
            phase:
                The phase of the rotation pulse in radians. By default
                this is `π / 2`.
            length:
                The duration of the rotation pulse. By default this
                is determined by the qubit parameters.
            pulse:
                A dictionary of overrides for the qubit pulse parameters.

                The dictionary may contain sweep parameters for the pulse
                parameters other than `function`.

                If the `function` parameter is different to the one
                specified for the qubit, then this override dictionary
                completely replaces the existing pulse parameters.

                Otherwise the values override or extend the existing ones.
        """
        if amplitude is None:
            _, params = q.transition_parameters(transition)
            amplitude = params["amplitude_pi2"]

        self.ry.omit_section(
            q,
            self._PI_BY_2,
            transition=transition,
            amplitude=amplitude,
            phase=phase,
            length=length,
            pulse=pulse,
        )

    @dsl.quantum_operation
    def y180(
        self,
        q: TunableTransmonQubit,
        transition: str | None = None,
        amplitude: float | None = None,
        phase: float = _PI_BY_2,
        length: float | None = None,
        pulse: dict | None = None,
    ) -> None:
        """Rotate the qubit by 180 degrees about the Y axis.

        This implementation calls `ry(q, π, ...)`.

        Arguments:
            q:
                The qubit to rotate.
            transition:
                The transition to rotate. By default this is "ge"
                (i.e. the 0-1 transition).
            amplitude:
                The amplitude of the rotation pulse. By default this
                is determined from the qubit parameter "amplitude_pi".
            phase:
                The phase of the rotation pulse in radians. By default
                this is `π / 2`.
            length:
                The duration of the rotation pulse. By default this
                is determined by the qubit parameters.
            pulse:
                A dictionary of overrides for the qubit pulse parameters.

                The dictionary may contain sweep parameters for the pulse
                parameters other than `function`.

                If the `function` parameter is different to the one
                specified for the qubit, then this override dictionary
                completely replaces the existing pulse parameters.

                Otherwise the values override or extend the existing ones.
        """
        if amplitude is None:
            _, params = q.transition_parameters(transition)
            amplitude = params["amplitude_pi"]

        self.ry.omit_section(
            q,
            self._PI,
            transition=transition,
            amplitude=amplitude,
            phase=phase,
            length=length,
            pulse=pulse,
        )

    @dsl.quantum_operation
    def rz(
        self,
        q: TunableTransmonQubit,
        angle: float,
        transition: str | None = None,
    ) -> None:
        """Rotate the qubit by the given angle (in radians) about the Z-axis.

        Arguments:
            q:
                The qubit to rotate.
            angle:
                The angle to rotate by in radians.
            transition:
                The transition to rotate. By default this is "ge"
                (i.e. the 0-1 transition).
        """
        drive_line, _params = q.transition_parameters(transition)

        dsl.play(
            signal=q.signals[drive_line],
            pulse=None,
            increment_oscillator_phase=angle,
        )

    @dsl.quantum_operation
    def z90(self, q: TunableTransmonQubit, transition: str | None = None) -> None:
        """Rotate the qubit by 90 degrees about the Z-axis.

        This implementation calls `rz(q, π / 2, ...)`.

        Arguments:
            q:
                The qubit to rotate.
            transition:
                The transition to rotate. By default this is "ge"
                (i.e. the 0-1 transition).
        """
        self.rz.omit_section(q, self._PI_BY_2, transition=transition)

    @dsl.quantum_operation
    def z180(self, q: TunableTransmonQubit, transition: str | None = None) -> None:
        """Rotate the qubit by 180 degrees about the Z-axis.

        This implementation calls `rz(q, π, ...)`.

        Arguments:
            q:
                The qubit to rotate.
            transition:
                The transition to rotate. By default this is "ge"
                (i.e. the 0-1 transition).
        """
        self.rz.omit_section(q, self._PI, transition=transition)

    @dsl.quantum_operation
    def spectroscopy_drive(
        self,
        q: TunableTransmonQubit,
        amplitude: float | SweepParameter | None = None,
        phase: float = 0.0,
    ) -> None:
        """Long pulse used for qubit spectroscopy that emulates a coherent field.

        Arguments:
            q:
                The qubit to apply the spectroscopy drive.
            amplitude:
                The amplitude of the pulse. By default the
                qubit parameter "spectroscopy_amplitude".
            phase:
                The phase of the pulse in radians. By default
                this is 0.0.
        """
        drive_line, _ = q.transition_parameters("ge")
        if amplitude is None:
            amplitude = q.parameters.spectroscopy_amplitude
        spectroscopy_drive = dsl.create_pulse(
            {
                "function": "const",
                "can_compress": True,
            },
            name="coherent_drive",
        )
        dsl.play(
            q.signals[drive_line],
            amplitude=amplitude,
            phase=phase,
            length=q.parameters.spectroscopy_length,
            pulse=spectroscopy_drive,
        )

    @dsl.quantum_operation(broadcast=False)
    def ramsey(
        self,
        q: TunableTransmonQubit,
        delay: float,
        phase: float,
        echo_pulse: Literal["x180", "y180"] | None = None,
        transition: str | None = None,
    ) -> None:
        """Performs a Ramsey operation on a qubit.

        This operation consists of the following steps:
        x90 - delay/2 - [x180] or [y180] - delay/2 - x90

        Arguments:
            q:
                The qubit to rotate
            delay:
                The duration between two rotations, excluding the
                echo pulse length if an echo pulse is included.
            phase:
                The phase of the second rotation
            echo_pulse:
                The echo pulse to include.
            transition:
                The transition to rotate. By default this is "ge"
                (i.e. the 0-1 transition).

        Raise:
            ValueError:
                If the transition is not "ge" nor "ef".

            ValueError:
                If the echo pulse is not None and not x180 or y180.
        """
        transition = "ge" if transition is None else transition
        if transition == "ef":
            on_system_grid = True
        elif transition == "ge":
            on_system_grid = False
        else:
            raise ValueError(f"Support only ge or ef transitions, not {transition!r}")

        if echo_pulse is not None and echo_pulse not in ("x180", "y180"):
            raise ValueError(
                f"Support only x180 or y180 for echo pulse, not {echo_pulse}"
            )

        with dsl.section(
            name=f"ramsey_{q.uid}",
            on_system_grid=on_system_grid,
            alignment=SectionAlignment.RIGHT,
        ):
            sec_x90_1 = self.x90(q, transition=transition)
            sec_x90_1.alignment = SectionAlignment.RIGHT
            if echo_pulse is not None:
                self.delay(q, time=delay / 2)
                sec_echo = self[echo_pulse](q, transition=transition)
                sec_echo.alignment = SectionAlignment.RIGHT
                self.delay(q, time=delay / 2)
            else:
                self.delay(q, time=delay)
            sec_x90_2 = self.x90(q, phase=phase, transition=transition)
            sec_x90_2.alignment = SectionAlignment.RIGHT

        # to remove the gap due to oscillator switching for driving ef transitions.
        if echo_pulse is not None:
            sec_echo.on_system_grid = False
        sec_x90_1.on_system_grid = False
        sec_x90_2.on_system_grid = False

    @dsl.quantum_operation
    def x180_ef_reset(
        self,
        q: TunableTransmonQubit,
    ) -> None:
        """Modulated x180 operation on the ef transition, used for active reset.

        Arguments:
            q:
                The qubit to reset.
        """
        _, params = q.transition_parameters("ef")
        frequency = q.parameters.drive_frequency_ef - q.parameters.drive_frequency_ge
        drag_params = list(params["pulse"].items())
        rst_pls_params = {
            "function": "x180_ef_reset_pulse",
            "frequency": frequency,
            "pulse_params": tuple(drag_params),
        }
        reset_pulse = dsl.create_pulse(rst_pls_params, name="x180_ef_reset")
        dsl.play(
            q.signals["drive"],
            amplitude=params["amplitude_pi"],
            phase=0,
            length=params["length"],
            pulse=reset_pulse,
        )

    @dsl.quantum_operation(broadcast=False)
    def active_reset(
        self,
        qubits: Qubits,
        active_reset_states: str | tuple = "ge",
        number_resets: int = 1,
        feedback_processing_delay: float = 0.0,
        handles: Sequence[str] | None = None,
        measure_section_length: float | None = None,
    ) -> None:
        """Reset a qubit into the ground state, 'g', using feedback-based active reset.

        Arguments:
            qubits:
                The qubits to reset.
            handles:
                The handles to store the active-reset acquisition results in for each
                qubit.
            active_reset_states:
                The qubit states to reset. Can be any combination of ("g", "e", "f").
                Default: "ge"
            number_resets:
                The number of active reset rounds to apply
            feedback_processing_delay:
                Feedback processing time.
                Default: 300ns
            measure_section_length:
                The length of the measure section. If multiple qubits are passed, the
                measure section must have the same length for each qubit.
                Default: None.
        """
        if isinstance(qubits, TunableTransmonQubit):
            qubits = [qubits]
        if len(qubits) > 1:
            raise NotImplementedError(
                "The active reset operation only supports one qubit at the moment. "
                "Multi-qubit support will be added soon."
            )

        if not all(s in ["g", "e", "f"] for s in active_reset_states):
            raise NotImplementedError(
                "The active reset operation can only be applied on the states 'g', "
                "'e', 'f' at the moment."
            )

        if handles is None:
            handles = [dsl.handles.active_reset_handle(q.uid) for q in qubits]
        if len(handles) != len(qubits):
            raise ValueError(
                f"Please provide a handle for each qubits. Currently, there are "
                f"{len(qubits)} qubits and {len(handles)} handles."
            )

        for qidx, q in enumerate(qubits):
            for _nr in range(number_resets):
                sec = self.measure(q, handle=handles[qidx])
                sec.length = measure_section_length
                self.passive_reset(q, delay=feedback_processing_delay)
                with dsl.match(name=f"match_{q.uid}", handle=handles[qidx]):
                    with dsl.case(name=f"case_{q.uid}_g", state=0):
                        pass
                    with dsl.case(name=f"case_{q.uid}_e", state=1):
                        if "e" in active_reset_states:
                            self.x180.omit_section(q)
                        else:
                            pass
                    with dsl.case(name=f"case_{q.uid}_f", state=2):
                        if "f" in active_reset_states:
                            self.x180_ef_reset.omit_section(q)
                            self.x180.omit_section(q)
                        else:
                            pass

    @dsl.quantum_operation(broadcast=False)
    def calibration_traces(
        self,
        qubits: Qubits,
        states: str | tuple = "ge",
        active_reset: bool = False,  # noqa: FBT001, FBT002
        active_reset_states: str | tuple = "ge",
        active_reset_repetitions: int = 1,
        feedback_processing_delay: float = 0.0,
        measure_section_length: float | None = None,
    ) -> None:
        """Add calibration-trace measurements.

        Arguments:
            qubits:
                The qubits to reset.
            states:
                The calibration states to prepare. Can be any combination of
                ("g", "e", "f"). The same states are prepared for each qubit.
                Default: "ge"
            active_reset: whether to use active reset to prepare the qubit in g before
                every calibration state preparation
            active_reset_states:
                The qubit states to reset. Can be any combination of ("g", "e", "f").
                Default: "ge"
            active_reset_repetitions:
                The number of active reset rounds to apply
            feedback_processing_delay:
                Feedback processing time.
                Default: 300ns
            measure_section_length:
                The length of the measure section. If multiple qubits are passed, the
                measure section must have the same length for each qubit.
                Default: None.
        """
        if isinstance(qubits, TunableTransmonQubit):
            qubits = [qubits]
        for state in states:
            if active_reset:
                active_reset_handles = [
                    dsl.handles.active_reset_calibration_trace_handle(q.uid, state)
                    for q in qubits
                ]
                self.active_reset(
                    qubits,
                    active_reset_states=active_reset_states,
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
                        # Fix the length of the measure section
                        sec.length = measure_section_length
                        self.passive_reset(q)


@dsl.pulse_library.register_pulse_functional
def x180_ef_reset_pulse(
    x: np.ndarray,
    frequency: float,
    pulse_params: list[tuple],
    length: float,
    **_,
) -> np.ndarray:
    """Modulated x180 pulse on the ef transition, used for active reset.

    Arguments:
        x:
            Array between -1 and 1.
        frequency:
            Modulation frequency of the pulse.
        pulse_params:
            Parameters of the pulse functional.
        length:
            Pulse length.
        **_: keyword arguments
            uid ([str][]): Unique identifier of the pulse
            length ([float][]): Length of the pulse in seconds
            amplitude ([float][]): Amplitude of the pulse

    Returns:
        the array describing the pulse waveform
    """
    pls_kwags = dict(pulse_params)
    pulse_func = pls_kwags.pop("function")
    time = 0.5 * (x + 1) * length
    wfm = dsl.pulse_library.pulse_factory(pulse_func)(**pls_kwags).evaluate(x)
    return np.exp(-1j * 2 * np.pi * frequency * time) * wfm
