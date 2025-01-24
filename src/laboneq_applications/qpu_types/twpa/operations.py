# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Traveling wave parametric amplifier (TWPA) operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.dsl.calibration import CancellationSource, Oscillator
from laboneq.dsl.enums import ModulationType
from laboneq.dsl.experiment.builtins_dsl import QuantumOperations, quantum_operation
from laboneq.simple import dsl

from .twpa_types import TWPA

if TYPE_CHECKING:
    from laboneq.dsl.calibration import Calibration
    from laboneq.dsl.parameter import SweepParameter


class TWPAOperations(QuantumOperations):
    """Operations for TWPA."""

    QUBIT_TYPES = TWPA

    @quantum_operation
    def set_readout_frequency(
        self,
        twpa: TWPA,
        frequency: float | SweepParameter,
        *,
        rf: bool = True,
    ) -> None:
        """Sets the frequency of the given readout line.

        Arguments:
            twpa:
                The parametric amplifier to set the readout frequency of.
            frequency:
                The frequency to set in Hz.
                By default the frequency specified is the RF frequency.
                The oscillator frequency may be set directly instead
                by passing `rf=False`.
            rf:
                If True, set the RF frequency of the transition.
                If False, set the oscillator frequency directly instead.
                The default is to set the RF frequency.

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
        signal_line = "measure"
        lo_frequency = twpa.parameters.readout_lo_frequency

        if rf:
            # This subtraction works for both numbers and SweepParameters
            frequency -= lo_frequency

        calibration = dsl.experiment_calibration()
        signal_calibration = calibration[twpa.signals[signal_line]]
        oscillator = signal_calibration.oscillator

        if oscillator is None:
            oscillator = signal_calibration.oscillator = Oscillator(frequency=frequency)
        if getattr(oscillator, "_set_frequency", False):
            # We mark the oscillator with a _set_frequency attribute to ensure that
            # set_frequency isn't performed on the same oscillator twice. Ideally
            # LabOne Q would provide a set_frequency DSL method that removes the
            # need for setting the frequency on the experiment calibration.
            raise RuntimeError(
                f"Readout frequency of twpa {twpa.uid} measure line was set multiple"
                f" times using the set_readout_frequency operation.",
            )

        oscillator._set_frequency = True
        oscillator.frequency = frequency
        # LabOne Q does not support software modulation of measurement
        # signal sweeps because it results in multiple readout waveforms
        # on the same readout signal. Ideally the LabOne Q compiler would
        # sort this out for us when the modulation type is AUTO, but currently
        # it does not.
        oscillator.modulation_type = ModulationType.HARDWARE

    @quantum_operation
    def set_readout_amplitude(
        self,
        twpa: TWPA,
        amplitude: float | SweepParameter,
        *,
        calibration: Calibration | None = None,
    ) -> None:
        """Sets the readout amplitude of the measure line.

        Arguments:
            twpa:
                The twpa associated with the measure line.
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
        signal_calibration = calibration[twpa.signals["measure"]]

        if getattr(calibration, "_set_readout_amplitude", False):
            # We mark the oscillator with a _set_readout_amplitude attribute to ensure
            # that set_readout_amplitude isn't performed on the same signal twice.
            # Ideally LabOne Q DSL provide a more direct method that removes the
            # need for setting amplitude on the experiment calibration.
            raise RuntimeError(
                f"Readout amplitude of twpa {twpa.uid}"
                f" measure line was set multiple times"
                f" using the set_readout_amplitude operation.",
            )

        calibration._set_readout_amplitude = True
        signal_calibration.amplitude = amplitude

    @quantum_operation
    def set_pump_frequency(
        self,
        twpa: TWPA,
        frequency: float | SweepParameter,
        calibration: Calibration | None = None,
    ) -> None:
        """Sets the pump frequency of the given twpa.

        Arguments:
            twpa:
                The parametric amplifier to set the pump frequency of.
            frequency:
                The frequency to set in Hz.
            calibration:
                The experiment calibration to update (see the note below).
                By default, the calibration from the currently active
                experiment context is used. If no experiment context is
                active, for example when using
                `@qubit_experiment(context=False)`, the calibration
                object may be passed explicitly.

        Raises:
            RuntimeError:
                If there is an attempt to call `set_pump_frequency` more than
                once on the same signal.
        """
        if calibration is None:
            calibration = dsl.experiment_calibration()
        signal_calibration = calibration[twpa.signals["acquire"]]
        if getattr(calibration, "_set_pump_frequency", False):
            raise RuntimeError(
                f"Pump frequency of twpa {twpa.uid}"
                f" measure line was set multiple times"
                f" using the set_pump_frequency operation.",
            )

        calibration._set_pump_frequency = True
        signal_calibration.amplifier_pump.pump_frequency = frequency

    @quantum_operation
    def set_pump_power(
        self,
        twpa: TWPA,
        power: float | SweepParameter,
        calibration: Calibration | None = None,
    ) -> None:
        """Sets the pump power of the given twpa.

        Arguments:
            twpa:
                The parametric amplifier to set the pump power of.
            power:
                The power to set in dBm.
            calibration:
                The experiment calibration to update (see the note below).
                By default, the calibration from the currently active
                experiment context is used. If no experiment context is
                active, for example when using
                `@qubit_experiment(context=False)`, the calibration
                object may be passed explicitly.

        Raises:
            RuntimeError:
                If there is an attempt to call `set_pump_power` more than
                once on the same signal.
        """
        if calibration is None:
            calibration = dsl.experiment_calibration()
        signal_calibration = calibration[twpa.signals["acquire"]]
        if getattr(calibration, "_set_pump_power", False):
            raise RuntimeError(
                f"Pump power of twpa {twpa.uid}"
                f" measure line was set multiple times"
                f" using the set_pump_power operation.",
            )

        calibration._set_pump_power = True
        signal_calibration.amplifier_pump.pump_power = power

    @quantum_operation
    def set_pump_cancellation(
        self,
        twpa: TWPA,
        cancellation_attenuation: float | SweepParameter,
        cancellation_phaseshift: float | SweepParameter,
        cancellation: bool,  # noqa: FBT001
        calibration: Calibration | None = None,
    ) -> None:
        """Sets the pump cancellation of the given twpa.

        Arguments:
            twpa:
                The parametric amplifier to set the pump cancellation of.
            cancellation_attenuation:
                The attenuation to set in dB.
            cancellation_phaseshift:
                The phase shift to set in degrees.
            cancellation:
                If True, the cancellation is on and the cancellation
                source is internal. If False, the cancellation is off
                and the cancellation source is switched to external
                for cross-talk reduction.
            calibration:
                The experiment calibration to update (see the note below).
                By default, the calibration from the currently active
                experiment context is used. If no experiment context is
                active, for example when using
                `@qubit_experiment(context=False)`, the calibration
                object may be passed explicitly.

        Raises:
            RuntimeError:
                If there is an attempt to call `set_pump_cancellation` more than
                once on the same signal.
        """
        if calibration is None:
            calibration = dsl.experiment_calibration()
        signal_calibration = calibration[twpa.signals["acquire"]]
        if getattr(calibration, "_set_pump_cancellation", False):
            raise RuntimeError(
                f"Pump cancellation of twpa {twpa.uid}"
                f" measure line was set multiple times"
                f" using the set_pump_cancellation operation.",
            )

        calibration._set_pump_cancellation = True

        signal_calibration.amplifier_pump.cancellation_on = cancellation
        signal_calibration.amplifier_pump.cancellation_phase = cancellation_phaseshift
        signal_calibration.amplifier_pump.cancellation_attenuation = (
            cancellation_attenuation
        )
        signal_calibration.amplifier_pump.cancellation_source = (
            CancellationSource.INTERNAL if cancellation else CancellationSource.EXTERNAL
        )

    @quantum_operation
    def twpa_measure(
        self,
        twpa: TWPA,
        handle: str,
        readout_pulse: dict | None = None,
    ) -> None:
        """Performs a readout measurement on the given twpa.

        Arguments:
            twpa:
                The twpa to perform the readout measurement on.
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
        """
        measure_line, ro_params = twpa.readout_parameters()
        ro_pulse = dsl.create_pulse(
            ro_params["pulse"], readout_pulse, name="readout_pulse"
        )
        dsl.play(
            signal=twpa.signals[measure_line],
            pulse=ro_pulse,
            amplitude=twpa.parameters.readout_amplitude,
            length=twpa.parameters.readout_length,
        )
        dsl.acquire(
            signal=twpa.signals["acquire"],
            handle=handle,
            length=twpa.parameters.readout_length,
        )

    @quantum_operation
    def twpa_acquire(
        self,
        twpa: TWPA,
        handle: str,
    ) -> None:
        """Perform an acquisition on the twpa.

        The acquire operation performs only an acquisition. If you wish to play
        a readout pulse and perform an acquisition, use the `measure` operation.

        Arguments:
            handle:
                The handle to store the acquisition results in.
            twpa:
                The twpa to acquire the signal from.
        """
        dsl.acquire(
            signal=twpa.signals["acquire"],
            handle=handle,
            length=twpa.parameters.readout_length,
        )

    @dsl.quantum_operation
    def twpa_delay(self, twpa: TWPA, time: float) -> None:
        """Add a delay on the twpa signals.

        Arguments:
            twpa:
                The qubit to delay on.
            time:
                The duration of the delay in seconds.
        """
        dsl.delay(twpa.signals["acquire"], time=time)
