"""Tunable transmon qubits, parameters and operations."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration import Oscillator
from laboneq.dsl.quantum import (
    Transmon,
    TransmonParameters,
)

from laboneq_applications import dsl
from laboneq_applications.core.quantum_operations import (
    QuantumOperations,
    create_pulse,
    quantum_operation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.dsl.device.io_units import LogicalSignal
    from laboneq.dsl.experiment.pulse import Pulse
    from laboneq.dsl.parameter import SweepParameter

# TODO: Add tests for TunableTransmonQubitParameters and TunableTransmonQubit.

# TODO: Implement multistate 0-1-2 measurement operation

# TODO: Implement support for discrimination thresholds

# TODO: Add support for specifying integration kernels as a list of sample
#       values.

# TODO: Add rotate_xy gate that performs a rotation about an axis in the xy-plane.

# TODO: Look at parameters in TransmonParameters (i.e. the base class).


@classformatter
@dataclass
class TunableTransmonQubitParameters(TransmonParameters):
    """Qubit parameters for `TunableTransmonQubit` instances."""

    #: ge drive-pulse parameters
    drive_parameters_ge: dict | None = field(
        default_factory=lambda: {
            "amplitude_pi": 0.2,
            "amplitude_pi2": 0.1,
            "length": 50e-9,
            "pulse": {"function": "drag", "beta": 0, "sigma": 0.25},
        },
    )
    #: ef drive-pulse parameters
    drive_parameters_ef: dict | None = field(
        default_factory=lambda: {
            "amplitude_pi": 0.2,
            "amplitude_pi2": 0.1,
            "length": 50e-9,
            "pulse": {"function": "drag", "beta": 0, "sigma": 0.25},
        },
    )
    #: readout parameters
    readout_parameters: dict | None = field(
        default_factory=lambda: {
            "amplitude": 1.0,
            "length": 2e-6,
            "pulse": {"function": "const"},
        },
    )
    #: integration parameters
    readout_integration_parameters: dict | None = field(
        default_factory=lambda: {
            #: duration of the weighted integration
            "length": 2e-6,
            #: integration kernels, either "default" or list of pulse dictionaries
            "kernels": "default",
            # TODO: It would be nice to be able to change to the default const pulses
            #       without losing any kernel pulse setting. Define a "kernels_type"?
            #: discrimination integration thresholds, either None or list of float
            "discrimination_thresholds": None,
        },
    )
    #: Duration of the wait time after readout and for reset
    reset_delay_length: float | None = 1e-6
    #: length of the qubit drive pulse in spectroscopy
    spectroscopy_pulse_length: float | None = 5e-6
    #: amplitude of the qubit drive pulse in spectroscopy
    spectroscopy_amplitude: float | None = 1
    #: slot number on the dc source used for applying a dc voltage to the qubit
    dc_slot: int | None = 0
    #: qubit dc parking voltage
    dc_voltage_parking: float | None = 0.0

    #: deprecated parameters still used by examples:
    readout_amplitude: float | None = 0.1
    readout_discrimination_thresholds: list | None = None
    readout_integration_kernels: list | None = None
    readout_integration_length: float | None = 1e-6
    readout_pulse_length: float | None = 1e-6
    readout_integration_kernels_type: str = "default"

    def _override(self, overrides: dict) -> None:
        invalid_params = self._get_invalid_param_paths(overrides)
        if invalid_params:
            raise ValueError(
                f"Update parameters do not match the qubit "
                f"parameters: {invalid_params}",
            )

        for param_path, value in overrides.items():
            keys = param_path.split(".")
            obj = self
            for key in keys[:-1]:
                obj = obj[key] if isinstance(obj, dict) else getattr(obj, key)
            if isinstance(obj, dict):
                if keys[-1] in obj:
                    obj[keys[-1]] = value
            elif hasattr(obj, keys[-1]):
                setattr(obj, keys[-1], value)

    def _get_invalid_param_paths(self, overrides: dict[str, Any]) -> Sequence:
        invalid_params = []
        for param_path in overrides:
            keys = param_path.split(".")
            obj = self
            for key in keys:
                if isinstance(obj, dict):
                    if key not in obj:
                        invalid_params.append(param_path)
                        break
                    obj = obj[key]
                elif not hasattr(obj, key):
                    invalid_params.append(param_path)
                    break
                else:
                    obj = getattr(obj, key)
        return invalid_params


@classformatter
@dataclass(init=False, repr=True, eq=False)
class TunableTransmonQubit(Transmon):
    """A class for a superconducting, flux-tuneable Transmon Qubit."""

    TRANSITIONS = ("ge", "ef")

    parameters: TunableTransmonQubitParameters

    def __init__(
        self,
        uid: str | None = None,
        signals: dict[str, LogicalSignal] | None = None,
        parameters: TunableTransmonQubitParameters | dict[str, Any] | None = None,
    ):
        """Initializes a new Transmon Qubit.

        Args:
            uid: A unique identifier for the Qubit.
            signals: A mapping of logical signals associated with the qubit.
                Qubit accepts the following keys in the mapping: 'drive', 'measure',
                'acquire', 'flux'.

                This is so that the Qubit parameters are assigned into the correct
                signal lines in calibration.
            parameters: Parameters associated with the qubit.
                Required for generating calibration and experiment signals via
                `calibration()` and `experiment_signals()`.
        """
        if parameters is None:
            parameters = TunableTransmonQubitParameters()
        elif isinstance(parameters, dict):
            parameters = TunableTransmonQubitParameters(**parameters)
        else:
            self.parameters = parameters
        super().__init__(uid=uid, signals=signals, parameters=parameters)

    def transition_parameters(self, transition: str | None = None) -> tuple[str, dict]:
        """Return the transition drive signal line and parameters.

        Arguments:
            transition:
                The transition to return parameters for. May be `None`,
                `"ge"` or `"ef"`. `None` defaults to `"ge"`.

        Returns:
            params:
                The drive parameters for the transition.
            line:
                The drive line for the transition.

        Raises:
            ValueError:
                If the transition is not `None`, `"ge"` or `"ef"`.
        """
        if transition is None:
            transition = "ge"
        if transition not in self.TRANSITIONS:
            raise ValueError(
                f"Transition {transition!r} is not one of None, 'ge' or 'ef'.",
            )
        line = "drive" if transition == "ge" else "drive_ef"
        params = getattr(self.parameters, f"drive_parameters_{transition}")
        return line, params

    def default_integration_kernels(self) -> list[Pulse]:
        """Return a default list of integration kernels.

        Returns:
            A list consisting of a single constant pulse with length equal to
            `readout_integration_parameters.length`.
        """
        return [
            create_pulse(
                {
                    "function": "const",
                    "length": self.parameters.readout_integration_parameters["length"],
                    "amplitude": 1.0,
                },
                name=f"integration_kernel_{self.uid}",
            ),
        ]

    def get_integration_kernels(
        self,
        kernel_pulses: list[dict] | str | None = None,
    ) -> list[Pulse]:
        """Create readout integration kernels for the transmon.

        Arguments:
            kernel_pulses:
                Custom definitions for the kernel pulses. If present,
                it replaces the values of the qubit parameter
                `readout_integration_parameters.kernels`.

        The special value `"default"` for either `kernel_pulses` or the
        `readout_integration_parameters.kernels` parameter returns
        the default kernels from `.default_integration_kernels()`.

        Returns:
            A list of integration kernel pulses.
        """
        if kernel_pulses is None:
            kernel_pulses = self.parameters.readout_integration_parameters["kernels"]

        if kernel_pulses == "default":
            integration_kernels = self.default_integration_kernels()
        elif isinstance(kernel_pulses, (list, tuple)) and len(kernel_pulses) > 0:
            integration_kernels = [
                create_pulse(kernel_pulse, name=f"integration_kernel_{self.uid}")
                for kernel_pulse in kernel_pulses
            ]
        else:
            raise TypeError(
                f"{self.__class__.__name__} readout integration kernels"
                f" should be either 'default' or a list of pulse dictionaries.",
            )

        return integration_kernels

    def replace(
        self,
        parameters: dict,
    ) -> TunableTransmonQubit:
        """Return a new qubit with replaced parameters.

        Args:
            parameters: A dictionary of parameter and values to replace.

        Returns:
            A new_qubit with the replaced values. If a qubit-parameter dictionary is
            empty, the unmodified qubit is returned.

        Raises:
            ValueError:
                Update parameters do not match the qubit parameters.

        Examples:
            ```python
            qubit = TunableTransmonQubit()
            parameters = {
                "readout_range_out":10,
                "drive_parameters_ge.length": 100e-9,
            }
            new_qubit = qubit.replace(parameters)
            ```

        """
        if not parameters:
            return self
        new_qubit = copy.deepcopy(self)
        try:
            new_qubit.parameters._override(parameters)
        except ValueError as err:
            raise ValueError(f"Cannot update {self.uid}") from err
        return new_qubit

    def update(
        self,
        parameters: dict,
    ) -> None:
        """Update the qubit with the given parameters.

        Args:
            parameters: A dictionary of parameter and values to update.

        Raises:
            ValueError:
                No updates are made if any of the parameters is not found in the qubit.

        Examples:
            ```python
            qubit = TunableTransmonQubit()
            parameters = {
                "readout_range_out":10,
                "drive_parameters_ge.length": 100e-9,
            }
            qubit.update(parameters)
        """
        try:
            self.parameters._override(parameters)
        except ValueError as err:
            raise ValueError(f"Cannot update {self.uid}") from err

    def _get_invalid_params_to_update(self, parameters: dict) -> Sequence:
        """Check if the parameters to update exist in the qubit.

        Args:
            parameters: A dictionary of parameter and values to update.

        Returns:
            invalid_params: A list of parameters that are not found in the qubit.
        """
        return self.parameters._get_invalid_param_paths(parameters)


class TunableTransmonOperations(QuantumOperations):
    """Operations for TunableTransmonQubits."""

    QUBIT_TYPES = TunableTransmonQubit

    # common angles used by rx, ry and rz.
    _PI = np.pi
    _PI_BY_2 = np.pi / 2

    @quantum_operation
    def barrier(self, q: TunableTransmonQubit) -> None:
        """Add a barrier on all the qubit signals.

        Arguments:
            q:
                The qubit to block on.

        Note:
            A barrier returns an empty section that
            reserves all the qubit signals. The
            signals are reserved via `@quantum_operation` so
            the implementation of this operations is just
            `pass`.
        """

    @quantum_operation
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

    @quantum_operation
    def set_frequency(
        self,
        q: TunableTransmonQubit,
        frequency: float | SweepParameter,
        *,
        transition: str | None = None,
        readout: bool = False,
        rf: bool = True,
    ) -> None:
        """Sets the frequency of the given qubit transition drive line.

        Arguments:
            q:
                The qubit to set the drive frequency of.
            frequency:
                The frequency to set the drive line to in Hz.
                By default the frequency specified is the RF frequency.
                The oscillator frequency may be set directly instead
                by passing `rf=False`.
            transition:
                The transition to rotate. By default this is "ge"
                (i.e. the 0-1 transition).
            readout:
                If true, the frequency of the readout line is set
                instead.
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
        if readout:
            signal_line = "measure"
            lo_frequency = q.parameters.readout_lo_frequency
        else:
            signal_line, _ = q.transition_parameters(transition)
            lo_frequency = q.parameters.drive_lo_frequency

        if rf:
            # This subtraction works for both numbers and SweepParameters
            frequency -= lo_frequency

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

    @quantum_operation
    def measure(
        self,
        q: TunableTransmonQubit,
        handle: str,
        readout_pulse: dict | None = None,
        kernel_pulses: list[dict] | Literal["default"] | None = None,
    ) -> None:
        """Perform a measurement on the qubit.

        The measurement operation plays a readout pulse and performs an acquistion.
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
                `readout_integration_parameters.length` parameter is used.

                If not specified or `None`, the kernels specified in the
                qubit's `readout_integration_parameters.kernels` are used.
        """
        ro_params = q.parameters.readout_parameters
        ro_pulse = create_pulse(ro_params["pulse"], readout_pulse, name="readout_pulse")

        integration_params = q.parameters.readout_integration_parameters
        kernels = q.get_integration_kernels(kernel_pulses)

        dsl.measure(
            measure_signal=q.signals["measure"],
            measure_pulse_amplitude=ro_params["amplitude"],
            measure_pulse_length=ro_params["length"],
            measure_pulse=ro_pulse,
            handle=handle,
            acquire_signal=q.signals["acquire"],
            integration_kernel=kernels,
            integration_length=integration_params["length"],
            reset_delay=None,
        )

    @quantum_operation
    def acquire(
        self,
        q: TunableTransmonQubit,
        handle: str,
        kernel_pulses: list[dict] | Literal["default"] | None = None,
    ) -> None:
        """Perform an acquistion on the qubit.

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
                `readout_integration_parameters.length` parameter is used.

                If not specified or `None`, the kernels specified in the
                qubit's `readout_integration_parameters.kernels` are used.
        """
        integration_params = q.parameters.readout_integration_parameters
        kernels = q.get_integration_kernels(kernel_pulses)

        dsl.acquire(
            signal=q.signals["acquire"],
            handle=handle,
            kernel=kernels,
            length=integration_params["length"],
        )

    @quantum_operation
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
            self.x180(q, transition="ge")
        elif state == "f":
            self.x180(q, transition="ge")
            self.x180(q, transition="ef")
        else:
            raise ValueError(f"Only states g, e and f can be prepared, not {state!r}")

    @quantum_operation
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

    @quantum_operation
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

        rx_pulse = create_pulse(params["pulse"], pulse, name="rx_pulse")

        dsl.play(
            q.signals[drive_line],
            amplitude=amplitude,
            phase=phase,
            length=length,
            pulse=rx_pulse,
        )

    @quantum_operation
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

    @quantum_operation
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

    @quantum_operation
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

        ry_pulse = create_pulse(params["pulse"], pulse, name="ry_pulse")

        dsl.play(
            q.signals[drive_line],
            amplitude=amplitude,
            phase=phase,
            length=length,
            pulse=ry_pulse,
        )

    @quantum_operation
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

    @quantum_operation
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

    @quantum_operation
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

    @quantum_operation
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

    @quantum_operation
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
