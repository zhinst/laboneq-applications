"""Tunable transmon qubits and parameters."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.quantum import (
    Transmon,
    TransmonParameters,
)

from laboneq_applications.core.quantum_operations import (
    create_pulse,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.dsl.device.io_units import LogicalSignal
    from laboneq.dsl.experiment.pulse import Pulse


# TODO: Add tests for TunableTransmonQubitParameters and TunableTransmonQubit.

# TODO: Implement support for discrimination thresholds

# TODO: Add support for specifying integration kernels as a list of sample
#       values.

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

        Arguments:
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
