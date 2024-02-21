from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.quantum.quantum_element import QuantumElement
from laboneq.dsl.quantum import (
    Transmon,
    TransmonParameters,
)
from laboneq.dsl.device.io_units import LogicalSignal
from laboneq.dsl.experiment import pulse_library
from laboneq.simple import Section

from laboneq_library.core.quantum_operations import (
    QuantumOperations,
    quantum_operation,
    dsl,
)

# TODO: Add docstrings and type annotations.

# TODO: Add tests.


@classformatter
@dataclass
class TunableTransmonQubitParameters(TransmonParameters):
    #: readout amplitude
    readout_amplitude: Optional[float] = 1
    #: length of the readout pulse
    readout_pulse_length: Optional[float] = 2e-6
    #: duration of the weighted integration, defaults to 2 us.
    readout_integration_length: Optional[float] = 2.0e-6
    #: integration kernels type
    readout_integration_kernels_type: Optional[list] = "default"
    #: integration kernels
    readout_integration_kernels: Optional[list] = None
    #: discrimination integration thresholds
    readout_discrimination_thresholds: Optional[list] = None
    #: ge drive-pulse parameters
    drive_parameters_ge: Optional[dict] = field(
        default_factory=lambda: dict(
            amplitude_pi=0.2, amplitude_pi2=0.1, length=50e-9, beta=0, sigma=0.25
        )
    )
    #: ef drive-pulse parameters
    drive_parameters_ef: Optional[dict] = field(
        default_factory=lambda: dict(
            amplitude_pi=0.2, amplitude_pi2=0.1, length=50e-9, beta=0, sigma=0.25
        )
    )
    #: length of the qubit drive pulse in spectroscopy
    spectroscopy_pulse_length: Optional[float] = 5e-6
    #: amplitude of the qubit drive pulse in spectroscopy
    spectroscopy_amplitude: Optional[float] = 1
    #: slot number on the dc source used for applying a dc voltage to the qubit
    dc_slot: Optional[int] = 0
    #: qubit dc parking voltage
    dc_voltage_parking: Optional[float] = 0
    #: Duration of the wait time after readout
    reset_delay_length: Optional[float] = 1e-6


@classformatter
@dataclass(init=False, repr=True, eq=False)
class TunableTransmonQubit(Transmon):
    """A class for a superconducting, flux-tuneable Transmon Qubit."""

    parameters: TunableTransmonQubitParameters

    def __init__(
        self,
        uid: str | None = None,
        signals: dict[str, LogicalSignal] | None = None,
        parameters: TunableTransmonQubitParameters | dict[str, Any] | None = None,
    ):
        """
        Initializes a new Transmon Qubit.

        Args:
            uid: A unique identifier for the Qubit.
            signals: A mapping of logical signals associated with the qubit.
                Qubit accepts the following keys in the mapping: 'drive', 'measure', 'acquire', 'flux'

                This is so that the Qubit parameters are assigned into the correct signal lines in
                calibration.
            parameters: Parameters associated with the qubit.
                Required for generating calibration and experiment signals via `calibration()` and `experiment_signals()`.
        """
        if parameters is None:
            self.parameters = TunableTransmonQubitParameters()
        elif isinstance(parameters, dict):
            self.parameters = TunableTransmonQubitParameters(**parameters)
        else:
            self.parameters = parameters
        # TODO: Should this skip the Transmon base class __init__? Or should this class no inherent from Transmon?
        QuantumElement.__init__(self, uid=uid, signals=signals)

    def default_integration_kernels(self):
        return [
            pulse_library.const(
                uid=f"integration_kernel_{self.uid}",
                length=self.parameters.readout_integration_length,
                amplitude=1,
            )
        ]

    def set_default_integration_kernels(self):
        self.parameters.readout_integration_kernels = self.default_integration_kernels()

    def get_integration_kernels(self):
        """
        Get the readout_integration_kernels of the transmon.

        Returns:
            self.parameters.readout_integration_kernels if it is not None, else
            it returns a list containing one constant pulse with the length
            given by parameters.readout_integration_length,

        """
        integration_kernels = self.parameters.readout_integration_kernels
        ro_int_type = self.parameters.readout_integration_kernels_type
        if self.parameters.readout_integration_kernels_type == "default":
            integration_kernels = self.default_integration_kernels()
        elif ro_int_type == "optimal":
            if integration_kernels is None:
                raise ValueError(
                    f"No optimal weights exist for {self.uid}. Please set "
                    f"readout_integration_kernels_type to 'default'.")
        else:
            raise ValueError(
                f"Unknown readout_integration_kernels_type {ro_int_type}. "
                f"Currently supported values are 'default' and 'optimal'.")
        return integration_kernels


class TunableTransmonOperations(QuantumOperations):
    """Operations for TunableTransmonQubits."""

    QUBIT_TYPE = TunableTransmonQubit
    TRANSITIONS = ("ge", "ef")

    @quantum_operation
    def delay(q, time, transition=None) -> Section:
        dsl.delay(q.signals["drive"], time=time)

    @quantum_operation
    def barrier(q, transition=None) -> Section:
        dsl.reserve(q.signals["drive"])

    @quantum_operation
    def rx(q, angle, transition=None) -> Section:
        transition = "ge" if transition is None else "ef"
        drive_pulse = "TODO"
        dsl.play(
            signal=q.signals["drive"],
            pulse=drive_pulse,
            # TODO: convert angle to amplitude
            #       angle may be a float or a sweep parameter
            # TODO: better name for parameter
            amplitude=angle,  # TODO: * q.parameter["drive_amplitude_per_radian"],
        )

    @quantum_operation
    def measure(q, handle, transition=None) -> Section:
        transition = "ge" if transition is None else "ef"
        ro_pulse = "TODO"
        # TODO: Use multistate discrimination if we need to
        #       measure ef?
        # TODO: Better way to handle creating / setting the handle? Should
        #       it just be an argument?
        dsl.measure(
            measure_signal=q.signals["measure"],
            measure_pulse=ro_pulse,
            handle=handle,
            acquire_signal=q.signals["acquire"],
            # TODO: integration_kernel=integration_kernel,
            # TODO: integration_length=q.parameters.readout_integration_length,
            # TODO: reset_delay=q.parameters.reset_delay_length,
        )
