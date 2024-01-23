from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from dataclasses import dataclass, field
from typing import Any, Optional

from laboneq.dsl.quantum.quantum_element import QuantumElement
from laboneq.dsl.quantum import (
    Transmon,
    TransmonParameters,
)
from laboneq.dsl.device.io_units import LogicalSignal
from laboneq.dsl.experiment import pulse_library

@classformatter
@dataclass
class TransmonQubitParameters(TransmonParameters):
    #: readout amplitude
    readout_amplitude: Optional[float] = 1
    #: length of the readout pulse
    readout_pulse_length: Optional[float] = 2e-6
    #: duration of the weighted integration, defaults to 2 us.
    readout_integration_length: Optional[float] = 2.0e-6
    #: integration kernels
    readout_integration_kernels: Optional[list] = None
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
class TransmonQubit(Transmon):
    """A class for a superconducting, flux-tuneable Transmon Qubit."""

    parameters: TransmonQubitParameters

    def __init__(
        self,
        uid: str | None = None,
        signals: dict[str, LogicalSignal] | None = None,
        parameters: TransmonQubitParameters | dict[str, Any] | None = None,
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
            self.parameters = TransmonQubitParameters()
        elif isinstance(parameters, dict):
            self.parameters = TransmonQubitParameters(**parameters)
        else:
            self.parameters = parameters
        QuantumElement.__init__(self, uid=uid, signals=signals)

    def default_integration_kernels(self):
        return [pulse_library.const(
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
        if integration_kernels is None:
            integration_kernels = [self.default_integration_kernels()]
        return integration_kernels