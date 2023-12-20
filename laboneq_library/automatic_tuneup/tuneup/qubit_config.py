from collections import UserList
from dataclasses import dataclass
from typing import List, Optional

from laboneq.simple import Qubit

from laboneq_library.automatic_tuneup.tuneup.analyzer import Analyzer

from .params import SweepParams


@dataclass
class QubitConfig:
    """A class to store the information of a qubit and its scan parameters.
    Args:
        parameter: Tuneup SweepParams object.
        qubit: L1Q Qubit.
        need_to_verify: Whether the qubit needs to be verified. If False, the qubit will not be verified and updated.
        update_key: The parameter of the qubit that needs to be updated.
        pulses: Pulses used in the scan.
        analyzer: Analyzer used to analyze the results of the scan.
    """

    parameter: SweepParams
    qubit: Qubit
    need_to_verify: bool = True
    update_key: str = ""
    pulses: Optional[dict] = None
    analyzer: Optional[Analyzer] = None

    def __post_init__(self):
        self._analyzed_result = None
        self._update_value = None
        self._verified = False

    def __setattr__(self, prop, val):
        if prop == "update_key" and self.need_to_verify:
            self._check_key(val)
        super().__setattr__(prop, val)

    def _check_key(self, update_key):
        self._update_key_in_user_defined = False
        if hasattr(self.qubit.parameters, update_key):
            self._update_key = update_key
        elif update_key in self.qubit.parameters.user_defined:
            self._update_key_in_user_defined = True
        else:
            raise ValueError("The update key must be a valid parameter of the qubit")

    def update_qubit(self):
        """Update the qubit with the analyzed result."""
        if self._update_key_in_user_defined:
            self._update_value = self._analyzed_result
            self.qubit.parameters.user_defined[self.update_key] = self._update_value
        else:
            if self.update_key == "readout_resonator_frequency":
                self._update_value = (
                    self._analyzed_result + self.qubit.parameters.readout_lo_frequency
                )
            elif (
                "resonance_frequency" in self.update_key
            ):  # resonance_frequency_ge, resonance_frequency_ef, and resonance_frequency
                self._update_value = (
                    self._analyzed_result + self.qubit.parameters.drive_lo_frequency
                )
            else:
                self._update_value = self._analyzed_result
            setattr(self.qubit.parameters, self.update_key, self._update_value)

    def copy(self):
        return QubitConfig(
            self.parameter,
            self.qubit,
            need_to_verify=self.need_to_verify,
            update_key=self.update_key,
            pulses=self.pulses,
            analyzer=self.analyzer,
        )


class QubitConfigs(UserList[QubitConfig]):
    """A container class to store a list of QubitConfig objects, each of which stores the information of a qubit and its scan parameters."""

    def __init__(self, *args):
        super().__init__(*args)

    def get_qubits(self) -> List[Qubit]:
        """Get the qubits in the QubitConfigs."""
        return [qubit_config.qubit for qubit_config in self.data]

    def get_parameters(self) -> List[SweepParams]:
        """Get the parameters in the QubitConfigs."""
        return [qubit_config.parameter for qubit_config in self.data]

    def all_verified(self) -> bool:
        """Check if all the qubits have been verified. If all qubits need not to be verified,
        return True."""
        return all(
            [qubit_config._verified for qubit_config in self.get_need_to_verify()]
        )

    def get_need_to_verify(self) -> List[QubitConfig]:
        """Get the qubits that need to be verified."""
        return [
            qubit_config for qubit_config in self.data if qubit_config.need_to_verify
        ]

    def copy(self):
        return QubitConfigs([qubit_config.copy() for qubit_config in self.data])
