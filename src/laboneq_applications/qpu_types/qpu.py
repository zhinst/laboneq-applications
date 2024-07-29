"""This modules defines the QPU class.

The QPU class is a high-level interface to a quantum processing unit (QPU).
It contains the device setup, qubits, and quantum operations that are used
to build and run quantum experiments.

"""

from copy import deepcopy

from laboneq.dsl.device import DeviceSetup
from laboneq.dsl.session import Session

from laboneq_applications.core.quantum_operations import QuantumOperations
from laboneq_applications.typing import Qubits


class QPU:
    """Quantum Processing Unit (QPU) class."""

    def __init__(
        self,
        setup: DeviceSetup,
        qubits: Qubits,
        qop: QuantumOperations,
    ) -> None:
        """Initialize a new QPU.

        Arguments:
            setup:
                The device setup to use when building the experiment.
            qubits:
                The qubits to run the experiments on.
            qop:
                The quantum operations to use when building the experiment.
        """
        self.setup = setup
        self.qubits = qubits
        self.qop = qop

    def session(self, do_emulation: bool = False) -> Session:  # noqa: FBT001 FBT002
        """Return a new LabOne Q session.

        Arguments:
            do_emulation:
                Specifies if the session should connect
                to a emulator (in the case of 'True'),
                or the real system (in the case of 'False')
        """
        session = Session(self.setup)
        session.connect(do_emulation=do_emulation)
        return session

    def copy_qubits(self) -> Qubits:
        """Return new qubits that are a copy of the original qubits."""
        return deepcopy(self.qubits)

    def update_qubits(self, qubits: Qubits) -> None:
        """Update qubits."""
        raise NotImplementedError("This method is not implemented yet.")
