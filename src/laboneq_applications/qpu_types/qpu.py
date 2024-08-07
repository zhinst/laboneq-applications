"""This module defines the QuantumPlatform and QPU classes.

A `QPU` contains the "physics" of a quantum device -- the qubit parameters
and definition of operations on qubits.

A `QuantumPlatform` contains the `QPU`, and the `DeviceSetup` which describes
the control hardware used to interface to the device.

By itself a `QPU` provides everything needed to *build* or *design* an
experiment for a quantum device. The `DeviceSetup` provides the additional
information needed to *compile* an experiment for specific control hardware.

Together these provide a `QuantumPlatform` -- i.e. everything needed to build,
compile and run experiments on real devices.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from laboneq.dsl.session import Session

if TYPE_CHECKING:
    from laboneq.dsl.device import DeviceSetup

    from laboneq_applications.core.quantum_operations import QuantumOperations
    from laboneq_applications.typing import Qubits


class QuantumPlatform:
    """A quantum hardware platform.

    A `QuantumPlatform` provides the logical description of a quantum device needed to
    define experiments (the `QPU`) and the description of the control hardware needed to
    compile an experiment (the `DeviceSetup`).

    In short, a `QPU` defines the device physics and a `DeviceSetup` defines the control
    hardware being used.

    Arguments:
        setup:
            The `DeviceSetup` describing the control hardware of the device.
        qpu:
            The `QPU` describing the parameters and topology of the quantum device
            and providing the definition of quantum operations on the device.
    """

    def __init__(
        self,
        setup: DeviceSetup,
        qpu: QPU,
    ) -> None:
        """Initialize a new QPU.

        Arguments:
            setup:
                The device setup to use when running an experiment.
            qpu:
                The QPU to use when building an experiment.
        """
        self.setup = setup
        self.qpu = qpu

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


class QPU:
    """A Quantum Processing Unit (QPU).

    A `QPU` provides the logical description of a quantum device needed to *build*
    experiments for it. For example, the qubit parameters and the definition of
    operations on those qubits.

    It does not provide a description of the control hardware needed to *compile* an
    experiment.

    In short, a `QPU` defines the device physics and a `DeviceSetup` defines the control
    hardware being used.

    Arguments:
        qubits:
            The qubits to run the experiments on.
        qop:
            The quantum operations to use when building the experiment.
    """

    def __init__(
        self,
        qubits: Qubits,
        qop: QuantumOperations,
    ) -> None:
        self.qubits = qubits
        self.qop = qop

    def copy_qubits(self) -> Qubits:
        """Return new qubits that are a copy of the original qubits."""
        return deepcopy(self.qubits)

    def update_qubits(self, qubits: Qubits) -> None:
        """Update qubits."""
        raise NotImplementedError("This method is not implemented yet.")
