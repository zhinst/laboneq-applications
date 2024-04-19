"""Core classes for working with qubit registers."""

from __future__ import annotations

import json
from collections import UserList
from dataclasses import asdict
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.simple import DeviceSetup


def _serialize_qubit(obj: QuantumElement) -> dict:
    """Serialize a qubit instance."""
    # TODO: Should be in the base `QuantumElement` class
    return {
        "uid": obj.uid,
        "signals": dict(obj.signals),  # TODO: Fix in LabOneQ to be a normal dictionary
        "parameters": asdict(obj.parameters),
        "__type": ".".join([obj.__class__.__module__, obj.__class__.__name__]),
    }


def _deserialize_object(data: dict) -> object:
    """Deserialize an object.

    Looks for the target object from key `__type` in the given dictionary
    and tries to import and instantiate the target object.
    """
    # TODO: Should be in the base `QuantumElement` class
    *module_parts, class_name = data.pop("__type").split(".")
    module_name = ".".join(module_parts)
    module = import_module(module_name)
    obj = getattr(module, class_name)
    return obj(**data)


class QubitRegister(UserList):
    """Qubit register.

    Arguments:
        qubits: List of qubits.

    Example:
        Creating the `QubitRegister`:

        >>> qubits = QubitRegister([Qubit("q0"), Qubit("q1")])
        >>> len(qubits)
        2

        Saving and loading to and from a file:

        >>> qubits.save("my_qubits.json")
        >>> qubits = QubitRegister.load("my_qubits.json")

        Connecting the qubits to an LabOne Q `DeviceSetup`:

        >>> qubits.link_signals(device_setup)
    """

    def __init__(self, qubits: Sequence[QuantumElement]):
        self.data: Sequence[QuantumElement] = qubits

    def __str__(self):
        return str([f"{o.__class__.__name__}({o.uid})" for o in self.data])

    @classmethod
    def load(cls, filename: str) -> QubitRegister:
        """Load qubits from a JSON file.

        Arguments:
            filename: The path to save the file to.

        Returns:
            `QubitRegister` loaded from the file.
        """
        with open(filename) as f:
            data = json.load(f)
        return cls([_deserialize_object(q) for q in data])

    def save(self, filename: str) -> None:
        """Save the qubits to an JSON file.

        Arguments:
            filename: The path to save the file to.
        """
        data = [_serialize_qubit(q) for q in self.data]
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def link_signals(self, device_setup: DeviceSetup) -> None:
        """Connects the signals of the qubits to the given device setup.

        Each qubit's signals are connected to the signals in the device setup
        logical signal group with the same name as the qubit unique identifier.

        Modifies the qubits' signals in-place.

        Arguments:
            device_setup: `DeviceSetup` to link the qubits to.

        Raises:
            KeyError:
                Qubit unique identifier does not exists in the
                `device_setup`'s logical signals.
        """
        for qubit in self.data:
            if qubit.uid not in device_setup.logical_signal_groups:
                msg = f"Qubit {qubit.uid} not in device setup"
                raise KeyError(msg)

        for qubit in self.data:
            qubit.add_signals(
                device_setup.logical_signal_groups[qubit.uid].logical_signals,
            )
