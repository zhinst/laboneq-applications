from __future__ import annotations
import json
from typing import TYPE_CHECKING, Sequence
from collections import UserList
from importlib import import_module

if TYPE_CHECKING:
    from laboneq.simple import DeviceSetup
    from laboneq.dsl.quantum.quantum_element import QuantumElement


def _deserialize_object(data: dict) -> object:
    """Deserialize an object.

    Looks for the target object from key `__type` in the given dictionary
    and tries to import and instantiate the target object.
    """
    # TODO: Should be in the base `QuantumElement` class
    split = data.pop("__type").split(".")
    module_name = ".".join(split[:-1])
    module = import_module(module_name)
    obj = getattr(module, split[-1])
    return obj(**data)


class QubitRegister(UserList):
    """Qubit register.

    Args:
        qubits: List of qubits.
    """
    def __init__(self, qubits: Sequence[QuantumElement]):
        self.data: Sequence[QuantumElement] = qubits

    def __str__(self):
        return str([o.__class__.__name__ + f"({o.uid})" for o in self.data])

    @classmethod
    def load(cls, filename: str) -> QubitRegister:
        """Load qubits from a JSON file.

        Args:
            filename: Filename.
        """
        with open(filename, "r") as f:
            data = json.load(f)
        return cls([_deserialize_object(q) for q in data])

    def save(self, filename: str):
        """Save the qubits to an JSON file.

        Args:
            filename: Filename.
        """
        data = [q._serialize_() for q in self.data]
        with open(filename, "w") as f:
            json.dump(data, f)

    def link_signals(qubits: Sequence[QuantumElement], device_setup: DeviceSetup):
        """Link the signals of the qubits to the given `DeviceSetup`.

        Modifies the qubits' signals in-place.
        """
        for q in qubits:
            if q.uid in device_setup.logical_signal_groups:
                q.add_signals(device_setup.logical_signal_groups[q.uid].logical_signals)
