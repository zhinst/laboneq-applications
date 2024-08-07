"""A collection of qubit and quantum operation definitions.

Each sub-module supports a different kind of quantum device.
"""

__all__ = ["QPU", "QuantumPlatform"]

from .qpu import QPU, QuantumPlatform
