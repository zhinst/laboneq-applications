# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Common type hints for the laboneq_applications library.

This module provides a set of common types for use within the
laboneq_applications library.

Consistent use of a small set of types throughout the library
keeps the list of idioms users need to learn small.

For library developers, this file documents the common idioms
used and guides us in building consistent interfaces.

Type hints
----------

* [QuantumElements]()

    Either a single qubit or a sequence of qubits.

* [QubitSweepPoints]()

    Sweep values for either a single qubit or multiple qubits. What
    value is being swept is typically described by the parameter
    name, e.g. `amplitudes`.

    If the values are for a single qubit, [QubitSweepPoints]() is
    sequence of floats or a numpy `ArrayLike`.

    If the values are for multiple qubits, [QubitSweepPoints]() is
    a sequence of such values, one for each qubit.
"""

from collections.abc import Sequence

from laboneq.workflow.typing import QuantumElements
from numpy.typing import ArrayLike
from typing_extensions import TypeAlias

__all__ = [
    "QuantumElements",
    "QubitSweepPoints",
]

QubitSweepPoints: TypeAlias = ArrayLike | Sequence[ArrayLike]
