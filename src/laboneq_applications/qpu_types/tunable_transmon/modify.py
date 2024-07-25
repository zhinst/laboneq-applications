"""Functions for modifying tunable transmon qubits."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from .qubit_types import TunableTransmonQubit


def modify_qubits(
    parameters: Sequence[tuple[TunableTransmonQubit, dict]],
) -> Sequence[TunableTransmonQubit]:
    """Create new qubits with replaced parameter values.

    New qubits are created by copying the original qubits and replacing
    the parameter values.

    Args:
        parameters:
            A sequence of pairs of qubits and dictionaries of qubit-parameter
            values to override. If a qubit-parameter dictionary is empty, the
            unmodified qubit is returned.


    Returns:
        new_qubits:
            A list of new qubits with the replaced values. The list is in the
            same order as the input qubits.

    Raises:
        ValueError:
            No updates are made if any of the parameters is not found in the qubit.

    Examples:
        ```python
        [q0, q1, q2] = [TunableTransmonQubit() for _ in range(3)]
        parameters = [
            (q0, {"readout_range_out": 10, "drive_parameters_ge.length": 100e-9}),
            (q1, {"readout_range_out": 20, "drive_parameters_ge.length": 200e-9}),
            (q2, {"readout_range_out": 30, "drive_parameters_ge.length": 300e-9}),
        ]
        new_qubits = modify_qubits(parameters)
        # same qubits returned if parameters are empty
        [q0, q1, q2] = [TunableTransmonQubit() for _ in range(3)]
        parameters = [
            (q0,{}),
            (q1,{}),
            (q2,{}),
        ]
        same_qubits = modify_qubits(parameters)
        ```
    """
    new_qubits = []
    for qubit, temp_value in parameters:
        new_qubits.append(qubit.replace(temp_value))
    return new_qubits


@contextmanager
def modify_qubits_context(
    parameters: Sequence[tuple[TunableTransmonQubit, dict]],
) -> Generator[TunableTransmonQubit, None, None]:
    """Context manager for creating temporary qubits.

    Args:
        parameters: A sequence of pair of qubits and dictionaries of
                                    parameter and values to override.

    Yields:
        new_qubits: A generator that yields new qubits with the replaced values.

    Raises:
        ValueError:
            No updates are made if any of the parameters is not found in the qubit.

    Examples:
        ```python
        [q0,q1,q2] = [TunableTransmonQubit() for _ in range(3)]
        parameters = [
            (q0,{"readout_range_out": 10, "drive_parameters_ge.length": 100e-9}),
            (q1,{"readout_range_out": 20, "drive_parameters_ge.length": 200e-9}),
            (q2,{"readout_range_out": 30, "drive_parameters_ge.length": 300e-9}),
        ]
        with modify_qubits_context(parameters) as new_qubits:
            # do something with new_qubits
        ```
    """
    new_qubits = modify_qubits(parameters)
    yield new_qubits
