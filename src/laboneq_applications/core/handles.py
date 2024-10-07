"""Module that defines acquisition handle formatters for LabOneQ experiments."""

from __future__ import annotations

RESULT_PREFIX = "result"
CALIBRATION_TRACE_PREFIX = "cal_trace"
ACTIVE_RESET_PREFIX = "active_reset"


def result_handle(
    qubit_name: str, prefix: str = RESULT_PREFIX, suffix: str | None = None
) -> str:
    """Return the acquisition handle for the main sweep result.

    The equivalent of `"result_{qubit_name}".format(qubit_name=qubit_name).`

    Args:
        qubit_name: The name of the qubit.
        prefix: The prefix to use for the handle.
        suffix: The suffix to use for the handle.

    Returns:
        The acquisition handle for the main sweep result for the given qubit.

    Example:
        ```python
        qubit_name = "q0"
        handle = result_handle(qubit_name)
        ```
    """
    return (
        f"{prefix}/{qubit_name}"
        if suffix is None
        else f"{prefix}/{qubit_name}/{suffix}"
    )


def calibration_trace_handle(
    qubit_name: str,
    state: str,
    prefix: str = CALIBRATION_TRACE_PREFIX,
) -> str:
    """Return the acquisition handle for a calibration trace.

    The equivalent of
    `"cal_trace/{qubit_name}/{state}".format(qubit_name=qubit_name, state=state).`

    Args:
        qubit_name: The name of the qubit.
        state: The state of the qubit.
        prefix: The prefix to use for the handle.

    Returns:
        The acquisition handle for the calibration trace for the given qubit and state.

    Example:
        ```python
        qubit_name = "q0"
        state = "e"
        handle = trace_handle(qubit_name, state)
        ```
    """
    return f"{prefix}/{qubit_name}/{state}"


def active_reset_handle(
    qubit_name: str,
    tag: str,
    prefix: str = ACTIVE_RESET_PREFIX,
) -> str:
    """Return the acquisition handle for an active reset.

    The equivalent of
    `"active_reset/{qubit_name}/{tag}".format(qubit_name=qubit_name, tag=tag).`

    Args:
        qubit_name: The name of the qubit.
        tag: The tag of the active reset.
        prefix: The prefix to use for the handle.

    Returns:
        The acquisition handle for the active reset for the given qubit and tag.

    Example:
        ```python
        qubit_name = "q0"
        tag = "0"
        handle = active_reset_handle(qubit_name, tag)
        ```
    """
    return f"{prefix}/{qubit_name}/{tag}"
