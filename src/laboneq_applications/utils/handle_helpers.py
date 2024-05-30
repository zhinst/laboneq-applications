"""Provide helpers for creating and parsing acquisition handles."""

from __future__ import annotations

RESULT_PREFIX = "result"
CALIBRATION_TRACE_PREFIX = "cal_trace"
ACTIVE_RESET_PREFIX = "active_reset"


def result_handle(qubit_name: str, prefix: str = RESULT_PREFIX) -> str:
    """Return the acquisition handle for the main sweep result.

    The equivalent of `"result_{qubit_name}".format(qubit_name=qubit_name).`

    Args:
        qubit_name: The name of the qubit.
        prefix: The prefix to use for the handle.

    Returns:
        The acquisition handle for the main sweep result for the given qubit.

    Example:
        ```python
        qubit_name = "q0"
        handle = result_handle(qubit_name)
        ```
    """
    return f"{prefix}/{qubit_name}"


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


def parse_result_handle(handle: str, prefix: str = RESULT_PREFIX) -> str | None:
    """Parse the qubit name from a result handle.

    Roughly the equivalent of the regular expression
    `"prefix/(?P<qubit_name>.*)$".`

    Args:
        handle: The acquisition handle to parse.
        prefix: The prefix to use for the handle.

    Returns:
        The qubit name parsed from the handle.

    Example:
        ```python
        handle = "result/q0"
        qubit_name = parse_result_handle(handle)
        ```
    """
    if prefix + "/" not in handle:
        return None
    return handle[handle.find(prefix) + len(prefix) + 1 :]


def parse_calibration_trace_handle(
    handle: str,
    prefix: str = CALIBRATION_TRACE_PREFIX,
) -> tuple[str, str]:
    """Parse the qubit name and state from a calibration trace handle.

    Roughly the equivalent of the regular expression
    `"prefix/(?P<qubit_name>.+)/(?P<state>[^/])$".`

    States may not contain a `/`.

    Args:
        handle: The acquisition handle to parse.
        prefix: The prefix to use for the handle.

    Returns:
        The qubit name and state parsed from the handle. If the handle is not matching
        the expected format, `(None, None)` is returned.

    Example:
        ```python
        handle = "cal_trace/q0/e"
        qubit_name, state = parse_calibration_trace_handle(handle)
        ```
    """
    if prefix + "/" not in handle:
        return (None, None)
    rest = handle[handle.find(prefix) + len(prefix) + 1 :]
    state_index = rest.rfind("/")
    if state_index in {-1, 0, len(rest) - 1}:
        return (None, None)
    state = rest[state_index + 1 :]
    qubit_name = rest[:state_index]
    return qubit_name, state


def parse_active_reset_handle(
    handle: str,
    prefix: str = ACTIVE_RESET_PREFIX,
) -> tuple[str, str]:
    """Parse the qubit name and tag from an active reset handle.

    Roughly the equivalent of the regular expression
    `"prefix/(?P<qubit_name>.+)/(?P<tag>[^/]+)$".`

    Tags may not contain a `/`.

    Args:
        handle: The acquisition handle to parse.
        prefix: The prefix to use for the handle.

    Returns:
        The qubit name and tag parsed from the handle. If the handle is not matching
        the expected format, `(None, None)` is returned.

    Example:
        ```python
        handle = "active_reset/q0/0"
        qubit_name, tag = parse_active_reset_handle(handle)
        ```
    """
    if prefix + "/" not in handle:
        return (None, None)
    rest = handle[handle.find(prefix) + len(prefix) + 1 :]
    tag_index = rest.rfind("/")
    if tag_index in {-1, 0, len(rest) - 1}:
        return (None, None)
    tag = rest[tag_index + 1 :]
    qubit_name = rest[:tag_index]
    return qubit_name, tag
