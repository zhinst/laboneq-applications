"""This module provides utility functionality for debugging."""
from __future__ import annotations

import json
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Callable

from laboneq.dsl.result import Results
from laboneq.dsl.result.acquired_result import AcquiredResults
from laboneq.dsl.serialization import Serializer

if TYPE_CHECKING:
    from collections.abc import Generator

    from laboneq.dsl.result.acquired_result import AcquiredResult
    from laboneq.dsl.session import Session


def _mock_session_run(func: Callable[..., Results], *args, **kwargs) -> Results:
    mocked_acqr = kwargs.pop("__mock_acquired_results")
    res = func(*args, **kwargs)
    res.acquired_results = mocked_acqr
    return res


@contextmanager
def mock_acquired_results(
    session: Session,
    data: dict[str, AcquiredResult] | AcquiredResults,
) -> Generator[Session]:
    """A context manager to mock the acquired results of LabOne Q.

    Mocks LabOne Q `Session.run()` to always return `Results.acquired_results` with the
    given data.

    Other functionality of the emulated `Session.run()` remains untouched.

    The `run()` method is returned to the original state after exiting the context
    manager.

    Arguments:
        session:
            LabOne Q Session
        data:
            A dictionary of acquired results, where keys are names of the used handles.

    Returns:
        A LabOne Q Session.

    Example:
        >>> from laboneq_applications.utils.debugging import mock_acquired_results
        >>> with mock_acquired_results(
                session, {"q1_acquired_results": AcquiredResult()}
            ) as mock_session:
        >>>     MyExperiment([q1], mock_session, ...)
    """
    orig = session.run
    if not isinstance(data, AcquiredResults):
        data = AcquiredResults(data)
    session.run = partial(_mock_session_run, session.run, __mock_acquired_results=data)
    try:
        yield session
    finally:
        session.run = orig


def get_acquired_results_from_results_json(json_or_file: str | dict) -> AcquiredResults:
    """Get acquired results from a serialized LabOne Q `Results` object.

    This is a fallback mechanism to access the `AcquiredResults` if deserializing
    fails due to LabOne Q version and serialized object incompatibility.

    NOTE: Might break when `AcquiredResults` is changed.

    Arguments:
        json_or_file: A filepath or LabOne Q `Results` object in JSON format.

    Returns:
        Acquired results of the serialized data.
    """
    if isinstance(json_or_file, dict):
        data = json_or_file
    else:
        with open(json_or_file) as f:
            data = json.load(f)["results"]
    for key in data.copy():
        if key not in ("__type", "acquired_results"):
            data[key] = None
    return Serializer.from_json(json.dumps({"results": data}), Results).acquired_results
