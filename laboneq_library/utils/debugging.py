"""This module provides utility functionality for debugging."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generator
from contextlib import contextmanager
from functools import partial
import json

from laboneq.dsl.session import Session
from laboneq.dsl.serialization import Serializer
from laboneq.dsl.result import Results
from laboneq.dsl.result.acquired_result import AcquiredResults

if TYPE_CHECKING:
    from laboneq.dsl.result.acquired_result import AcquiredResult


def _mock_session_run(func: Callable[..., Results], *args, **kwargs):
    mocked_acqr = kwargs.pop("__mock_acquired_results")
    res = func(*args, **kwargs)
    res.acquired_results = mocked_acqr
    return res


@contextmanager
def mock_acquired_results(session: Session, data: dict[str, AcquiredResult] | AcquiredResults) -> Generator[Session]:
    """A context manager to mock the acquired results of LabOne Q.

    Mocks LabOne Q `Session.run()` to always return `Results.acquired_results` with the given data.
    Other functionality of the emulated `Session.run()` remains untouched.

    The `run()` method is returned to the original state after exiting the context manager.

    Args:
        session: LabOne Q Session
        data: A dictionary of acquired results, where keys are names of the used handles.

    Returns:
        A LabOne Q Session.

    Example:

        >>> import from laboneq_library.utils.debugging import mock_acquired_results
        >>> with mock_acquired_results(session, {"q1_acquired_results": AcquiredResult()}) as mock_session:
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
