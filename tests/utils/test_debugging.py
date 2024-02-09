import pytest
import numpy as np
import json

from laboneq.simple import DeviceSetup, Experiment, Session
from laboneq.dsl.result.acquired_result import AcquiredResult, AcquiredResults
from laboneq.dsl.result import Results

from laboneq_library.utils.debugging import mock_acquired_results


def test_mock_acquired_results():
    device_setup = DeviceSetup()
    exp = Experiment()
    session = Session(device_setup=device_setup)
    session.connect(do_emulation=True)

    # Test initial state
    res = session.run(experiment=exp)
    assert res.acquired_results == {}

    # Test mock state
    with mock_acquired_results(session, {"foo": 123}) as mock_session:
        res = mock_session.run(experiment=exp)
    assert res.acquired_results == AcquiredResults({"foo": 123})

    with mock_acquired_results(session, AcquiredResults({"foo": 123})) as mock_session:
        res = mock_session.run(experiment=exp)
    assert res.acquired_results == AcquiredResults({"foo": 123})

    # Test back to initial state
    res = session.run(experiment=exp)
    assert res.acquired_results == {}
