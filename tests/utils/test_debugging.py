import json

import numpy as np
import pytest
from laboneq.dsl.result import Results
from laboneq.dsl.result.acquired_result import AcquiredResult, AcquiredResults
from laboneq.simple import DeviceSetup, Experiment, Session

from laboneq_applications.utils.debugging import (
    get_acquired_results_from_results_json,
    mock_acquired_results,
)


def test_mock_acquired_results():
    device_setup = DeviceSetup()
    session = Session(device_setup=device_setup)
    session.connect(do_emulation=True)

    exp = Experiment()
    with exp.acquire_loop_rt(count=10):
        pass

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


def test_get_acquired_results_from_results_json(tmp_path):
    results = Results(
        experiment=123,
        device_setup=DeviceSetup(instruments=[1]),
        acquired_results=AcquiredResults(
            {
                "first": AcquiredResult(data=np.array([1, 2])),
                "second": AcquiredResult(data=np.array([0.2 + 0.3j])),
            },
        ),
    )
    tmp_dir = tmp_path / "test"
    tmp_dir.mkdir()
    filename = tmp_dir / "test.json"
    results.save(filename)
    # Create faulty serialized `Results`
    with open(filename) as f:
        json_data = json.load(f)
        json_data["results"]["compiled_experiment"] = {
            "__type": "CompiledExperiment123",
        }
    with open(filename, "w") as f:
        json.dump(json_data, f)
    # LabOne Q Serializer raises `Exception` due to invalid `__type`
    with pytest.raises(Exception):  # noqa: B017
        Results.load(filename)
    assert get_acquired_results_from_results_json(filename) == results.acquired_results
