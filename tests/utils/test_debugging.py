import pytest
import numpy as np
import json

from laboneq.simple import DeviceSetup, Experiment, Session
from laboneq.dsl.result.acquired_result import AcquiredResult, AcquiredResults
from laboneq.dsl.result import Results

from laboneq_library.utils.debugging import mock_acquired_results, get_acquired_results_from_results_json


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


def test_get_acquired_results_from_results_json(tmp_path):
    results = Results(
        experiment=123,
        device_setup=DeviceSetup(instruments=[1]),
        acquired_results=AcquiredResults(
            {
                "first": AcquiredResult(data=np.array([1, 2])),
                "second": AcquiredResult(data=np.array([0.2+0.3j]))
            }
        )
    )
    tmp_dir = tmp_path / "test"
    tmp_dir.mkdir()
    filename = tmp_dir / "test.json"
    results.save(filename)
    # Create faulty serialized `Results`
    with open(filename, "r") as f:
        json_data = json.load(f)
        json_data["results"]["compiled_experiment"] = {"__type": "CompiledExperiment123"}
    with open(filename, "w") as f:
        json.dump(json_data, f)
    # LabOne Q Serializer raises `Exception` due to invalid `__type`
    with pytest.raises(Exception):
        Results.load(filename)
    assert get_acquired_results_from_results_json(filename) == results.acquired_results
