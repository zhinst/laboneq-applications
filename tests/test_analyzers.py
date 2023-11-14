import numpy as np
from laboneq.dsl.result import AcquiredResult, AcquiredResults
from laboneq.simple import Results

from laboneq_library.automatic_tuneup.tuneup.analyzer import (
    AnalyzeData,
    AnalyzeDatum,
    Lorentzian,
    MockAnalyzer,
    RabiAnalyzer,
)


def test_preprocess_result():
    res1 = AcquiredResult(
        data=np.array([[1, 2], [2, 1]]),
        axis_name=["param1", "param2"],
        axis=[[0, 1], [1, 0]],
    )
    res2 = AcquiredResult(data=np.array([3, 4]), axis_name=["param1"], axis=[[0, 1]])
    results = Results(acquired_results=AcquiredResults(h1=res1, h2=res2))

    analyzer_one_handle = MockAnalyzer()
    res = analyzer_one_handle._preprocess_result(results)
    assert isinstance(res, AnalyzeData)
    assert len(res) == 1
    assert res["h1"].x == res1.axis
    assert np.array_equal(res["h1"].y, res1.data)

    analyzer_two_handles = MockAnalyzer(handles=["h1", "h2"])
    res = analyzer_two_handles._preprocess_result(results)
    assert isinstance(res, AnalyzeData)
    assert len(res) == 2
    assert res["h1"].x == res1.axis
    assert np.array_equal(res["h1"].y, res1.data)
    assert res["h2"].x == res2.axis
    assert np.array_equal(res["h2"].y, res2.data)

    update_res1 = AnalyzeDatum(name="h1", x=res1.axis, y=res1.data)
    update_res2 = AnalyzeDatum(name="h2", x=res2.axis, y=res2.data)
    results = AnalyzeData(h1=update_res1, h2=update_res2)
    res = analyzer_two_handles._preprocess_result(results)
    assert isinstance(res, AnalyzeData)
    assert len(res) == 2
    assert res["h1"].x == res1.axis
    assert np.array_equal(res["h1"].y, res1.data)
    assert res["h2"].x == res2.axis
    assert np.array_equal(res["h2"].y, res2.data)


def test_analyzer_default():
    res1 = AcquiredResult(
        data=np.array([[1, 2], [2, 1]]),
        axis_name=["param1", "param2"],
        axis=[[0, 1], [1, 0]],
    )
    res2 = AcquiredResult(data=np.array([3, 4]), axis_name=["param1"], axis=[[0, 1]])
    results = Results(acquired_results=AcquiredResults(h1=res1, h2=res2))

    analyzer = MockAnalyzer(handles=["h1", "h2"])
    assert analyzer._result == AnalyzeData()
    assert analyzer.analyze(results) == 1234
    assert analyzer._result == AnalyzeData(
        h1=AnalyzeDatum(name="h1", x=res1.axis, y=res1.data),
        h2=AnalyzeDatum(name="h2", x=res2.axis, y=res2.data),
    )
    assert analyzer.verify(1234) is True


def test_lorentzian_analyzer():
    x = np.linspace(-10, 10, 100)
    f0 = 0
    offset = 0.1
    amplitude = 1
    gamma = 0.1
    flip = 1
    y = offset + flip * amplitude * gamma**2 / (gamma * 2 + (x - f0) ** 2)

    res1 = AcquiredResult(data=y, axis_name=["qspec"], axis=[x])
    res2 = AcquiredResult(data=np.array([3, 4]), axis_name=["param1"], axis=[[0, 1]])
    results = Results(acquired_results=AcquiredResults(qspec=res1, dummy=res2))
    analyzer = Lorentzian(
        truth=f0,
        handles=["qspec"],
        tolerance=5 * gamma,
        f0=f0,
        a=amplitude,
        gamma=gamma,
        offset=offset,
    )
    analyzed_result = analyzer.analyze(results)
    assert analyzer.verify(analyzed_result)


def test_rabi_analyzer():
    x = np.linspace(0, 5, 50)
    amplitude = 1
    frequency = 0.5
    phase = 0.0
    offset = 0.01
    y = offset + amplitude * np.sin(2 * np.pi * frequency * x + phase)

    res1 = AcquiredResult(data=y, axis_name=["q0_measure"], axis=[x])
    res2 = AcquiredResult(data=np.array([3, 4]), axis_name=["param1"], axis=[[0, 1]])
    results = Results(acquired_results=AcquiredResults(q0_measure=res1, dummy=res2))
    analyzer = RabiAnalyzer(
        truth=1 / frequency / 2,
        handles=["q0_measure"],
        tolerance=0.1,
        amp_pi=1 / frequency / 2,
        phase=0.5,
        offset=0.1,
        rotate=False,
        real=False,
    )
    analyzed_result = analyzer.analyze(results)
    assert analyzer.verify(analyzed_result)
