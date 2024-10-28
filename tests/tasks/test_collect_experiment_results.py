import numpy as np
import pytest

from laboneq_applications.core.handles import result_handle
from laboneq_applications.tasks import append_result, combine_results
from laboneq_applications.tasks.run_experiment import (
    AcquiredResult,
    RunExperimentResults,
)


@pytest.fixture()
def results_list():
    """A list of RunExperimentResults."""
    data_list = [
        {
            result_handle("q0", suffix="g"): AcquiredResult(np.linspace(0, 1, 10)),
        },
        {
            result_handle("q0", suffix="e"): AcquiredResult(np.linspace(0, 2, 10)),
        },
        {
            result_handle("q0", suffix="f"): AcquiredResult(np.linspace(0, 3, 10)),
        },
    ]
    return [RunExperimentResults(data=data) for data in data_list]


def test_append_results(results_list):
    result_to_append = RunExperimentResults(
        data={result_handle("q1", suffix="f"): AcquiredResult(np.linspace(0, 4, 10))}
    )
    append_result(results_list, result_to_append)
    assert len(results_list) == 4
    assert result_handle("q1", suffix="f") in results_list[-1]
    np.testing.assert_array_equal(
        results_list[-1][result_handle("q1", suffix="f")].data, np.linspace(0, 4, 10)
    )


def test_combine_results(results_list):
    combined_result = combine_results(results_list)
    assert result_handle("q0", suffix="g") in combined_result
    np.testing.assert_array_equal(
        combined_result[result_handle("q0", suffix="g")].data, np.linspace(0, 1, 10)
    )

    assert result_handle("q0", suffix="e") in combined_result
    np.testing.assert_array_equal(
        combined_result[result_handle("q0", suffix="e")].data, np.linspace(0, 2, 10)
    )

    assert result_handle("q0", suffix="f") in combined_result
    np.testing.assert_array_equal(
        combined_result[result_handle("q0", suffix="f")].data, np.linspace(0, 3, 10)
    )
