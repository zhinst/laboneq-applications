"""This module provides tasks for collecting several experiment results."""

from __future__ import annotations

from laboneq_applications import workflow
from laboneq_applications.tasks.run_experiment import RunExperimentResults


@workflow.task
def append_result(
    results: list[RunExperimentResults], result: RunExperimentResults
) -> None:
    """Appends result to results.

    Arguments:
        results: list of RunExperimentResults instances
        result: instance of RunExperimentResults to be appended to results
    """
    results.append(result)


@workflow.task
def combine_results(results: list[RunExperimentResults]) -> RunExperimentResults:
    """Combines the results in results into a single RunExperimentResults instance.

    Args:
        results: list of RunExperimentResults instances to be combined into a single
            instance of RunExperimentResults

    Returns: instance of RunExperimentResults with all the data in the individual
        RunExperimentResults instances in results.
    """
    data = {}
    for res in results:
        q_uid = next(iter(res["result"]))
        state = next(iter(res["result"][q_uid]))
        data[f"result/{q_uid}/{state}"] = res["result"][q_uid][state]
    return RunExperimentResults(data=data)
