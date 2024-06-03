"""This module provides a task to run a compiled experiment in a session."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq_applications.tasks.datatypes import AcquiredResult, RunExperimentResults
from laboneq_applications.workflow.task import task

if TYPE_CHECKING:
    from laboneq.core.types import CompiledExperiment

    from laboneq_applications.dsl import Results, Session


def extract_results(results: Results) -> RunExperimentResults:
    """Extract the results from the LabOne Q results.

    Args:
        results: The LabOne Q results to extract the results from.

    Returns:
        The extracted results.

    Example:
        ```python
        from laboneq_library.tasks.run_experiment import extract_results

        laboneq_results = session.run(compiled_experiment)
        extracted_results = extract_results(laboneq_results)
        ```
    """
    for h, r in results.acquired_results.items():
        if h != r.handle:
            raise ValueError(
                f"Handle '{h}' does not match the handle '{r.handle}'"
                "in the acquired result.",
            )
    return RunExperimentResults(
        acquired_results={
            h: AcquiredResult(data=r.data, axis=r.axis, axis_name=r.axis_name)
            for h, r in results.acquired_results.items()
        },
        neartime_callback_results=results.neartime_callback_results,
        execution_errors=results.execution_errors,
    )


@task
def run_experiment(
    session: Session,
    compiled_experiment: CompiledExperiment,
    *,
    return_raw_results: bool = False,
    options: dict | None = None,  # pylint: disable=W0613
) -> RunExperimentResults | tuple[RunExperimentResults, Results]:
    """Run the compiled experiment on the quantum processor via the specified session.

    Args:
        session: The connected session to use for running the experiment.
        compiled_experiment: The compiled experiment to run.
        return_raw_results: If true, the raw LabOne Q results are returned in addition.
        options:
            The options for building the workflow.

    Returns:
        The measurement results as ...
            ... an a tuple consisting of an instance of the LabOne Q Results class
                (returned from `Session.run()`) and an instance of
                `RunExperimentResults` if `return_raw_results` is `True`.
            ... an instance of RunExperimentResults if `return_raw_results` is `False`.

    Example:
        ```python
        from laboneq_library.tasks import run_experiment
        from laboneq_library.workflow.engine import Workflow

        with Workflow() as wf:
            results = run_experiment(
                session=session,
                compiled_experiment=compiled_experiment,
            )
        ```
    """
    laboneq_results = session.run(compiled_experiment)
    extracted_results = extract_results(laboneq_results)
    return (
        (extracted_results, laboneq_results)
        if return_raw_results
        else extracted_results
    )
