"""This module provides a task to run a compiled experiment in a session."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from laboneq.dsl.result.results import Results

from laboneq_applications.core.options import (
    create_validate_opts,
)
from laboneq_applications.tasks.datatypes import AcquiredResult, RunExperimentResults
from laboneq_applications.tasks.extract_sweep_results import (
    default_extract_sweep_results,
)
from laboneq_applications.workflow.task import task

if TYPE_CHECKING:
    from laboneq.core.types import CompiledExperiment
    from laboneq.dsl.result.results import Results
    from laboneq.dsl.session import Session


def default_extract_results(results: Results) -> RunExperimentResults:
    """Extract the results from the LabOne Q results.

    Args:
        results: The LabOne Q results to extract the results from.

    Returns:
        The extracted results.

    Example:
        ```python
        from laboneq_library.tasks import default_extract_sweep_results

        results = session.run(compiled_experiment)
        sweep_results = default_extract_sweep_results(results)
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
    options: dict | None = None,
) -> Results:
    """Run the compiled experiment on the quantum processor via the specified session.

    Args:
        session: The connected session to use for running the experiment.
        compiled_experiment: The compiled experiment to run.
        options:
            The options for building the workflow.
            In addition to options from [BaseExperimentOptions], the following
            custom options are supported:
                extractor (optional):
                    Function to extract the data from the LabOne Q results. If None,
                    an instance of the LabOne Q Results class is produced.
                    Default: default_extract_results.
                postprocessor (optional):
                    Function to postprocess the extracted data. If None, no
                    further processing is done on the results.
                    Default: default_extract_sweep_results.

    Returns:
        The measurement results as ...
            ... an instance of the LabOne Q Results class (returned from `Session.run())
            if both the extractor and the postprocessor are None.
            ... an instance of RunExperimentResults if using the default value of the
            extractor.
            ... an instance of SweepResults if using the default value of the
            postporcessor.


    Example:
        ```python
        from laboneq_library.tasks import run_experiment
        from laboneq_library.workflow.engine import Workflow

        with Workflow() as wf:
            run_experiment(
                session=session,
                compiled_experiment=compiled_experiment,
            )
        ```
    """
    # Define the custom options for the experiment
    option_fields = {
        "extractor": (Optional[Callable[[Any], Any]], default_extract_results),
        "postprocessor": (
            Optional[Callable[[Any], Any]],
            default_extract_sweep_results,
        ),
    }
    opts = create_validate_opts(options, option_fields)

    generic_results = session.run(compiled_experiment)
    if opts.extractor is not None:
        generic_results = opts.extractor(generic_results)
    if opts.postprocessor is not None:
        generic_results = opts.postprocessor(generic_results)
    return generic_results
