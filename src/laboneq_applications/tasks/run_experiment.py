"""This module provides a task to run a compiled experiment in a session."""

from laboneq.core.types import CompiledExperiment
from laboneq.dsl.result.results import Results
from laboneq.dsl.session import Session

from laboneq_applications.workflow.task import task


@task
def run_experiment(
    session: Session,
    compiled_experiment: CompiledExperiment,
) -> Results:
    """Run the compiled experiment on the quantum processor via the specified session.

    Args:
        session: The connected session to use for running the experiment.
        compiled_experiment: The compiled experiment to run.

    Returns:
        The result of the LabOne Q Session.run() call.

    Example:
    ```python
    from laboneq_library.tasks import run_experiment
    from laboneq_library.workflow.workflow import Workflow

    with Workflow() as wf:
        run_experiment(
            session=session,
            compiled_experiment=compiled_experiment,
        )
    ```
    """
    return session.run(compiled_experiment)
