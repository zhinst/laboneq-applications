"""This module provides a task to transform a DSL experiment into a compiled experiment."""  # noqa: E501

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.core.exceptions import LabOneQException

if TYPE_CHECKING:
    from laboneq.core.types import CompiledExperiment
    from laboneq.dsl.experiment import Experiment
    from laboneq.dsl.session import Session

from laboneq_applications.workflow.task import task


@task
def compile_experiment(
    session: Session,
    experiment: Experiment,
    compiler_settings: dict | None = None,
) -> CompiledExperiment:
    """Workflow task to compile the specified experiment for a given setup.

    This task is used to prepare a LabOne Q DSL experiment for execution on a quantum
    processor. It will return the results of a LabOneQ Session.compile() call.

    Args:
        session:
            A calibrated session to compile the experiment for.
        experiment:
            The LabOne Q DSL experiment to compile.
        compiler_settings:
            Optional settings to pass to the compiler.

    Returns:
        The compiled experiment.

    Example:
        >>> from laboneq_library.tasks.compile_experiment import compile_experiment
        >>> from laboneq_library.workflow.workflow import Workflow
        >>> with Workflow() as wf:
        >>>   compiled_experiment = compile_experiment(session, experiment)
    """
    compiled_experiment = session.compile(
        experiment=experiment,
        compiler_settings=compiler_settings,
    )
    if compiled_experiment is None:
        msg = "Failed to compile experiment."
        raise LabOneQException(msg)
    return compiled_experiment
