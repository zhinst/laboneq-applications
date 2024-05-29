"""Test the run_experiment task."""

from laboneq_applications.tasks import compile_experiment
from laboneq_applications.workflow.engine import Workflow


def test_run_experiment_standalone(simple_experiment, simple_session):
    """Test that the compile_experiment task compiles the experiment in the session when
    called directly."""
    compiled_exp = compile_experiment(
        session=simple_session,
        experiment=simple_experiment,
    )
    assert compiled_exp.scheduled_experiment is not None


def test_run_experiment_as_task(simple_experiment, simple_session):
    """Test that the compile_experiment task compiles the experiment in the session when
    called as a task."""
    with Workflow() as wf:
        compile_experiment(
            session=simple_session,
            experiment=simple_experiment,
        )
    run = wf.run()
    assert len(run.tasklog) == 1
    assert "compile_experiment" in run.tasklog
    assert run.tasklog["compile_experiment"][0].scheduled_experiment is not None
