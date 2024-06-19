"""Test the run_experiment task."""

import pytest
from laboneq.dsl.experiment import Experiment
from laboneq.dsl.result.results import Results

from laboneq_applications.tasks import RunExperimentResults, run_experiment
from laboneq_applications.workflow.engine import Workflow


class TestRunExperiment:
    """Test the run_experiment task."""

    @pytest.fixture(autouse=True)
    def _setup(self, simple_session, simple_experiment: Experiment):
        # pylint: disable=attribute-defined-outside-init
        self.compiled_experiment = simple_session.compile(simple_experiment)

    def test_run_experiment_standalone(self, simple_session):
        """Test that the run_experiment task runs the compiled
        experiment in the session when called directly."""

        results = run_experiment(
            session=simple_session,
            compiled_experiment=self.compiled_experiment,
        )
        # Then the session should run the compiled experiment
        assert isinstance(results, RunExperimentResults)
        assert "ac0" in results

    def test_run_experiment_standalone_with_raw(self, simple_session):
        """Test that the run_experiment task runs the compiled
        experiment in the session when called directly."""

        results = run_experiment(
            session=simple_session,
            compiled_experiment=self.compiled_experiment,
            return_raw_results=True,
        )
        # Then the session should run the compiled experiment
        assert isinstance(results[0], RunExperimentResults)
        assert isinstance(results[1], Results)
        assert "ac0" in results[0]
        assert "ac0" in results[1].acquired_results

    def test_run_experiment_as_task(self, simple_session):
        """Test that the run_experiment task runs the compiled
        experiment in the session when called as a task."""

        with Workflow() as wf:
            run_experiment(
                session=simple_session,
                compiled_experiment=self.compiled_experiment,
            )
        assert len(wf.run().tasklog) == 1
        [results] = wf.run().tasklog["run_experiment"]
        assert "ac0" in results
