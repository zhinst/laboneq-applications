"""Test the run_experiment task."""

import pytest
from laboneq.dsl.experiment import Experiment

from laboneq_applications.tasks import run_experiment
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
        assert results.single_qubit_data == {}

    def test_run_experiment_with_empty_options(self, simple_session):
        """Test that the run_experiment task runs the compiled
        experiment in the session when called directly with empty options."""

        results = run_experiment(
            session=simple_session,
            compiled_experiment=self.compiled_experiment,
            options={},
        )
        # Then the session should run the compiled experiment
        assert results.single_qubit_data == {}

    def test_run_experiment_with_options_none(self, simple_session):
        """Test that the run_experiment task runs the compiled
        experiment in the session when called directly with empty options."""

        results = run_experiment(
            session=simple_session,
            compiled_experiment=self.compiled_experiment,
            options={"extractor": None, "postprocessor": None},
        )
        # Then the session should run the compiled experiment
        assert "ac0" in results.acquired_results

    def test_run_experiment_with_options_not_none(self, simple_session):
        """Test that the run_experiment task runs the compiled
        experiment in the session when called directly with empty options."""

        results = run_experiment(
            session=simple_session,
            compiled_experiment=self.compiled_experiment,
            options={"extractor": lambda x: x, "postprocessor": lambda x: x},
        )
        # Then the session should run the compiled experiment
        assert "ac0" in results.acquired_results

    def test_run_experiment_as_task(self, simple_session):
        """Test that the run_experiment task runs the compiled
        experiment in the session when called as a task."""

        with Workflow() as wf:
            run_experiment(
                session=simple_session,
                compiled_experiment=self.compiled_experiment,
                options={"extractor": lambda x: x, "postprocessor": lambda x: x},
            )
        assert len(wf.run().tasklog) == 1
        [results] = wf.run().tasklog["run_experiment"]
        assert "ac0" in results.acquired_results
