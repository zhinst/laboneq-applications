"""Test the run_experiment task."""

import pytest
from laboneq.dsl.experiment import Experiment, ExperimentSignal, pulse_library
from laboneq.dsl.result.results import Results

from laboneq_applications.tasks import RunExperimentResults, run_experiment
from laboneq_applications.workflow.engine import workflow


@pytest.fixture()
def simple_compiled_experiment(single_tunable_transmon_platform):
    device_setup = single_tunable_transmon_platform.setup
    session = single_tunable_transmon_platform.session(do_emulation=True)

    exp = Experiment(
        signals=[
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )
    exp.map_signal("measure", device_setup.logical_signal_by_uid("q0/measure"))
    exp.map_signal("acquire", device_setup.logical_signal_by_uid("q0/acquire"))

    with exp.acquire_loop_rt(count=4):
        with exp.section():
            exp.play("measure", pulse_library.const())
            exp.acquire("acquire", "ac0", length=100e-9)
            exp.delay("measure", 100e-9)

    return session.compile(exp)


class TestRunExperiment:
    """Test the run_experiment task."""

    def test_run_experiment_standalone(
        self,
        simple_compiled_experiment,
        single_tunable_transmon_platform,
    ):
        """Test that the run_experiment task runs the compiled
        experiment in the session when called directly."""
        results = run_experiment(
            session=single_tunable_transmon_platform.session(do_emulation=True),
            compiled_experiment=simple_compiled_experiment,
        )
        # Then the session should run the compiled experiment
        assert isinstance(results, RunExperimentResults)
        assert "ac0" in results

    def test_run_experiment_standalone_with_raw(
        self,
        simple_compiled_experiment,
        single_tunable_transmon_platform,
    ):
        """Test that the run_experiment task runs the compiled
        experiment in the session when called directly."""
        results = run_experiment(
            session=single_tunable_transmon_platform.session(do_emulation=True),
            compiled_experiment=simple_compiled_experiment,
            return_raw_results=True,
        )
        # Then the session should run the compiled experiment
        assert isinstance(results[0], RunExperimentResults)
        assert isinstance(results[1], Results)
        assert "ac0" in results[0]
        assert "ac0" in results[1].acquired_results

    def test_run_experiment_as_task(
        self,
        simple_compiled_experiment,
        single_tunable_transmon_platform,
    ):
        """Test that the run_experiment task runs the compiled
        experiment in the session when called as a task."""

        @workflow
        def wff():
            run_experiment(
                session=single_tunable_transmon_platform.session(do_emulation=True),
                compiled_experiment=simple_compiled_experiment,
            )

        result = wff().run()
        assert len(result.tasks) == 1
        assert "ac0" in result.tasks["run_experiment"].output
