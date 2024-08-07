"""Test the run_experiment task."""

import pytest
from laboneq.dsl.experiment import Experiment, ExperimentSignal, pulse_library

from laboneq_applications.tasks import compile_experiment
from laboneq_applications.workflow.engine import workflow


@pytest.fixture()
def simple_experiment(single_tunable_transmon_platform):
    device_setup = single_tunable_transmon_platform.setup

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

    return exp


def test_compile_experiment_standalone(
    simple_experiment,
    single_tunable_transmon_platform,
):
    """Test that the compile_experiment task compiles the experiment in the session when
    called directly."""
    compiled_exp = compile_experiment(
        session=single_tunable_transmon_platform.session(do_emulation=True),
        experiment=simple_experiment,
    )
    assert compiled_exp.scheduled_experiment is not None


def test_compile_experiment_as_task(
    simple_experiment,
    single_tunable_transmon_platform,
):
    """Test that the compile_experiment task compiles the experiment in the session when
    called as a task."""

    @workflow
    def wf():
        compile_experiment(
            session=single_tunable_transmon_platform.session(do_emulation=True),
            experiment=simple_experiment,
        )

    run = wf().run()
    assert len(run.tasks) == 1
    assert run.tasks["compile_experiment"].output.scheduled_experiment is not None
