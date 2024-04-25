"""Test the run_experiment task."""

import pytest
from laboneq.dsl.calibration import Oscillator, SignalCalibration
from laboneq.dsl.device import DeviceSetup
from laboneq.dsl.experiment import Experiment, pulse_library
from laboneq.dsl.session import Session

from laboneq_applications.tasks import run_experiment
from laboneq_applications.workflow.workflow import Workflow
from tests.helpers.device_setups import single_tunable_transmon_setup


@pytest.fixture()
def device_setup():
    device_setup = single_tunable_transmon_setup()
    device_setup.logical_signal_by_uid("q0/measure").calibration = SignalCalibration(
        local_oscillator=Oscillator(frequency=5e9),
        oscillator=Oscillator(frequency=100e6),
    )
    device_setup.logical_signal_by_uid("q0/acquire").calibration = SignalCalibration(
        local_oscillator=Oscillator(frequency=5e9),
        oscillator=Oscillator(frequency=100e6),
    )
    return device_setup


@pytest.fixture()
def simple_experiment(device_setup):
    exp = Experiment(signals=["measure", "acquire"])
    exp.map_signal("measure", device_setup.logical_signal_by_uid("q0/measure"))
    exp.map_signal("acquire", device_setup.logical_signal_by_uid("q0/acquire"))
    with exp.acquire_loop_rt(count=4):
        with exp.section():
            exp.play("measure", pulse_library.const())
            exp.acquire("acquire", "ac0", length=100e-9)
            exp.delay("measure", 100e-9)
    return exp


class TestRunExperiment:
    @pytest.fixture(autouse=True)
    def _setup(self, device_setup: DeviceSetup, simple_experiment: Experiment):
        self.session = Session(device_setup)
        self.session.connect(do_emulation=True)
        self.compiled_experiment = self.session.compile(simple_experiment)

    def test_run_experiment_standalone(self):
        """Test that the run_experiment task runs the compiled
        experiment in the session when called directly."""

        results = run_experiment(
            session=self.session,
            compiled_experiment=self.compiled_experiment,
        )
        # Then the session should run the compiled experiment
        assert "ac0" in results.acquired_results

    def test_run_experiment_as_task(self):
        """Test that the run_experiment task runs the compiled
        experiment in the session when called as a task."""

        with Workflow() as wf:
            run_experiment(
                session=self.session,
                compiled_experiment=self.compiled_experiment,
            )
        assert len(wf.run().tasklog) == 1
        [results] = wf.run().tasklog["run_experiment"]
        assert "ac0" in results.acquired_results
