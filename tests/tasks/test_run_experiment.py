"""Test the run_experiment task."""

from __future__ import annotations

from textwrap import dedent

import pytest
from laboneq.dsl.experiment import Experiment, ExperimentSignal, pulse_library
from laboneq.dsl.result.results import Results
from laboneq.workflow import workflow

from laboneq_applications.common.attribute_wrapper import AttributeWrapper
from laboneq_applications.tasks import (
    RunExperimentOptions,
    RunExperimentResults,
    run_experiment,
)


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


class PrettyPrinter:
    """A class to test the pretty printer."""

    def __init__(self) -> None:
        self.pretty_text: str | None = None

    def text(self, text: str):
        self.pretty_text = text


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

    def test_run_experiment_standalone_with_legacy(
        self,
        simple_compiled_experiment,
        single_tunable_transmon_platform,
    ):
        """Test that the run_experiment task runs the compiled
        experiment in the session when called directly."""
        options = RunExperimentOptions()
        options.return_legacy_results = True
        results = run_experiment(
            session=single_tunable_transmon_platform.session(do_emulation=True),
            compiled_experiment=simple_compiled_experiment,
            options=options,
        )
        assert isinstance(results, Results)
        assert "ac0" in results.acquired_results

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


def test_run_experiment_results():
    data = {"cal_trace/q0/g": 12345, "cal_trace/q1/g": 2345, "sweep_data/q0": 345}
    neartime_results = {"nt1": 12345, "nt2": {"a": "b", "c": "d"}}
    errors = [(0, "error1", "error1 message"), (1, "error2", "error2 message")]
    results = RunExperimentResults(data, neartime_results, errors)
    assert set(results._keys()) == {
        "cal_trace",
        "sweep_data",
        "neartime_callbacks",
        "errors",
    }
    assert set(results) == {"cal_trace", "sweep_data", "neartime_callbacks", "errors"}
    assert all(
        k in dir(results)
        for k in ["cal_trace", "sweep_data", "neartime_callbacks", "errors"]
    )
    assert results.errors is errors
    assert results["errors"] is errors
    assert isinstance(results.neartime_callbacks, AttributeWrapper)
    assert results["neartime_callbacks"] == results.neartime_callbacks
    assert "nt1" in results.neartime_callbacks
    assert isinstance(results.neartime_callbacks, AttributeWrapper)
    assert results.neartime_callbacks.nt2 == neartime_results["nt2"]
    result_str = dedent("""\
            │   'sweep_data': {
            │   │   'q0': 345
            │   }""")
    q1_str = dedent("""\
            │   │   'q1': {
            │   │   │   'g': 2345
            │   │   }""")
    q0_str = dedent("""\
            │   │   'q0': {
            │   │   │   'g': 12345
            │   │   }""")
    error_str = dedent("""\
            │   'errors': [
            │   │   (
            │   │   │   0,
            │   │   │   'error1',
            │   │   │   'error1 message'
            │   │   ),
            │   │   (
            │   │   │   1,
            │   │   │   'error2',
            │   │   │   'error2 message'
            │   │   )
            │   ]""")
    nt1_str = dedent("""\
            │   │   'nt1': 12345""")
    nt2_str = dedent("""\
            │   │   'nt2': {
            │   │   │   'a': 'b',
            │   │   │   'c': 'd'
            │   │   }""")
    s = str(results)
    assert result_str in s
    assert q1_str in s
    assert q0_str in s
    assert error_str in s
    assert nt1_str in s
    assert nt2_str in s


def test_result_formatting():
    data = {"cal_trace/q0/g": 12345}
    neartime_results = {"nt2": {"a": "b", "c": "d"}}
    errors = [(0, "error1", "error1 message")]
    results = RunExperimentResults(data, neartime_results, errors)

    assert "'cal_trace': {'q0': {'g': 12345}}" in f"{results}"
    assert "'errors': [(0, 'error1', 'error1 message')]" in f"{results}"
    assert "'neartime_callbacks': {'nt2': {'a': 'b', 'c': 'd'}}" in f"{results}"
    assert (
        "'cal_trace': {\n│   │   'q0': {\n│   │   │   'g': 12345\n│   │   }\n│   }"
        in str(results)
    )
    assert (
        "'errors': [\n│   │   (\n│   │   │   0,\n│   │   │   'error1',\n│   │   │   "
        "'error1 message'\n│   │   )\n│   ]" in str(results)
    )
    assert (
        "'neartime_callbacks': {\n│   │   'nt2': {\n│   │   │   'a': 'b',\n│   │   │   "
        "'c': 'd'\n│   │   }\n│   }" in str(results)
    )
    assert (
        repr(results)
        == "RunExperimentResults(data={'cal_trace/q0/g': 12345}, near_time_callbacks="
        "{'nt2': {'a': 'b', 'c': 'd'}}, errors=[(0, 'error1', 'error1 message')], "
        "path = '', separator='/')"
    )
    assert f"{results.cal_trace}" == "{'q0': {'g': 12345}}"
    assert (
        repr(results.cal_trace)
        == "AttributeWrapper(data={'cal_trace/q0/g': 12345}, path='cal_trace', "
        "separator='/')"
    )

    p = PrettyPrinter()
    results._repr_pretty_(p, None)
    assert str(results) == p.pretty_text
    results.cal_trace._repr_pretty_(p, None)
    assert str(results.cal_trace) == p.pretty_text
