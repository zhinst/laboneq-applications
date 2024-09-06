"""Test the compile_experiment task."""

from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING

import pytest
from laboneq.dsl.experiment import Experiment, ExperimentSignal, pulse_library
from laboneq.dsl.experiment.builtins import (
    acquire,
    acquire_loop_rt,
    experiment,
    section,
    sweep,
)

from laboneq_applications.tasks import compile_experiment
from laboneq_applications.tasks.compile_experiment import _validate_handles
from laboneq_applications.workflow import workflow

if TYPE_CHECKING:
    from laboneq_applications.qpu_types.qpu import QuantumPlatform


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


def create_experiment_with_invalid_handles(hq1s: str, hq1r: str, hq2s: str, hq2r: str):
    @experiment(signals=["q1_acquire", "q2_acquire"])
    def exp():
        with acquire_loop_rt(count=4):
            with section():
                acquire(signal="q1_acquire", handle=hq1r, length=100e-9)
            with sweep():
                acquire(signal="q1_acquire", handle=hq1s, length=100e-9)
            with section():
                with section():
                    acquire(signal="q2_acquire", handle=hq2r, length=100e-9)
                    with sweep():
                        acquire(signal="q2_acquire", handle=hq2s, length=100e-9)

    return exp()


@pytest.mark.parametrize(
    ("hq1s", "hq1r", "hq2s", "hq2r", "h1", "h2"),
    [
        ("", "", "", "", None, None),
        ("sweep1", "reset", "sweep2", "reset", None, None),
        ("dat", "data/reset", "sweep2", "reset", None, None),
        ("data", "data/reset", "sweep2", "reset", "data", "data/reset"),
    ],
)
def test_input_validation(
    hq1s: str,
    hq1r: str,
    hq2s: str,
    hq2r: str,
    h1: str | None,
    h2: str | None,
    simple_experiment: Experiment,
):
    if hq1s == "":
        with does_not_raise():
            _validate_handles(simple_experiment)
    else:
        exp = create_experiment_with_invalid_handles(hq1s, hq1r, hq2s, hq2r)
        with (
            does_not_raise()
            if h1 is None
            else pytest.raises(
                ValueError,
                match=f"Handle '{h1}' is a prefix of handle '{h2}', which is not "
                "allowed, because a results entry cannot contain both data and "
                "another results subtree. Please rename one of the handles.",
            )
        ):
            _validate_handles(exp)


def test_input_validation_via_task(single_tunable_transmon_platform: QuantumPlatform):
    simple_session = single_tunable_transmon_platform.session(do_emulation=True)
    exp = create_experiment_with_invalid_handles(
        "data",
        "data/reset",
        "sweep2",
        "reset",
    )

    @workflow
    def wf():
        compile_experiment(session=simple_session, experiment=exp)

    with pytest.raises(
        ValueError,
        match="Invalid input. The following issues were detected: "
        "Handle 'data' is a prefix of handle 'data/reset', which is not "
        "allowed, because a results entry cannot contain both data and "
        "another results subtree. Please rename one of the handles.",
    ):
        wf().run()
