"""Test the default sweep extraction task."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest
from laboneq.dsl.experiment import pulse_library
from laboneq.dsl.experiment.builtins import (
    acquire_loop_rt,
    case,
    delay,
    experiment,
    map_signal,
    match,
    measure,
    play,
    section,
    sweep,
)
from laboneq.dsl.parameter import LinearSweepParameter
from laboneq.dsl.session import Session

from laboneq_applications.tasks.datatypes import AcquiredResult, RunExperimentResults
from laboneq_applications.tasks.extract_sweep_results import (
    default_extract_sweep_results,
)
from laboneq_applications.tasks.run_experiment import (
    default_extract_results,
)
from tests.helpers.device_setups import two_tunable_transmon_setup


def test_extract_sweep_results_example():
    """Test the example in the docstring."""
    # pylint: disable=import-outside-toplevel
    # pylint: disable=redefined-outer-name
    # pylint: disable=reimported
    import numpy as np

    from laboneq_applications.tasks.datatypes import (
        AcquiredResult,
        RunExperimentResults,
    )
    from laboneq_applications.tasks.extract_sweep_results import (
        default_extract_sweep_results,
    )

    results = RunExperimentResults()
    results.acquired_results["my_result/qubit_1"] = AcquiredResult(
        data=[1, 2, 3, 4, 5],
        axis_name=["amplitude"],
        axis=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    results.acquired_results["my_result/qubit_2"] = AcquiredResult(
        data=[2, 2, 3, 4, 5],
        axis_name=["amplitude"],
        axis=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    )
    results.acquired_results["cal_trace/qubit_1/g"] = AcquiredResult(
        data=[2, 3, 4, 5, 6],
        axis_name=["amplitude_ct"],
        axis=[0.15, 0.2, 0.3, 0.4, 0.5],
    )
    results.acquired_results["active_reset/qubit_1/1"] = AcquiredResult(
        data=[4, 3, 4, 5, 6],
        axis_name=["amplitude"],
        axis=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    )
    sweep_results = default_extract_sweep_results(results, result_prefix="my_result")

    np.testing.assert_array_equal(
        sweep_results.sweep_data("qubit_1"),
        ([0.1, 0.2, 0.3, 0.4, 0.5], [1, 2, 3, 4, 5]),
    )
    np.testing.assert_array_equal(
        sweep_results.sweep_data("qubit_2"),
        ([0.1, 0.2, 0.3, 0.4, 0.5], [2, 2, 3, 4, 5]),
    )
    np.testing.assert_array_equal(
        sweep_results.calibration_traces("qubit_1", "g"),
        [2, 3, 4, 5, 6],
    )
    np.testing.assert_array_equal(
        sweep_results.active_reset_data("qubit_1", "1"),
        ([0.1, 0.2, 0.3, 0.4, 0.5], [4, 3, 4, 5, 6]),
    )
    np.testing.assert_array_equal(sweep_results.sweep_points, [0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_array_equal(
        sweep_results.sweep_points_calibration_traces,
        [0.15, 0.2, 0.3, 0.4, 0.5],
    )
    assert sweep_results.sweep_parameter_names == ["amplitude"]
    assert sweep_results.sweep_parameter_names_calibration_traces == ["amplitude_ct"]


MY_ACTIVE_RESET_FORMAT = "actif_reset/{qubit_name}/{repetition}"
MY_CALIBRATION_TRACE_FORMAT = "calib_trace/{qubit_name}/{state_name}"
MY_SWEEP_FORMAT = "resultat/{qubit_name}"


def rabi_experiment_builder(
    device_setup,
    *,
    has_cal_traces=True,
    has_sweep=True,
    sweep_count=10,
):
    """Create a complex Rabi experiment."""

    @experiment(
        signals=[
            "drive_q0",
            "measure_q0",
            "acquire_q0",
            "drive_q1",
            "measure_q1",
            "acquire_q1",
        ],
    )
    def _exp():
        for q in ["q0", "q1"]:
            lsg = device_setup.logical_signal_groups[q].logical_signals
            map_signal(f"drive_{q}", lsg["drive"])
            map_signal(f"measure_{q}", lsg["measure"])
            map_signal(f"acquire_{q}", lsg["acquire"])

        num_avg_exp = 10
        measure_pulse = pulse_library.const(length=200e-9)  # pylint: disable=no-value-for-parameter
        x180 = pulse_library.gaussian(amp=0.5, length=100e-9, sigma=20e-9)  # pylint: disable=no-value-for-parameter
        amplitude_sweep = LinearSweepParameter(
            uid="rabi_amp",
            start=0.1,
            stop=1.0,
            count=sweep_count,
        )

        def my_measure(qubit_name: str, handle: str) -> None:
            measure(
                acquire_signal=f"acquire_{qubit_name}",
                handle=handle,
                integration_length=100e-9,
                measure_signal=f"measure_{qubit_name}",
                measure_pulse=measure_pulse,
            )

        with acquire_loop_rt(name="shots", count=2**num_avg_exp):
            if has_sweep:
                with sweep(name="rabi_sweep", parameter=amplitude_sweep):
                    for q in ["q0", "q1"]:
                        with section(name=f"reset_{q}_1") as reset_section:
                            reset_handle = MY_ACTIVE_RESET_FORMAT.format(
                                qubit_name=q,
                                repetition=0,
                            )
                            my_measure(qubit_name=q, handle=reset_handle)
                        with match(
                            name=f"match_reset_{q}_1",
                            handle=reset_handle,
                            play_after=reset_section,
                        ):
                            with case(0):
                                pass
                            with case(1):
                                play(f"drive_{q}", x180)
                        with section(
                            name=f"qubit_excitation_{q}",
                        ) as excitation_section:
                            play(f"drive_{q}", x180, amplitude=amplitude_sweep)
                            delay(signal=f"drive_{q}", time=100e-9)
                        with section(
                            name=f"measure_{q}",
                            play_after=excitation_section,
                        ):
                            my_measure(
                                qubit_name=q,
                                handle=MY_SWEEP_FORMAT.format(qubit_name=q),
                            )
            if has_cal_traces:
                for q in ["q0", "q1"]:
                    with section(name=f"reset_{q}_2") as reset_section:
                        reset_handle = MY_ACTIVE_RESET_FORMAT.format(
                            qubit_name=q,
                            repetition=1,
                        )
                        my_measure(qubit_name=q, handle=reset_handle)
                    with match(
                        name=f"match_reset_{q}_2",
                        handle=reset_handle,
                        play_after=reset_section,
                    ) as match_section:
                        with case(0):
                            pass
                        with case(1):
                            play(f"drive_{q}", x180)
                    with section(
                        name=f"cal_trace_{q}_g",
                        play_after=match_section,
                    ) as cal_trace_g:
                        my_measure(
                            qubit_name=q,
                            handle=MY_CALIBRATION_TRACE_FORMAT.format(
                                qubit_name=q,
                                state_name="g",
                            ),
                        )
                    with section(
                        name=f"cal_trace_{q}_e_prep",
                        play_after=cal_trace_g,
                    ) as cal_trace_e_prep:
                        play(f"drive_{q}", x180, amplitude=1.0)
                    with section(name=f"cal_trace_{q}_e", play_after=cal_trace_e_prep):
                        my_measure(
                            qubit_name=q,
                            handle=MY_CALIBRATION_TRACE_FORMAT.format(
                                qubit_name=q,
                                state_name="e",
                            ),
                        )

    return _exp()


def test_extract_sweep_results_from_experiment():
    ttt_setup = two_tunable_transmon_setup()
    exp = rabi_experiment_builder(ttt_setup)
    session = Session(ttt_setup)
    session.connect(do_emulation=True)
    compiled_experiment = session.compile(exp)
    result = session.run(compiled_experiment)
    extracted = default_extract_sweep_results(
        default_extract_results(result),
        result_prefix="resultat",
        calibration_trace_prefix="calib_trace",
        active_reset_prefix="actif_reset",
    )
    assert np.array_equal(
        extracted.sweep_points,
        result.acquired_results["resultat/q0"].axis,
    )
    assert np.array_equal(
        extracted.sweep_data("q0")[0],
        result.acquired_results["resultat/q0"].axis,
    )
    assert np.array_equal(
        extracted.sweep_data("q0")[1],
        result.acquired_results["resultat/q0"].data,
    )
    assert np.array_equal(
        extracted.sweep_data("q1")[0],
        result.acquired_results["resultat/q1"].axis,
    )
    assert np.array_equal(
        extracted.sweep_data("q1")[1],
        result.acquired_results["resultat/q1"].data,
    )
    assert np.array_equal(
        extracted.sweep_points_calibration_traces,
        result.acquired_results["calib_trace/q0/g"].axis,
    )
    assert np.array_equal(
        extracted.calibration_traces("q0", "g"),
        result.acquired_results["calib_trace/q0/g"].data,
    )
    assert np.array_equal(
        extracted.calibration_traces("q0", "e"),
        result.acquired_results["calib_trace/q0/e"].data,
    )
    assert np.array_equal(
        extracted.calibration_traces("q0", ["e", "g"]),
        [
            result.acquired_results["calib_trace/q0/e"].data,
            result.acquired_results["calib_trace/q0/g"].data,
        ],
    )
    assert np.array_equal(
        extracted.calibration_traces("q1", "g"),
        result.acquired_results["calib_trace/q1/g"].data,
    )
    assert np.array_equal(
        extracted.active_reset_data("q0", "0")[0],
        result.acquired_results["resultat/q0"].axis,
    )
    assert np.array_equal(
        extracted.active_reset_data("q0", "0")[1],
        result.acquired_results["actif_reset/q0/0"].data,
    )
    assert np.array_equal(
        extracted.active_reset_data("q0", ["0", "1"])[0][0],
        result.acquired_results["resultat/q0"].axis,
    )
    assert np.array_equal(
        extracted.active_reset_data("q0", ["0", "1"])[0][1],
        result.acquired_results["actif_reset/q0/0"].data,
    )
    assert np.array_equal(
        extracted.active_reset_data("q0", ["0", "1"])[1][0],
        result.acquired_results["resultat/q0"].axis,
    )
    assert np.array_equal(
        extracted.active_reset_data("q0", ["0", "1"])[1][1],
        result.acquired_results["actif_reset/q0/1"].data,
    )
    assert np.array_equal(
        extracted.active_reset_data("q0", "1")[1],
        result.acquired_results["actif_reset/q0/1"].data,
    )
    assert np.array_equal(
        extracted.active_reset_data("q1", "0")[1],
        result.acquired_results["actif_reset/q1/0"].data,
    )
    assert np.array_equal(
        extracted.active_reset_data("q1", "1")[1],
        result.acquired_results["actif_reset/q1/1"].data,
    )
    assert extracted.calibration_traces("q0", "f") is None
    assert extracted.calibration_traces("q2", "g") is None
    assert extracted.sweep_parameter_names == ["rabi_amp"]
    assert extracted.sweep_data("ThisQubitDoesNotExist") == (None, None)
    assert extracted.active_reset_data("q0", "ThisTagDoesNotExist") is None
    assert (
        extracted.active_reset_data("q0", ["0", "ThisTagDoesNotExist"])[0] is not None
    )
    assert extracted.active_reset_data("q0", ["0", "ThisTagDoesNotExist"])[1] is None


def test_extract_sweep_results_from_experiment_no_caltraces():
    ttt_setup = two_tunable_transmon_setup()
    exp = rabi_experiment_builder(ttt_setup, has_cal_traces=False)
    session = Session(ttt_setup)
    session.connect(do_emulation=True)
    compiled_experiment = session.compile(exp)
    result = session.run(compiled_experiment)
    extracted = default_extract_sweep_results(
        default_extract_results(result),
        result_prefix="resultat",
        calibration_trace_prefix="calib_trace",
        active_reset_prefix="actif_reset",
    )
    points, data = extracted.sweep_data("q0", append_calibration_traces=True)
    assert np.allclose(points, [np.arange(0.1, 1.1, 0.1)])
    assert np.allclose(data, result.acquired_results["resultat/q0"].data)
    resets = extracted.active_reset_data("q0", "0", append_calibration_traces=True)
    assert len(resets) == 2
    assert np.allclose(resets[0], [np.arange(0.1, 1.1, 0.1)])
    assert np.allclose(resets[1], result.acquired_results["actif_reset/q0/0"].data)


def test_extract_sweep_results_from_experiment_no_sweep():
    ttt_setup = two_tunable_transmon_setup()
    exp = rabi_experiment_builder(ttt_setup, has_sweep=False)
    session = Session(ttt_setup)
    session.connect(do_emulation=True)
    compiled_experiment = session.compile(exp)
    result = session.run(compiled_experiment)
    extracted = default_extract_sweep_results(
        default_extract_results(result),
        result_prefix="resultat",
        calibration_trace_prefix="calib_trace",
        active_reset_prefix="actif_reset",
    )
    points, data = extracted.sweep_data("q0", append_calibration_traces=True)
    assert np.allclose(points, [0, 1])
    assert np.allclose(
        data,
        np.concatenate(
            [
                [result.acquired_results["calib_trace/q0/g"].data],
                [result.acquired_results["calib_trace/q0/e"].data],
            ],
        ),
    )
    resets = extracted.active_reset_data("q0", "0", append_calibration_traces=True)
    assert resets is None
    resets = extracted.active_reset_data("q0", "1", append_calibration_traces=True)
    assert len(resets) == 2
    assert np.allclose(resets[0], [0, 1])
    assert np.allclose(
        resets[1],
        np.concatenate(
            [
                [result.acquired_results["actif_reset/q0/1"].data],
                [result.acquired_results["calib_trace/q0/g"].data],
                [result.acquired_results["calib_trace/q0/e"].data],
            ],
        ),
    )


def test_extract_sweep_results_with_caltraces():
    ttt_setup = two_tunable_transmon_setup()
    exp = rabi_experiment_builder(ttt_setup)
    session = Session(ttt_setup)
    session.connect(do_emulation=True)
    compiled_experiment = session.compile(exp)
    result = session.run(compiled_experiment)
    extracted = default_extract_sweep_results(
        default_extract_results(result),
        result_prefix="resultat",
        calibration_trace_prefix="calib_trace",
        active_reset_prefix="actif_reset",
    )
    points, data = extracted.sweep_data("q0", append_calibration_traces=True)
    assert np.allclose(points, [np.arange(0.1, 1.3, 0.1)])
    assert np.allclose(
        data,
        np.concatenate(
            [
                result.acquired_results["resultat/q0"].data,
                [result.acquired_results["calib_trace/q0/g"].data],
                [result.acquired_results["calib_trace/q0/e"].data],
            ],
        ),
    )
    resets = extracted.active_reset_data("q0", "0", append_calibration_traces=True)
    assert len(resets) == 2
    assert np.allclose(resets[0], [np.arange(0.1, 1.3, 0.1)])
    assert np.allclose(
        resets[1],
        np.concatenate(
            [
                result.acquired_results["actif_reset/q0/0"].data,
                [result.acquired_results["calib_trace/q0/g"].data],
                [result.acquired_results["calib_trace/q0/e"].data],
            ],
        ),
    )

    resets = extracted.active_reset_data(
        "q0",
        ["0", "1"],
        append_calibration_traces=True,
    )

    assert np.allclose(resets[1][0], [np.arange(0.1, 1.3, 0.1)])
    assert np.allclose(
        resets[1][1],
        np.concatenate(
            [
                [result.acquired_results["actif_reset/q0/1"].data],
                [result.acquired_results["calib_trace/q0/g"].data],
                [result.acquired_results["calib_trace/q0/e"].data],
            ],
        ),
    )


def test_extract_trivial_sweep_results_with_caltraces():
    ttt_setup = two_tunable_transmon_setup()
    exp = rabi_experiment_builder(ttt_setup, sweep_count=1)
    session = Session(ttt_setup)
    session.connect(do_emulation=True)
    compiled_experiment = session.compile(exp)
    result = session.run(compiled_experiment)
    extracted = default_extract_sweep_results(
        default_extract_results(result),
        result_prefix="resultat",
        calibration_trace_prefix="calib_trace",
        active_reset_prefix="actif_reset",
    )
    points, data = extracted.sweep_data("q0", append_calibration_traces=True)
    assert np.allclose(points, [[0.1, 1.1, 2.1]])
    assert np.allclose(
        data,
        np.concatenate(
            [
                result.acquired_results["resultat/q0"].data,
                [result.acquired_results["calib_trace/q0/g"].data],
                [result.acquired_results["calib_trace/q0/e"].data],
            ],
        ),
    )
    resets = extracted.active_reset_data("q0", "0", append_calibration_traces=True)
    assert len(resets) == 2
    assert np.allclose(resets[0], [[0.1, 1.1, 2.1]])
    assert np.allclose(
        resets[1],
        np.concatenate(
            [
                result.acquired_results["actif_reset/q0/0"].data,
                [result.acquired_results["calib_trace/q0/g"].data],
                [result.acquired_results["calib_trace/q0/e"].data],
            ],
        ),
    )

    resets = extracted.active_reset_data(
        "q0",
        ["0", "1"],
        append_calibration_traces=True,
    )

    assert np.allclose(resets[1][0], [[0.1, 1.1, 2.1]])
    assert np.allclose(
        resets[1][1],
        np.concatenate(
            [
                [result.acquired_results["actif_reset/q0/1"].data],
                [result.acquired_results["calib_trace/q0/g"].data],
                [result.acquired_results["calib_trace/q0/e"].data],
            ],
        ),
    )


def test_wrong_sweep_points():
    """Test that the function raises an error if the sweep points aren't as expected."""

    results = RunExperimentResults()
    results.acquired_results["my_result/qubit_1"] = AcquiredResult(
        data=[1, 2, 3, 4, 5],
        axis_name=["amplitude", "phase"],
        axis=[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]],
    )
    results.acquired_results["cal_trace/qubit_1/g"] = AcquiredResult(
        data=[2],
        axis_name=["amplitude_ct"],
        axis=[],
    )
    extracted = default_extract_sweep_results(results, result_prefix="my_result")
    with pytest.raises(
        ValueError,
        match="Sweep points are not a one dimensional list of numbers.",
    ):
        extracted.sweep_data("qubit_1", append_calibration_traces=True)

    results = RunExperimentResults()
    results.acquired_results["my_result/qubit_1"] = AcquiredResult(
        data=[1, 2, 3, 4, 5],
        axis_name=["amplitude"],
        axis=[0.1, 0.2, 0.3, 0.4, 0.6],
    )
    results.acquired_results["my_result/qubit_2"] = AcquiredResult(
        data=[2, 2, 3, 4, 5],
        axis_name=["amplitude"],
        axis=[0.2, 0.2, 0.3, 0.4, 0.6],
    )
    with pytest.raises(
        ValueError,
        match="Sweep points are not consistent for handle 'my_result/qubit_2'.",
    ):
        default_extract_sweep_results(results, result_prefix="my_result")
