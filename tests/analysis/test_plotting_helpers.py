# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from laboneq.simple import Results
from laboneq.workflow import handles
from laboneq.workflow.tasks.run_experiment import (
    AcquiredResult,
    RunExperimentResults,
)

from laboneq_applications.analysis import plotting_helpers as plt_hlp

rng = np.random.default_rng()


@pytest.fixture
def result_1d():
    """Results generated from random data."""
    data = {}
    data[handles.result_handle("q0")] = AcquiredResult(data=rng.uniform(size=21))
    data[handles.calibration_trace_handle("q0", "g")] = AcquiredResult(
        data=rng.uniform(size=1),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q0", "e")] = AcquiredResult(
        data=rng.uniform(size=1),
        axis_name=[],
        axis=[],
    )

    sweep_points = np.linspace(0, 1, 21)
    return data, sweep_points


@pytest.fixture
def result_1d_nested_two_qubits():
    """Results generated from random data."""
    data = {}
    data[handles.result_handle("q0", suffix="nest")] = AcquiredResult(
        data=rng.uniform(size=21)
    )
    data[handles.calibration_trace_handle("q0", "g")] = AcquiredResult(
        data=rng.uniform(size=1),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q0", "e")] = AcquiredResult(
        data=rng.uniform(size=1),
        axis_name=[],
        axis=[],
    )
    data[handles.result_handle("q1", suffix="nest")] = AcquiredResult(
        data=rng.uniform(size=21)
    )
    data[handles.calibration_trace_handle("q1", "g")] = AcquiredResult(
        data=rng.uniform(size=1),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q1", "e")] = AcquiredResult(
        data=rng.uniform(size=1),
        axis_name=[],
        axis=[],
    )

    sweep_points = np.linspace(0, 1, 21)
    return data, [sweep_points, sweep_points]


class TestRawPlotting1D:
    def test_run_no_cal_traces(self, single_tunable_transmon_platform, result_1d):
        [q0] = single_tunable_transmon_platform.qpu.qubits

        raw_data, sweep_points = result_1d
        result = RunExperimentResults(data=raw_data)
        # plot_raw_complex_data_1d contains is a task that contains a call to
        # save_artifact if options.save_figures == True, and save_artifacts
        # can only be run inside a workflow, so we disable saving figures.
        options = plt_hlp.PlotRawDataOptions()
        options.save_figures = False
        figures = plt_hlp.plot_raw_complex_data_1d(
            qubits=q0,
            result=result,
            sweep_points=sweep_points,
            xlabel="xlabel",
            options=options,
        )

        assert len(figures) == 1
        assert "q0" in figures

    def test_run_with_cal_traces(self, single_tunable_transmon_platform, result_1d):
        [q0] = single_tunable_transmon_platform.qpu.qubits

        raw_data, sweep_points = result_1d
        result = RunExperimentResults(data=raw_data)
        # plot_raw_complex_data_1d contains is a task that contains a call to
        # save_artifact if options.save_figures == True, and save_artifacts
        # can only be run inside a workflow, so we disable saving figures.
        options = plt_hlp.PlotRawDataOptions()
        options.save_figures = False
        options.use_cal_traces = True
        figures = plt_hlp.plot_raw_complex_data_1d(
            qubits=q0,
            result=result,
            sweep_points=sweep_points,
            xlabel="xlabel",
            options=options,
        )

        assert len(figures) == 1
        assert "q0" in figures

    def test_run_nested_two_qubit_no_cal_traces(
        self, two_tunable_transmon_platform, result_1d_nested_two_qubits
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits

        raw_data, sweep_points = result_1d_nested_two_qubits
        result = RunExperimentResults(data=raw_data)
        # plot_raw_complex_data_1d contains is a task that contains a call to
        # save_artifact if options.save_figures == True, and save_artifacts
        # can only be run inside a workflow, so we disable saving figures.
        options = plt_hlp.PlotRawDataOptions()
        options.save_figures = False
        figures = plt_hlp.plot_raw_complex_data_1d(
            qubits=qubits,
            result=result,
            sweep_points=sweep_points,
            xlabel="xlabel",
            options=options,
        )

        assert len(figures) == 2
        assert "q0" in figures
        assert "q1" in figures

    def test_run_nested_two_qubit_with_cal_traces(
        self, two_tunable_transmon_platform, result_1d_nested_two_qubits
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits

        raw_data, sweep_points = result_1d_nested_two_qubits
        result = RunExperimentResults(data=raw_data)
        options = plt_hlp.PlotRawDataOptions()
        options.save_figures = False
        options.use_cal_traces = True
        figures = plt_hlp.plot_raw_complex_data_1d(
            qubits=qubits,
            result=result,
            sweep_points=sweep_points,
            xlabel="xlabel",
            options=options,
        )

        assert len(figures) == 2
        assert "q0" in figures
        assert "q1" in figures

    def test_run_nested_two_qubit_with_cal_traces_legacy_results(
        self, two_tunable_transmon_platform, result_1d_nested_two_qubits
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits

        raw_data, sweep_points = result_1d_nested_two_qubits
        result = Results(acquired_results=raw_data)
        options = plt_hlp.PlotRawDataOptions()
        options.save_figures = False
        options.use_cal_traces = True
        figures = plt_hlp.plot_raw_complex_data_1d(
            qubits=qubits,
            result=result,
            sweep_points=sweep_points,
            xlabel="xlabel",
            options=options,
        )

        assert len(figures) == 2
        assert "q0" in figures
        assert "q1" in figures


@pytest.fixture
def result_2d():
    """Results generated from random data.

    Two-dimensional sweep with 10 sweep points in the first dimension and 7 sweep points
    in the second dimension.
    """
    data = {}
    data[handles.result_handle("q0")] = AcquiredResult(data=rng.uniform(size=(7, 10)))
    data[handles.calibration_trace_handle("q0", "g")] = AcquiredResult(
        data=rng.uniform(size=7),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q0", "e")] = AcquiredResult(
        data=rng.uniform(size=7),
        axis_name=[],
        axis=[],
    )

    sweep_points_1d = np.linspace(0, 1, 10)
    sweep_points_2d = np.arange(0, 7)
    return data, sweep_points_1d, sweep_points_2d


@pytest.fixture
def result_2d_two_cal_points():
    """Results generated from random data.

    Two-dimensional sweep with 10 sweep points in the first dimension and 7 sweep points
    in the second dimension.
    """
    data = {}
    data[handles.result_handle("q0")] = AcquiredResult(data=rng.uniform(size=(7, 10)))
    data[handles.calibration_trace_handle("q0", "g")] = AcquiredResult(
        data=rng.uniform(size=1),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q0", "e")] = AcquiredResult(
        data=rng.uniform(size=1),
        axis_name=[],
        axis=[],
    )

    sweep_points_1d = np.linspace(0, 1, 10)
    sweep_points_2d = np.arange(0, 7)
    return data, sweep_points_1d, sweep_points_2d


@pytest.fixture
def result_2d_nested_two_qubits():
    """Results generated from random data.

    Two-dimensional sweep with 10 sweep points in the first dimension and 7 sweep points
    in the second dimension.
    """
    data = {}
    data[handles.result_handle("q0", suffix="nest")] = AcquiredResult(
        data=rng.uniform(size=(7, 10))
    )
    data[handles.calibration_trace_handle("q0", "g")] = AcquiredResult(
        data=rng.uniform(size=7),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q0", "e")] = AcquiredResult(
        data=rng.uniform(size=7),
        axis_name=[],
        axis=[],
    )

    data[handles.result_handle("q1", suffix="nest")] = AcquiredResult(
        data=rng.uniform(size=(7, 10))
    )
    data[handles.calibration_trace_handle("q1", "g")] = AcquiredResult(
        data=rng.uniform(size=7),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q1", "e")] = AcquiredResult(
        data=rng.uniform(size=7),
        axis_name=[],
        axis=[],
    )

    sweep_points_1d = np.linspace(0, 1, 10)
    sweep_points_2d = np.arange(0, 7)
    return (
        data,
        [sweep_points_1d, sweep_points_1d],
        [sweep_points_2d, sweep_points_2d],
    )


class TestRawPlotting2D:
    def test_run_no_cal_traces(self, single_tunable_transmon_platform, result_2d):
        [q0] = single_tunable_transmon_platform.qpu.qubits

        raw_data, sweep_points_1d, sweep_points_2d = result_2d
        result = RunExperimentResults(data=raw_data)
        options = plt_hlp.PlotRawDataOptions()
        options.save_figures = False
        figures = plt_hlp.plot_raw_complex_data_2d(
            qubits=q0,
            result=result,
            sweep_points_1d=sweep_points_1d,
            sweep_points_2d=sweep_points_2d,
            label_sweep_points_1d="xlabel",
            label_sweep_points_2d="ylabel",
            options=options,
        )

        assert len(figures) == 1
        assert "q0" in figures

    def test_run_with_cal_traces(self, single_tunable_transmon_platform, result_2d):
        [q0] = single_tunable_transmon_platform.qpu.qubits

        raw_data, sweep_points_1d, sweep_points_2d = result_2d
        result = RunExperimentResults(data=raw_data)
        options = plt_hlp.PlotRawDataOptions()
        options.save_figures = False
        options.use_cal_traces = True
        figures = plt_hlp.plot_raw_complex_data_2d(
            q0,
            result=result,
            sweep_points_1d=sweep_points_1d,
            sweep_points_2d=sweep_points_2d,
            label_sweep_points_1d="xlabel",
            label_sweep_points_2d="ylabel",
            options=options,
        )

        assert len(figures) == 1
        assert "q0" in figures

    def test_run_with_cal_traces_2_points(
        self, single_tunable_transmon_platform, result_2d_two_cal_points
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits

        raw_data, sweep_points_1d, sweep_points_2d = result_2d_two_cal_points
        result = RunExperimentResults(data=raw_data)
        options = plt_hlp.PlotRawDataOptions()
        options.save_figures = False
        options.use_cal_traces = True
        figures = plt_hlp.plot_raw_complex_data_2d(
            q0,
            result=result,
            sweep_points_1d=sweep_points_1d,
            sweep_points_2d=sweep_points_2d,
            label_sweep_points_1d="xlabel",
            label_sweep_points_2d="ylabel",
            options=options,
        )

        assert len(figures) == 1
        assert "q0" in figures

    def test_run_nested_two_qubit_no_cal_traces(
        self, two_tunable_transmon_platform, result_2d_nested_two_qubits
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits

        raw_data, sweep_points_1d, sweep_points_2d = result_2d_nested_two_qubits
        result = RunExperimentResults(data=raw_data)
        options = plt_hlp.PlotRawDataOptions()
        options.save_figures = False
        figures = plt_hlp.plot_raw_complex_data_2d(
            qubits,
            result=result,
            sweep_points_1d=sweep_points_1d,
            sweep_points_2d=sweep_points_2d,
            label_sweep_points_1d="xlabel",
            label_sweep_points_2d="ylabel",
            options=options,
        )

        assert len(figures) == 2
        assert len(figures["q0"]) == 1
        assert len(figures["q1"]) == 1

    def test_run_nested_two_qubit_with_cal_traces(
        self, two_tunable_transmon_platform, result_2d_nested_two_qubits
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits

        raw_data, sweep_points_1d, sweep_points_2d = result_2d_nested_two_qubits
        result = RunExperimentResults(data=raw_data)
        options = plt_hlp.PlotRawDataOptions()
        options.save_figures = False
        options.use_cal_traces = True
        figures = plt_hlp.plot_raw_complex_data_2d(
            qubits,
            result=result,
            sweep_points_1d=sweep_points_1d,
            sweep_points_2d=sweep_points_2d,
            label_sweep_points_1d="xlabel",
            label_sweep_points_2d="ylabel",
            options=options,
        )

        assert len(figures) == 2
        assert len(figures["q0"]) == 1
        assert len(figures["q1"]) == 1

    def test_run_nested_two_qubit_with_cal_traces_legacy_results(
        self, two_tunable_transmon_platform, result_2d_nested_two_qubits
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits

        raw_data, sweep_points_1d, sweep_points_2d = result_2d_nested_two_qubits
        result = Results(acquired_results=raw_data)
        options = plt_hlp.PlotRawDataOptions()
        options.save_figures = False
        options.use_cal_traces = True
        figures = plt_hlp.plot_raw_complex_data_2d(
            qubits,
            result=result,
            sweep_points_1d=sweep_points_1d,
            sweep_points_2d=sweep_points_2d,
            label_sweep_points_1d="xlabel",
            label_sweep_points_2d="ylabel",
            options=options,
        )

        assert len(figures) == 2
        assert len(figures["q0"]) == 1
        assert len(figures["q1"]) == 1


class TestPlotSignalMagnitudeAndPhase2D:
    def test_run_single_qubit(self, single_tunable_transmon_platform, result_2d):
        [q0] = single_tunable_transmon_platform.qpu.qubits

        raw_data, sweep_points_1d, sweep_points_2d = result_2d
        result = RunExperimentResults(data=raw_data)
        options = plt_hlp.PlotSignalMagnitudeAndPhase2DOptions()
        options.save_figures = False
        figures = plt_hlp.plot_signal_magnitude_and_phase_2d(
            qubits=q0,
            result=result,
            sweep_points_1d=sweep_points_1d,
            sweep_points_2d=sweep_points_2d,
            label_sweep_points_1d="xlabel",
            label_sweep_points_2d="ylabel",
            scaling_sweep_points_1d=1e-9,
            options=options,
        )

        assert len(figures) == 1
        assert "q0" in figures

    def test_run_nested_two_qubit(
        self, two_tunable_transmon_platform, result_2d_nested_two_qubits
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits

        raw_data, sweep_points_1d, sweep_points_2d = result_2d_nested_two_qubits
        result = RunExperimentResults(data=raw_data)
        options = plt_hlp.PlotSignalMagnitudeAndPhase2DOptions()
        options.save_figures = False
        figures = plt_hlp.plot_signal_magnitude_and_phase_2d(
            qubits,
            result=result,
            sweep_points_1d=sweep_points_1d,
            sweep_points_2d=sweep_points_2d,
            label_sweep_points_1d="xlabel",
            label_sweep_points_2d="ylabel",
            scaling_sweep_points_1d=1e-9,
            options=options,
        )

        assert len(figures) == 2
        assert len(figures["q0"]) == 1
        assert len(figures["q1"]) == 1

    def test_run_two_qubit_with_legacy_results(
        self, two_tunable_transmon_platform, result_2d_nested_two_qubits
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits

        raw_data, sweep_points_1d, sweep_points_2d = result_2d_nested_two_qubits
        result = Results(acquired_results=raw_data)
        options = plt_hlp.PlotSignalMagnitudeAndPhase2DOptions()
        options.save_figures = False
        figures = plt_hlp.plot_signal_magnitude_and_phase_2d(
            qubits,
            result=result,
            sweep_points_1d=sweep_points_1d,
            sweep_points_2d=sweep_points_2d,
            label_sweep_points_1d="xlabel",
            label_sweep_points_2d="ylabel",
            scaling_sweep_points_1d=1e-9,
            options=options,
        )

        assert len(figures) == 2
        assert len(figures["q0"]) == 1
        assert len(figures["q1"]) == 1
