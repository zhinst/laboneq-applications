# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for the calculate_qubit_population_2d analysis using the testing utilities."""

from pathlib import Path

import numpy as np
import pytest
from laboneq.workflow import handles
from laboneq.workflow.tasks.run_experiment import (
    AcquiredResult,
    RunExperimentResults,
)
from numpy.testing import assert_array_almost_equal

from laboneq_applications.analysis.calibration_traces_rotation import (
    CalculateQubitPopulationOptions,
    calculate_qubit_population_2d,
)

DATA_FOLDER = Path(__file__).parent


@pytest.fixture
def raw_data_q0():
    """Results from a lifetime_measurement experiment.

    In the AcquiredResults below, the axis corresponds to the time-delay after the x180
    pulse in the lifetime_measurement experiment, and the data is the raw acquisition
    result obtained in integrated-average mode.
    """
    data_q0 = np.load(DATA_FOLDER / "iswap_ac0.npy")
    cal_trace_g = np.load(DATA_FOLDER / "iswap_ac0_cal_trace_g.npy")
    cal_trace_e = np.load(DATA_FOLDER / "iswap_ac0_cal_trace_e.npy")

    return data_q0, cal_trace_g, cal_trace_e


@pytest.fixture
def raw_data_q1():
    data_q1 = np.load(DATA_FOLDER / "iswap_ac1.npy")
    cal_trace_g = np.load(DATA_FOLDER / "iswap_ac1_cal_trace_g.npy")
    cal_trace_e = np.load(DATA_FOLDER / "iswap_ac1_cal_trace_e.npy")

    return data_q1, cal_trace_g, cal_trace_e


@pytest.fixture
def sweep_points():
    sweep_points_0 = np.load(DATA_FOLDER / "iswap_durations.npy")
    sweep_points_1 = np.load(DATA_FOLDER / "iswap_amps.npy")

    return sweep_points_0, sweep_points_1


@pytest.fixture
def results_single_qubit_cal_traces(raw_data_q0, sweep_points):
    data_q0, cal_traces_g, cal_traces_e = raw_data_q0
    sp0, sp1 = sweep_points
    data_q0 = {
        handles.result_handle("q0"): AcquiredResult(data=data_q0),
        handles.calibration_trace_handle("q0", "g"): AcquiredResult(data=cal_traces_g),
        handles.calibration_trace_handle("q0", "e"): AcquiredResult(data=cal_traces_e),
    }
    return RunExperimentResults(data_q0), sp0, sp1


@pytest.fixture
def results_single_qubit_no_cal_traces(raw_data_q0, sweep_points):
    data_q0, _, _ = raw_data_q0
    sp0, sp1 = sweep_points
    data_q0 = {
        handles.result_handle("q0"): AcquiredResult(data=data_q0),
    }
    return RunExperimentResults(data_q0), sp0, sp1


@pytest.fixture
def results_single_qubit_one_cal_trace_per_state(raw_data_q0, sweep_points):
    data_q0, cal_traces_g, cal_traces_e = raw_data_q0
    sp0, sp1 = sweep_points
    data_q0 = {
        handles.result_handle("q0"): AcquiredResult(data=data_q0),
        handles.calibration_trace_handle("q0", "g"): AcquiredResult(
            data=cal_traces_g[0]
        ),
        handles.calibration_trace_handle("q0", "e"): AcquiredResult(
            data=cal_traces_e[0]
        ),
    }
    return RunExperimentResults(data_q0), sp0, sp1


@pytest.fixture
def results_two_qubits_cal_traces(raw_data_q0, raw_data_q1, sweep_points):
    data_q0, cal_traces_g_q0, cal_traces_e_q0 = raw_data_q0
    data_q1, cal_traces_g_q1, cal_traces_e_q1 = raw_data_q1
    sp0, sp1 = sweep_points
    data = {
        handles.result_handle("q0"): AcquiredResult(data=data_q0),
        handles.calibration_trace_handle("q0", "g"): AcquiredResult(
            data=cal_traces_g_q0
        ),
        handles.calibration_trace_handle("q0", "e"): AcquiredResult(
            data=cal_traces_e_q0
        ),
        handles.result_handle("q1"): AcquiredResult(data=data_q1),
        handles.calibration_trace_handle("q1", "g"): AcquiredResult(
            data=cal_traces_g_q1
        ),
        handles.calibration_trace_handle("q1", "e"): AcquiredResult(
            data=cal_traces_e_q1
        ),
    }
    return RunExperimentResults(data), sp0, sp1


@pytest.fixture
def results_two_qubits_no_cal_traces(raw_data_q0, raw_data_q1, sweep_points):
    data_q0, _, _ = raw_data_q0
    data_q1, _, _ = raw_data_q1
    sp0, sp1 = sweep_points
    data = {
        handles.result_handle("q0"): AcquiredResult(data=data_q0),
        handles.result_handle("q1"): AcquiredResult(data=data_q1),
    }
    return RunExperimentResults(data), sp0, sp1


@pytest.fixture
def results_two_qubits_one_cal_trace_per_state(raw_data_q0, raw_data_q1, sweep_points):
    data_q0, cal_traces_g_q0, cal_traces_e_q0 = raw_data_q0
    data_q1, cal_traces_g_q1, cal_traces_e_q1 = raw_data_q1
    sp0, sp1 = sweep_points
    data = {
        handles.result_handle("q0"): AcquiredResult(data=data_q0),
        handles.calibration_trace_handle("q0", "g"): AcquiredResult(
            data=cal_traces_g_q0[0]
        ),
        handles.calibration_trace_handle("q0", "e"): AcquiredResult(
            data=cal_traces_e_q0[0]
        ),
        handles.result_handle("q1"): AcquiredResult(data=data_q1),
        handles.calibration_trace_handle("q1", "g"): AcquiredResult(
            data=cal_traces_g_q1[0]
        ),
        handles.calibration_trace_handle("q1", "e"): AcquiredResult(
            data=cal_traces_e_q1[0]
        ),
    }
    return RunExperimentResults(data), sp0, sp1


population_q0_q1_cal_trace_rotation = np.load(
    DATA_FOLDER / "population_cal_trace_rotation.npy"
)

population_q0_q1_cal_traces_pca = np.load(DATA_FOLDER / "population_cal_traces_pca.npy")
population_q0_q1_no_cal_traces = np.load(DATA_FOLDER / "population_no_cal_traces.npy")
population_q0_q1_one_cal_trace_per_state = np.load(
    DATA_FOLDER / "population_one_cal_trace_per_state.npy"
)


### Tests  ###
class TestCalculateQubitPopulation2DSingleQubit:
    def test_single_qubit_cal_traces(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_cal_traces,
    ):
        result, sp1d, sp2d = results_single_qubit_cal_traces
        [q0] = single_tunable_transmon_platform.qpu.qubits
        proc_data_dict = calculate_qubit_population_2d(
            qubits=q0,
            result=result,
            sweep_points_1d=sp1d,
            sweep_points_2d=sp2d,
        )

        assert len(proc_data_dict["q0"]) == 11
        assert proc_data_dict["q0"]["num_cal_traces"] == 2
        assert_array_almost_equal(
            proc_data_dict["q0"]["population_with_cal_traces"],
            population_q0_q1_cal_trace_rotation[0],
            decimal=4,
        )

    def test_single_qubit_no_cal_traces(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_no_cal_traces,
    ):
        result, sp1d, sp2d = results_single_qubit_no_cal_traces
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = CalculateQubitPopulationOptions()
        options.use_cal_traces = False
        proc_data_dict = calculate_qubit_population_2d(
            qubits=q0,
            result=result,
            sweep_points_1d=sp1d,
            sweep_points_2d=sp2d,
            options=options,
        )

        assert len(proc_data_dict["q0"]) == 11
        assert proc_data_dict["q0"]["num_cal_traces"] == 0
        assert_array_almost_equal(
            proc_data_dict["q0"]["population_with_cal_traces"],
            population_q0_q1_no_cal_traces[0],
            decimal=4,
        )

    def test_single_qubit_cal_traces_pca(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_cal_traces,
    ):
        result, sp1d, sp2d = results_single_qubit_cal_traces
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = CalculateQubitPopulationOptions()
        options.do_pca = True
        proc_data_dict = calculate_qubit_population_2d(
            qubits=q0,
            result=result,
            sweep_points_1d=sp1d,
            sweep_points_2d=sp2d,
            options=options,
        )

        assert len(proc_data_dict["q0"]) == 11
        assert proc_data_dict["q0"]["num_cal_traces"] == 2
        assert_array_almost_equal(
            proc_data_dict["q0"]["population_with_cal_traces"],
            population_q0_q1_cal_traces_pca[0],
            decimal=4,
        )

    def test_single_qubit_one_cal_trace_per_state(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_one_cal_trace_per_state,
    ):
        result, sp1d, sp2d = results_single_qubit_one_cal_trace_per_state
        [q0] = single_tunable_transmon_platform.qpu.qubits
        proc_data_dict = calculate_qubit_population_2d(
            qubits=q0,
            result=result,
            sweep_points_1d=sp1d,
            sweep_points_2d=sp2d,
        )

        assert len(proc_data_dict["q0"]) == 11
        assert proc_data_dict["q0"]["num_cal_traces"] == 2
        assert_array_almost_equal(
            proc_data_dict["q0"]["population_with_cal_traces"],
            population_q0_q1_one_cal_trace_per_state[0],
            decimal=4,
        )


class TestCalculateQubitPopulation2DTwoQubit:
    def test_two_qubit_cal_traces(
        self,
        two_tunable_transmon_platform,
        results_two_qubits_cal_traces,
    ):
        result, sp1d, sp2d = results_two_qubits_cal_traces
        [q0, q1] = two_tunable_transmon_platform.qpu.qubits
        proc_data_dict = calculate_qubit_population_2d(
            qubits=[q0, q1],
            result=result,
            sweep_points_1d=[sp1d, sp1d],
            sweep_points_2d=[sp2d, sp2d],
        )

        assert len(proc_data_dict["q0"]) == 11
        assert proc_data_dict["q0"]["num_cal_traces"] == 2
        assert_array_almost_equal(
            proc_data_dict["q0"]["population_with_cal_traces"],
            population_q0_q1_cal_trace_rotation[0],
            decimal=4,
        )

        assert len(proc_data_dict["q1"]) == 11
        assert proc_data_dict["q1"]["num_cal_traces"] == 2
        assert_array_almost_equal(
            proc_data_dict["q1"]["population_with_cal_traces"],
            population_q0_q1_cal_trace_rotation[1],
            decimal=4,
        )

    def test_two_qubit_no_cal_traces(
        self,
        two_tunable_transmon_platform,
        results_two_qubits_no_cal_traces,
    ):
        result, sp1d, sp2d = results_two_qubits_no_cal_traces
        [q0, q1] = two_tunable_transmon_platform.qpu.qubits
        options = CalculateQubitPopulationOptions()
        options.use_cal_traces = False
        proc_data_dict = calculate_qubit_population_2d(
            qubits=[q0, q1],
            result=result,
            sweep_points_1d=[sp1d, sp1d],
            sweep_points_2d=[sp2d, sp2d],
            options=options,
        )

        assert len(proc_data_dict["q0"]) == 11
        assert proc_data_dict["q0"]["num_cal_traces"] == 0
        assert_array_almost_equal(
            proc_data_dict["q0"]["population_with_cal_traces"],
            population_q0_q1_no_cal_traces[0],
            decimal=4,
        )

        assert len(proc_data_dict["q1"]) == 11
        assert proc_data_dict["q1"]["num_cal_traces"] == 0
        assert_array_almost_equal(
            proc_data_dict["q1"]["population_with_cal_traces"],
            population_q0_q1_no_cal_traces[1],
            decimal=4,
        )

    def test_two_qubit_cal_traces_pca(
        self,
        two_tunable_transmon_platform,
        results_two_qubits_cal_traces,
    ):
        result, sp1d, sp2d = results_two_qubits_cal_traces
        [q0, q1] = two_tunable_transmon_platform.qpu.qubits
        options = CalculateQubitPopulationOptions()
        options.do_pca = True
        proc_data_dict = calculate_qubit_population_2d(
            qubits=[q0, q1],
            result=result,
            sweep_points_1d=[sp1d, sp1d],
            sweep_points_2d=[sp2d, sp2d],
            options=options,
        )

        assert len(proc_data_dict["q0"]) == 11
        assert proc_data_dict["q0"]["num_cal_traces"] == 2
        assert_array_almost_equal(
            proc_data_dict["q0"]["population_with_cal_traces"],
            population_q0_q1_cal_traces_pca[0],
            decimal=4,
        )

        assert len(proc_data_dict["q1"]) == 11
        assert proc_data_dict["q1"]["num_cal_traces"] == 2
        assert_array_almost_equal(
            proc_data_dict["q1"]["population_with_cal_traces"],
            population_q0_q1_cal_traces_pca[1],
            decimal=4,
        )

    def test_two_qubit_one_cal_trace_per_state(
        self,
        two_tunable_transmon_platform,
        results_two_qubits_one_cal_trace_per_state,
    ):
        result, sp1d, sp2d = results_two_qubits_one_cal_trace_per_state
        [q0, q1] = two_tunable_transmon_platform.qpu.qubits
        proc_data_dict = calculate_qubit_population_2d(
            qubits=[q0, q1],
            result=result,
            sweep_points_1d=[sp1d, sp1d],
            sweep_points_2d=[sp2d, sp2d],
        )

        assert len(proc_data_dict["q0"]) == 11
        assert proc_data_dict["q0"]["num_cal_traces"] == 2
        assert_array_almost_equal(
            proc_data_dict["q0"]["population_with_cal_traces"],
            population_q0_q1_one_cal_trace_per_state[0],
            decimal=4,
        )

        assert len(proc_data_dict["q1"]) == 11
        assert proc_data_dict["q1"]["num_cal_traces"] == 2
        assert_array_almost_equal(
            proc_data_dict["q1"]["population_with_cal_traces"],
            population_q0_q1_one_cal_trace_per_state[1],
            decimal=4,
        )
