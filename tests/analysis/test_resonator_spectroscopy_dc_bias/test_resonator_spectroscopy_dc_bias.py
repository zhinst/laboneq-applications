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

from laboneq_applications.analysis.resonator_spectroscopy_dc_bias import (
    analysis_workflow,
)

DATA_FOLDER = Path(__file__).parent


@pytest.fixture
def results_large_voltage_sweep_range():
    """Results from a resonator-spectroscopy dc-bias sweep experiment."""
    raw_data = np.load(DATA_FOLDER / "large_voltage_sweep_range_raw_data.npy")
    frequencies = np.load(DATA_FOLDER / "large_voltage_sweep_range_frequencies.npy")
    voltages = np.load(DATA_FOLDER / "large_voltage_sweep_range_voltages.npy")

    sp0, sp1 = frequencies, voltages
    data_q0 = {
        handles.result_handle("q0"): AcquiredResult(data=raw_data),
    }
    return RunExperimentResults(data_q0), sp0, sp1


@pytest.fixture
def results_small_voltage_sweep_range():
    """Results from a resonator-spectroscopy dc-bias sweep experiment."""
    raw_data = np.load(DATA_FOLDER / "small_voltage_sweep_range_raw_data.npy")
    frequencies = np.load(DATA_FOLDER / "small_voltage_sweep_range_frequencies.npy")
    voltages = np.load(DATA_FOLDER / "small_voltage_sweep_range_voltages.npy")

    sp0, sp1 = frequencies, voltages
    data_q0 = {
        handles.result_handle("q0"): AcquiredResult(data=raw_data),
    }
    return RunExperimentResults(data_q0), sp0, sp1


### Tests  ###
class TestResonatorSpectroscopyDcBias:
    def test_large_voltage_sweep_range_choose_uss(
        self,
        single_tunable_transmon_platform,
        results_large_voltage_sweep_range,
    ):
        result, frequencies, voltages = results_large_voltage_sweep_range
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = analysis_workflow.options()
        options.find_peaks(True)
        options.parking_sweet_spot("uss")
        workflow_result = analysis_workflow(
            result=result,
            qubit=q0,
            frequencies=frequencies,
            voltages=voltages,
            options=options,
        ).run()

        assert len(workflow_result.tasks) == 7

        proc_data_dict = workflow_result.tasks["process_raw_data"].output
        assert proc_data_dict["sweep_points_1d_filtered"] is None

        # check that the readout-resonator frequencies were extracted correctly from
        # the raw data
        assert_array_almost_equal(
            proc_data_dict["rr_frequencies"],
            np.array(
                [
                    7.0180e09,
                    7.0135e09,
                    7.0130e09,
                    7.0165e09,
                    7.0235e09,
                    7.0325e09,
                    7.0395e09,
                    7.0405e09,
                    7.0335e09,
                    7.0250e09,
                    7.0175e09,
                    7.0130e09,
                    7.0135e09,
                    7.0175e09,
                    7.0250e09,
                ]
            ),
        )

        # check that the fit results were processed correctly
        proc_fit_res = workflow_result.tasks["process_fit_results"].output
        # uss values
        assert len(proc_fit_res["uss"]["frequencies"]) == 1
        np.testing.assert_allclose(
            proc_fit_res["uss"]["rr_frequency_parking"], 7039386889.165501, rtol=1e-4
        )
        np.testing.assert_allclose(
            proc_fit_res["uss"]["dc_voltage_parking"].nominal_value,
            -0.24047659118479459,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            proc_fit_res["uss"]["dc_voltage_parking"].std_dev,
            0.022818039101853343,
            rtol=1e-4,
        )
        # lss values
        assert len(proc_fit_res["lss"]["frequencies"]) == 2
        np.testing.assert_allclose(
            proc_fit_res["lss"]["rr_frequency_parking"], 7011863961.591858, rtol=1e-4
        )
        np.testing.assert_allclose(
            proc_fit_res["lss"]["dc_voltage_parking"].nominal_value,
            2.5824520530674593,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            proc_fit_res["lss"]["dc_voltage_parking"].std_dev,
            0.03249237799014087,
            rtol=1e-4,
        )

        # check that the qubit parameters were extracted correctly
        qubit_parameters = workflow_result.tasks["extract_qubit_parameters"].output
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"][
                "readout_resonator_frequency"
            ],
            7.1e9,
        )
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["dc_voltage_parking"],
            0,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "readout_resonator_frequency"
            ],
            7039386889.165501,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["dc_voltage_parking"],
            -0.24047659118479459,
            rtol=1e-4,
        )

        # check that the plotting tasks were run
        task_names = [t.name for t in workflow_result.tasks]
        assert "plot_raw_complex_data_2d" in task_names
        assert "plot_signal_magnitude" in task_names
        assert "plot_signal_phase" in task_names

    def test_large_voltage_sweep_range_choose_lss(
        self,
        single_tunable_transmon_platform,
        results_large_voltage_sweep_range,
    ):
        result, frequencies, voltages = results_large_voltage_sweep_range
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = analysis_workflow.options()
        options.find_peaks(True)
        options.parking_sweet_spot("lss")
        workflow_result = analysis_workflow(
            result=result,
            qubit=q0,
            frequencies=frequencies,
            voltages=voltages,
            options=options,
        ).run()

        assert len(workflow_result.tasks) == 7

        proc_data_dict = workflow_result.tasks["process_raw_data"].output
        assert proc_data_dict["sweep_points_1d_filtered"] is None
        # check that the readout-resonator frequencies were extracted correctly from
        # the raw data
        assert_array_almost_equal(
            proc_data_dict["rr_frequencies"],
            np.array(
                [
                    7.0180e09,
                    7.0135e09,
                    7.0130e09,
                    7.0165e09,
                    7.0235e09,
                    7.0325e09,
                    7.0395e09,
                    7.0405e09,
                    7.0335e09,
                    7.0250e09,
                    7.0175e09,
                    7.0130e09,
                    7.0135e09,
                    7.0175e09,
                    7.0250e09,
                ]
            ),
        )

        # check that the fit results were processed correctly
        proc_fit_res = workflow_result.tasks["process_fit_results"].output
        # uss values
        assert len(proc_fit_res["uss"]["frequencies"]) == 1
        np.testing.assert_allclose(
            proc_fit_res["uss"]["rr_frequency_parking"], 7039386889.165501, rtol=1e-4
        )
        np.testing.assert_allclose(
            proc_fit_res["uss"]["dc_voltage_parking"].nominal_value,
            -0.24047659118479459,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            proc_fit_res["uss"]["dc_voltage_parking"].std_dev,
            0.022818039101853343,
            rtol=1e-4,
        )
        # lss values
        assert len(proc_fit_res["lss"]["frequencies"]) == 2
        np.testing.assert_allclose(
            proc_fit_res["lss"]["rr_frequency_parking"], 7011863961.591858, rtol=1e-4
        )
        np.testing.assert_allclose(
            proc_fit_res["lss"]["dc_voltage_parking"].nominal_value,
            2.5824520530674593,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            proc_fit_res["lss"]["dc_voltage_parking"].std_dev,
            0.03249237799014087,
            rtol=1e-4,
        )

        # check that the qubit parameters were extracted correctly
        qubit_parameters = workflow_result.tasks["extract_qubit_parameters"].output
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"][
                "readout_resonator_frequency"
            ],
            7.1e9,
        )
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["dc_voltage_parking"],
            0,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "readout_resonator_frequency"
            ],
            7011863961.591858,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["dc_voltage_parking"],
            2.5824520530674593,
            rtol=1e-4,
        )

        # check that the plotting tasks were run
        task_names = [t.name for t in workflow_result.tasks]
        assert "plot_raw_complex_data_2d" in task_names
        assert "plot_signal_magnitude" in task_names
        assert "plot_signal_phase" in task_names

    def test_large_voltage_sweep_range_choose_uss_find_dips(
        self,
        single_tunable_transmon_platform,
        results_large_voltage_sweep_range,
    ):
        result, frequencies, voltages = results_large_voltage_sweep_range
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = analysis_workflow.options()
        options.find_peaks(False)
        options.parking_sweet_spot("uss")
        workflow_result = analysis_workflow(
            result=result,
            qubit=q0,
            frequencies=frequencies,
            voltages=voltages,
            options=options,
        ).run()

        assert len(workflow_result.tasks) == 7

        # check that the readout-resonator frequencies were extracted correctly from
        # the raw data
        assert_array_almost_equal(
            workflow_result.tasks["process_raw_data"].output["rr_frequencies"],
            np.array(
                [
                    7.0395e09,
                    7.0015e09,
                    7.0365e09,
                    7.0035e09,
                    7.0435e09,
                    7.0105e09,
                    7.0125e09,
                    7.0120e09,
                    7.0100e09,
                    7.0440e09,
                    7.0030e09,
                    7.0020e09,
                    7.0015e09,
                    7.0390e09,
                    7.0445e09,
                ]
            ),
        )

        # check that the fit results were processed correctly
        proc_fit_res = workflow_result.tasks["process_fit_results"].output
        # uss values
        assert len(proc_fit_res["uss"]["frequencies"]) == 3
        np.testing.assert_allclose(
            proc_fit_res["uss"]["rr_frequency_parking"], 7035275912.279198, rtol=1e-4
        )
        np.testing.assert_allclose(
            proc_fit_res["uss"]["dc_voltage_parking"].nominal_value,
            1.0527586780537865,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            proc_fit_res["uss"]["dc_voltage_parking"].std_dev,
            0.17827526758873363,
            rtol=1e-4,
        )
        # lss values
        assert len(proc_fit_res["lss"]["frequencies"]) == 3
        np.testing.assert_allclose(
            proc_fit_res["lss"]["rr_frequency_parking"], 7004101850.671261, rtol=1e-4
        )
        np.testing.assert_allclose(
            proc_fit_res["lss"]["dc_voltage_parking"].nominal_value,
            -0.31217301493804234,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            proc_fit_res["lss"]["dc_voltage_parking"].std_dev,
            0.16480258639608947,
            rtol=1e-4,
        )

        # check that the qubit parameters were extracted correctly
        qubit_parameters = workflow_result.tasks["extract_qubit_parameters"].output
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"][
                "readout_resonator_frequency"
            ],
            7.1e9,
        )
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["dc_voltage_parking"],
            0,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "readout_resonator_frequency"
            ],
            7035275912.279198,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["dc_voltage_parking"],
            1.0527586780537865,
            rtol=1e-4,
        )

        # check that the plotting tasks were run
        task_names = [t.name for t in workflow_result.tasks]
        assert "plot_raw_complex_data_2d" in task_names
        assert "plot_signal_magnitude" in task_names
        assert "plot_signal_phase" in task_names

    def test_large_voltage_sweep_range_no_fitting(
        self,
        single_tunable_transmon_platform,
        results_large_voltage_sweep_range,
    ):
        result, frequencies, voltages = results_large_voltage_sweep_range
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = analysis_workflow.options()
        options.find_peaks(True)
        options.parking_sweet_spot("uss")
        options.do_fitting(False)
        workflow_result = analysis_workflow(
            result=result,
            qubit=q0,
            frequencies=frequencies,
            voltages=voltages,
            options=options,
        ).run()

        assert len(workflow_result.tasks) == 7

        # check that the readout-resonator frequencies were extracted correctly from
        # the raw data
        assert_array_almost_equal(
            workflow_result.tasks["process_raw_data"].output["rr_frequencies"],
            np.array(
                [
                    7.0180e09,
                    7.0135e09,
                    7.0130e09,
                    7.0165e09,
                    7.0235e09,
                    7.0325e09,
                    7.0395e09,
                    7.0405e09,
                    7.0335e09,
                    7.0250e09,
                    7.0175e09,
                    7.0130e09,
                    7.0135e09,
                    7.0175e09,
                    7.0250e09,
                ]
            ),
        )

        # check that the fit was not performed
        assert workflow_result.tasks["fit_data"].output is None
        assert len(workflow_result.tasks["process_fit_results"].output) == 0

        # check that the qubit parameters were extracted correctly
        qubit_parameters = workflow_result.tasks["extract_qubit_parameters"].output
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"][
                "readout_resonator_frequency"
            ],
            7.1e9,
        )
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["dc_voltage_parking"],
            0,
        )
        assert len(qubit_parameters["new_parameter_values"]["q0"]) == 0

        # check that the plotting tasks were run
        task_names = [t.name for t in workflow_result.tasks]
        assert "plot_raw_complex_data_2d" in task_names
        assert "plot_signal_magnitude" in task_names
        assert "plot_signal_phase" in task_names

    def test_small_voltage_sweep_range(
        self,
        single_tunable_transmon_platform,
        results_small_voltage_sweep_range,
    ):
        result, frequencies, voltages = results_small_voltage_sweep_range
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = analysis_workflow.options()
        options.find_peaks(True)
        workflow_result = analysis_workflow(
            result=result,
            qubit=q0,
            frequencies=frequencies,
            voltages=voltages,
            options=options,
        ).run()

        assert len(workflow_result.tasks) == 7

        # check that the readout-resonator frequencies were extracted correctly from
        # the raw data
        assert_array_almost_equal(
            workflow_result.tasks["process_raw_data"].output["rr_frequencies"],
            np.array(
                [
                    7.0315e09,
                    7.0355e09,
                    7.0380e09,
                    7.0400e09,
                    7.0415e09,
                    7.0395e09,
                    7.0380e09,
                    7.0350e09,
                    7.0310e09,
                ]
            ),
        )

        # check that the fit results were processed correctly
        proc_fit_res = workflow_result.tasks["process_fit_results"].output
        # uss values
        assert len(proc_fit_res["uss"]["frequencies"]) == 1
        np.testing.assert_allclose(
            proc_fit_res["uss"]["rr_frequency_parking"], 7040805302.761387, rtol=1e-4
        )
        np.testing.assert_allclose(
            proc_fit_res["uss"]["dc_voltage_parking"].nominal_value,
            -0.418733471753031,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            proc_fit_res["uss"]["dc_voltage_parking"].std_dev,
            0.2502150189034397,
            rtol=1e-4,
        )
        # lss values
        assert "lss" not in proc_fit_res

        # check that the qubit parameters were extracted correctly
        qubit_parameters = workflow_result.tasks["extract_qubit_parameters"].output
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"][
                "readout_resonator_frequency"
            ],
            7.1e9,
        )
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["dc_voltage_parking"],
            0,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "readout_resonator_frequency"
            ],
            7040805302.761387,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["dc_voltage_parking"],
            -0.418733471753031,
            rtol=1e-4,
        )

        # check that the plotting tasks were run
        task_names = [t.name for t in workflow_result.tasks]
        assert "plot_raw_complex_data_2d" in task_names
        assert "plot_signal_magnitude" in task_names
        assert "plot_signal_phase" in task_names

    def test_small_voltage_sweep_range_raises_error(
        self,
        single_tunable_transmon_platform,
        results_small_voltage_sweep_range,
    ):
        result, frequencies, voltages = results_small_voltage_sweep_range
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = analysis_workflow.options()
        options.parking_sweet_spot("lss")  # not in the voltage sweep range
        with pytest.raises(ValueError) as err:
            analysis_workflow(
                result=result,
                qubit=q0,
                frequencies=frequencies,
                voltages=voltages,
                options=options,
            ).run()
        error_string = (
            "The lower sweet spot (lss) was not found in the chosen voltage sweep "
            "range. Please set `options.parking_sweet_spot(uss)`."
        )
        assert str(err.value) == error_string

    def test_no_plotting(
        self,
        single_tunable_transmon_platform,
        results_large_voltage_sweep_range,
    ):
        result, frequencies, voltages = results_large_voltage_sweep_range
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = analysis_workflow.options()
        options.find_peaks(True)
        options.parking_sweet_spot("uss")
        options.do_plotting(False)
        workflow_result = analysis_workflow(
            result=result,
            qubit=q0,
            frequencies=frequencies,
            voltages=voltages,
            options=options,
        ).run()

        assert len(workflow_result.tasks) == 4
        task_names = [t.name for t in workflow_result.tasks]
        assert "process_raw_data" in task_names
        assert "fit_data" in task_names
        assert "process_fit_results" in task_names
        assert "extract_qubit_parameters" in task_names
        assert "plot_raw_complex_data_2d" not in task_names
        assert "plot_signal_magnitude" not in task_names
        assert "plot_signal_phase" not in task_names

    def test_no_plotting_raw_data(
        self,
        single_tunable_transmon_platform,
        results_large_voltage_sweep_range,
    ):
        result, frequencies, voltages = results_large_voltage_sweep_range
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = analysis_workflow.options()
        options.find_peaks(True)
        options.parking_sweet_spot("uss")
        options.do_raw_data_plotting(False)
        workflow_result = analysis_workflow(
            result=result,
            qubit=q0,
            frequencies=frequencies,
            voltages=voltages,
            options=options,
        ).run()

        assert len(workflow_result.tasks) == 6
        task_names = [t.name for t in workflow_result.tasks]
        assert "process_raw_data" in task_names
        assert "fit_data" in task_names
        assert "process_fit_results" in task_names
        assert "extract_qubit_parameters" in task_names
        assert "plot_raw_complex_data_2d" not in task_names
        assert "plot_signal_magnitude" in task_names
        assert "plot_signal_phase" in task_names

    def test_no_plotting_magnitude(
        self,
        single_tunable_transmon_platform,
        results_large_voltage_sweep_range,
    ):
        result, frequencies, voltages = results_large_voltage_sweep_range
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = analysis_workflow.options()
        options.find_peaks(True)
        options.parking_sweet_spot("uss")
        options.do_plotting_magnitude(False)
        workflow_result = analysis_workflow(
            result=result,
            qubit=q0,
            frequencies=frequencies,
            voltages=voltages,
            options=options,
        ).run()

        assert len(workflow_result.tasks) == 6
        task_names = [t.name for t in workflow_result.tasks]
        assert "process_raw_data" in task_names
        assert "fit_data" in task_names
        assert "process_fit_results" in task_names
        assert "extract_qubit_parameters" in task_names
        assert "plot_raw_complex_data_2d" in task_names
        assert "plot_signal_magnitude" not in task_names
        assert "plot_signal_phase" in task_names

    def test_no_plotting_phase(
        self,
        single_tunable_transmon_platform,
        results_large_voltage_sweep_range,
    ):
        result, frequencies, voltages = results_large_voltage_sweep_range
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = analysis_workflow.options()
        options.find_peaks(True)
        options.parking_sweet_spot("uss")
        options.do_plotting_phase(False)
        workflow_result = analysis_workflow(
            result=result,
            qubit=q0,
            frequencies=frequencies,
            voltages=voltages,
            options=options,
        ).run()

        assert len(workflow_result.tasks) == 6
        task_names = [t.name for t in workflow_result.tasks]
        assert "process_raw_data" in task_names
        assert "fit_data" in task_names
        assert "process_fit_results" in task_names
        assert "extract_qubit_parameters" in task_names
        assert "plot_raw_complex_data_2d" in task_names
        assert "plot_signal_magnitude" in task_names
        assert "plot_signal_phase" not in task_names

    def test__frequency_filter_upper_limit(
        self,
        single_tunable_transmon_platform,
        results_large_voltage_sweep_range,
    ):
        result, frequencies, voltages = results_large_voltage_sweep_range
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = analysis_workflow.options()
        options.find_peaks(True)
        options.frequency_filter((None, 7.025e9))
        options.parking_sweet_spot("uss")
        workflow_result = analysis_workflow(
            result=result,
            qubit=q0,
            frequencies=frequencies,
            voltages=voltages,
            options=options,
        ).run()

        proc_data_dict = workflow_result.tasks["process_raw_data"].output
        assert len(proc_data_dict["sweep_points_1d_filtered"]) == 49
        # check that the readout-resonator frequencies were extracted correctly from
        # the raw data
        assert_array_almost_equal(
            proc_data_dict["rr_frequencies"],
            np.array(
                [
                    7.0180e09,
                    7.0135e09,
                    7.0130e09,
                    7.0165e09,
                    7.0235e09,
                    7.0245e09,
                    7.0245e09,
                    7.0245e09,
                    7.0245e09,
                    7.0245e09,
                    7.0175e09,
                    7.0130e09,
                    7.0135e09,
                    7.0175e09,
                    7.0245e09,
                ]
            ),
        )

    def test__frequency_filter_lower_limit(
        self,
        single_tunable_transmon_platform,
        results_large_voltage_sweep_range,
    ):
        result, frequencies, voltages = results_large_voltage_sweep_range
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = analysis_workflow.options()
        options.find_peaks(True)
        options.frequency_filter((7.01e9, None))
        options.parking_sweet_spot("uss")
        workflow_result = analysis_workflow(
            result=result,
            qubit=q0,
            frequencies=frequencies,
            voltages=voltages,
            options=options,
        ).run()
        proc_data_dict = workflow_result.tasks["process_raw_data"].output
        assert len(proc_data_dict["sweep_points_1d_filtered"]) == 79
        # check that the readout-resonator frequencies were extracted correctly from
        # the raw data
        assert_array_almost_equal(
            proc_data_dict["rr_frequencies"],
            np.array(
                [
                    7.0180e09,
                    7.0135e09,
                    7.0130e09,
                    7.0165e09,
                    7.0235e09,
                    7.0325e09,
                    7.0395e09,
                    7.0405e09,
                    7.0335e09,
                    7.0250e09,
                    7.0175e09,
                    7.0130e09,
                    7.0135e09,
                    7.0175e09,
                    7.0250e09,
                ]
            ),
        )

    def test__frequency_filter_uppwer_and_lower_limit(
        self,
        single_tunable_transmon_platform,
        results_large_voltage_sweep_range,
    ):
        result, frequencies, voltages = results_large_voltage_sweep_range
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = analysis_workflow.options()
        options.find_peaks(True)
        options.frequency_filter((7.01e9, 7.03e9))
        options.parking_sweet_spot("uss")
        workflow_result = analysis_workflow(
            result=result,
            qubit=q0,
            frequencies=frequencies,
            voltages=voltages,
            options=options,
        ).run()
        proc_data_dict = workflow_result.tasks["process_raw_data"].output
        assert len(proc_data_dict["sweep_points_1d_filtered"]) == 39
        # check that the readout-resonator frequencies were extracted correctly from
        # the raw data
        assert_array_almost_equal(
            proc_data_dict["rr_frequencies"],
            np.array(
                [
                    7.0180e09,
                    7.0135e09,
                    7.0130e09,
                    7.0165e09,
                    7.0235e09,
                    7.0295e09,
                    7.0295e09,
                    7.0295e09,
                    7.0295e09,
                    7.0250e09,
                    7.0175e09,
                    7.0130e09,
                    7.0135e09,
                    7.0175e09,
                    7.0250e09,
                ]
            ),
        )

    def test_frequency_filter_raises_error(
        self,
        single_tunable_transmon_platform,
        results_large_voltage_sweep_range,
    ):
        result, frequencies, voltages = results_large_voltage_sweep_range
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = analysis_workflow.options()

        # lower limit larger than upper limit
        options.frequency_filter((7.03e9, 7.02e9))
        with pytest.raises(ValueError) as err:
            analysis_workflow(
                result=result,
                qubit=q0,
                frequencies=frequencies,
                voltages=voltages,
                options=options,
            ).run()
        error_string = (
            "The first entry in the frequency_filter cannot be larger than "
            "the second entry."
        )
        assert str(err.value) == error_string

        # too many entries
        with pytest.raises(ValueError) as err:
            options.frequency_filter((7.03e9, 7.02e9, 7.05e9))
        error_string = "frequency_filter must have two entries."
        assert str(err.value) == error_string
