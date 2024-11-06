# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for the lifetime_measurement analysis using the testing utilities."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from test_lifetime_measurement_data import (  # noqa: F401
    population_q0_q1_no_pca,
    population_q0_q1_pca,
    raw_data_ge,
    results_single_qubit_ge,
    results_two_qubit_ge,
)

from laboneq_applications.analysis import lifetime_measurement


class TestLifetimeMeasurementAnalysisSingleQubit:
    def test_create_and_run_no_pca(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_ge,  # noqa: F811
        population_q0_q1_no_pca,  # noqa: F811
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = lifetime_measurement.analysis_workflow.options()
        options.do_pca(False)
        result = lifetime_measurement.analysis_workflow(
            result=results_single_qubit_ge[0],
            qubits=q0,
            delays=results_single_qubit_ge[1],
            options=options,
        ).run()

        assert len(result.tasks) == 5

        for task in result.tasks:
            if task.name != "extract_qubit_parameters":
                assert "q0" in task.output
        assert "new_parameter_values" in result.tasks["extract_qubit_parameters"].output

        proc_data_dict = result.tasks["calculate_qubit_population"].output
        assert proc_data_dict["q0"]["num_cal_traces"] == 2
        assert_array_almost_equal(
            proc_data_dict["q0"]["population"], population_q0_q1_no_pca[0]
        )

        fit_values = result.tasks["fit_data"].output["q0"].best_values
        np.testing.assert_allclose(
            fit_values["decay_rate"], 65645.08548221787, rtol=1e-4
        )
        np.testing.assert_allclose(
            fit_values["amplitude"], 0.971258379771985, rtol=1e-4
        )
        assert_almost_equal(fit_values["offset"], 0)

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["ge_T1"],
            0.0,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["ge_T1"].nominal_value,
            1.5233432825232331e-05,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["ge_T1"].std_dev,
            1.7538784708046533e-07,
            rtol=1e-4,
        )

    def test_create_and_run_pca(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_ge,  # noqa: F811
        population_q0_q1_pca,  # noqa: F811
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = lifetime_measurement.analysis_workflow.options()
        options.do_pca(True)
        result = lifetime_measurement.analysis_workflow(
            result=results_single_qubit_ge[0],
            qubits=q0,
            delays=results_single_qubit_ge[1],
            options=options,
        ).run()

        assert len(result.tasks) == 5

        for task in result.tasks:
            if task.name not in [
                "validate_and_convert_qubits_sweeps",
                "extract_qubit_parameters",
            ]:
                assert "q0" in task.output
        assert "new_parameter_values" in result.tasks["extract_qubit_parameters"].output

        proc_data_dict = result.tasks["calculate_qubit_population"].output
        assert proc_data_dict["q0"]["num_cal_traces"] == 2
        assert_array_almost_equal(
            proc_data_dict["q0"]["population"], population_q0_q1_pca[0]
        )

        fit_values = result.tasks["fit_data"].output["q0"].best_values
        np.testing.assert_allclose(
            fit_values["decay_rate"], 71133.73792785374, rtol=1e-4
        )
        np.testing.assert_allclose(
            fit_values["amplitude"], 1.2354133959247915, rtol=1e-4
        )
        np.testing.assert_allclose(
            fit_values["offset"], -0.36344008061260663, rtol=1e-4
        )

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["ge_T1"],
            0.0,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["ge_T1"].nominal_value,
            1.4058026881902847e-05,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["ge_T1"].std_dev,
            2.38004046315109e-07,
            rtol=1e-4,
        )

    def test_create_and_run_no_fitting(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_ge,  # noqa: F811
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = lifetime_measurement.analysis_workflow.options()
        options.do_fitting(False)

        result = lifetime_measurement.analysis_workflow(
            result=results_single_qubit_ge[0],
            qubits=q0,
            delays=results_single_qubit_ge[1],
            options=options,
        ).run()

        assert len(result.tasks) == 5

        assert result.tasks["fit_data"].output == {}
        qb_params = result.tasks["extract_qubit_parameters"]
        assert len(qb_params.output["old_parameter_values"]["q0"]) > 0
        assert len(qb_params.output["new_parameter_values"]["q0"]) == 0

    def test_create_and_run_no_plotting(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_ge,  # noqa: F811
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = lifetime_measurement.analysis_workflow.options()
        options.do_plotting(False)

        result = lifetime_measurement.analysis_workflow(
            result=results_single_qubit_ge[0],
            qubits=q0,
            delays=results_single_qubit_ge[1],
            options=options,
        ).run()

        assert len(result.tasks) == 3

        task_names = [t.name for t in result.tasks]
        assert "plot_raw_complex_data_1d" not in task_names
        assert "plot_population" not in task_names
        assert "fit_data" in task_names
        assert "extract_qubit_parameters" in task_names


class TestLifetimeMeasurementAnalysisTwoQubit:
    def test_create_and_run_no_pca(
        self,
        two_tunable_transmon_platform,
        results_two_qubit_ge,  # noqa: F811
        population_q0_q1_no_pca,  # noqa: F811
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = lifetime_measurement.analysis_workflow.options()
        options.do_pca(False)
        result = lifetime_measurement.analysis_workflow(
            result=results_two_qubit_ge[0],
            qubits=qubits,
            delays=results_two_qubit_ge[1],
            options=options,
        ).run()

        assert len(result.tasks) == 5

        for task in result.tasks:
            if task.name != "extract_qubit_parameters":
                assert "q0" in task.output
                assert "q1" in task.output
        assert "new_parameter_values" in result.tasks["extract_qubit_parameters"].output

        proc_data_dict = result.tasks["calculate_qubit_population"].output
        assert proc_data_dict["q0"]["num_cal_traces"] == 2
        assert proc_data_dict["q1"]["num_cal_traces"] == 2
        assert_array_almost_equal(
            proc_data_dict["q0"]["population"], population_q0_q1_no_pca[0]
        )
        assert_array_almost_equal(
            proc_data_dict["q1"]["population"], population_q0_q1_no_pca[1]
        )

        fit_values = result.tasks["fit_data"].output["q0"].best_values
        np.testing.assert_allclose(
            fit_values["decay_rate"], 65645.08548221787, rtol=1e-4
        )
        np.testing.assert_allclose(
            fit_values["amplitude"], 0.971258379771985, rtol=1e-4
        )
        assert_almost_equal(fit_values["offset"], 0)
        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["ge_T1"],
            0.0,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["ge_T1"].nominal_value,
            1.5233432825232331e-05,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["ge_T1"].std_dev,
            1.7538784708046533e-07,
            rtol=1e-4,
        )

        fit_values = result.tasks["fit_data"].output["q1"].best_values
        np.testing.assert_allclose(
            fit_values["decay_rate"],
            34619.52793115473,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            fit_values["amplitude"],
            0.9989747821396329,
            rtol=1e-4,
        )
        assert_almost_equal(fit_values["offset"], 0)
        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q1"]["ge_T1"],
            0.0,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q1"]["ge_T1"].nominal_value,
            2.8885431424386415e-05,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q1"]["ge_T1"].std_dev,
            2.2661395978112074e-07,
            rtol=1e-4,
        )

    def test_create_and_run_pca(
        self,
        two_tunable_transmon_platform,
        results_two_qubit_ge,  # noqa: F811
        population_q0_q1_pca,  # noqa: F811
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = lifetime_measurement.analysis_workflow.options()
        options.do_pca(True)

        result = lifetime_measurement.analysis_workflow(
            result=results_two_qubit_ge[0],
            qubits=qubits,
            delays=results_two_qubit_ge[1],
            options=options,
        ).run()

        assert len(result.tasks) == 5

        for task in result.tasks:
            if task.name not in [
                "validate_and_convert_qubits_sweeps",
                "extract_qubit_parameters",
            ]:
                assert "q0" in task.output
        assert "new_parameter_values" in result.tasks["extract_qubit_parameters"].output

        proc_data_dict = result.tasks["calculate_qubit_population"].output
        assert proc_data_dict["q0"]["num_cal_traces"] == 2
        assert proc_data_dict["q1"]["num_cal_traces"] == 2
        assert_array_almost_equal(
            proc_data_dict["q0"]["population"], population_q0_q1_pca[0]
        )
        assert_array_almost_equal(
            proc_data_dict["q1"]["population"], population_q0_q1_pca[1]
        )

        fit_values = result.tasks["fit_data"].output["q0"].best_values
        np.testing.assert_allclose(
            fit_values["decay_rate"], 71133.73792785374, rtol=1e-4
        )
        np.testing.assert_allclose(
            fit_values["amplitude"], 1.2354133959247915, rtol=1e-4
        )
        np.testing.assert_allclose(
            fit_values["offset"], -0.36344008061260663, rtol=1e-4
        )

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["ge_T1"],
            0.0,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["ge_T1"].nominal_value,
            1.4058026881902847e-05,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["ge_T1"].std_dev,
            2.38004046315109e-07,
            rtol=1e-4,
        )

        fit_values = result.tasks["fit_data"].output["q1"].best_values
        np.testing.assert_allclose(
            fit_values["decay_rate"], 35717.784188456186, rtol=1e-4
        )
        np.testing.assert_allclose(
            fit_values["amplitude"], 1.8684762419939445, rtol=1e-4
        )
        np.testing.assert_allclose(fit_values["offset"], -0.8806252822168226, rtol=1e-4)

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q1"]["ge_T1"],
            0.0,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q1"]["ge_T1"].nominal_value,
            2.799725746490162e-05,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q1"]["ge_T1"].std_dev,
            8.758721477575979e-07,
            rtol=1e-4,
        )

    def test_create_and_run_no_fitting(
        self,
        two_tunable_transmon_platform,
        results_two_qubit_ge,  # noqa: F811
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = lifetime_measurement.analysis_workflow.options()
        options.do_fitting(False)

        result = lifetime_measurement.analysis_workflow(
            result=results_two_qubit_ge[0],
            qubits=qubits,
            delays=results_two_qubit_ge[1],
            options=options,
        ).run()

        assert len(result.tasks) == 5

        assert result.tasks["fit_data"].output == {}
        qb_params = result.tasks["extract_qubit_parameters"]
        assert len(qb_params.output["old_parameter_values"]["q0"]) > 0
        assert len(qb_params.output["new_parameter_values"]["q0"]) == 0
        assert len(qb_params.output["old_parameter_values"]["q1"]) > 0
        assert len(qb_params.output["new_parameter_values"]["q1"]) == 0

    def test_create_and_run_no_plotting(
        self,
        two_tunable_transmon_platform,
        results_two_qubit_ge,  # noqa: F811
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = lifetime_measurement.analysis_workflow.options()
        options.do_plotting(False)

        result = lifetime_measurement.analysis_workflow(
            result=results_two_qubit_ge[0],
            qubits=qubits,
            delays=results_two_qubit_ge[1],
            options=options,
        ).run()

        assert len(result.tasks) == 3

        task_names = [t.name for t in result.tasks]
        assert "plot_raw_complex_data_1d" not in task_names
        assert "plot_population" not in task_names
        assert "fit_data" in task_names
        assert "extract_qubit_parameters" in task_names

    def test_create_and_run_close_figures(
        self,
        two_tunable_transmon_platform,
        results_two_qubit_ge,  # noqa: F811
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = lifetime_measurement.analysis_workflow.options()
        options.close_figures(True)

        result = lifetime_measurement.analysis_workflow(
            result=results_two_qubit_ge[0],
            qubits=qubits,
            delays=results_two_qubit_ge[1],
            options=options,
        ).run()

        assert isinstance(
            result.tasks["plot_population"].output["q0"], mpl.figure.Figure
        )
        assert isinstance(
            result.tasks["plot_population"].output["q1"], mpl.figure.Figure
        )

        options.close_figures(False)
        result = lifetime_measurement.analysis_workflow(
            result=results_two_qubit_ge[0],
            qubits=qubits,
            delays=results_two_qubit_ge[1],
            options=options,
        ).run()

        assert isinstance(
            result.tasks["plot_population"].output["q0"], mpl.figure.Figure
        )
        assert isinstance(
            result.tasks["plot_population"].output["q1"], mpl.figure.Figure
        )
        plt.close(result.tasks["plot_population"].output["q0"])
        plt.close(result.tasks["plot_population"].output["q1"])
