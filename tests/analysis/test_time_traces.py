# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for the IQ-Blobs analysis using the testing utilities."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from laboneq.workflow import handles
from test_time_traces_data import (  # noqa: F401
    discrimination_thresholds_ge,
    discrimination_thresholds_gef,
    integration_kernels_filtered_gef,
    integration_kernels_ge,
    integration_kernels_gef,
    raw_data,
    results_single_qubit_g,
    results_single_qubit_ge,
    results_single_qubit_gef,
)

from laboneq_applications.analysis import time_traces


class TestTimeTracesAnalysisSingleQubitGE:
    def test_create_and_run(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_ge,  # noqa: F811
        integration_kernels_ge,  # noqa: F811
        discrimination_thresholds_ge,  # noqa: F811
    ):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        options = time_traces.analysis_workflow.options()
        options.granularity(16)
        states = ["g", "e"]
        result = time_traces.analysis_workflow(
            result=results_single_qubit_ge,
            qubits=q0,
            states=states,
            options=options,
        ).run()

        assert len(result.tasks) == 6
        for task in result.tasks:
            if task.name not in [
                "extract_kernels_thresholds",
                "extract_qubit_parameters",
            ]:
                assert "q0" in task.output
        task_names = [t.name for t in result.tasks]
        assert "filter_integration_kernels" not in task_names
        assert len(result.tasks["truncate_time_traces"].output["q0"]) == 2
        data_e = results_single_qubit_ge[handles.result_handle("q0", suffix="e")].data
        assert len(result.tasks["truncate_time_traces"].output["q0"][0]) == len(
            data_e[: (len(data_e) // 16) * 16]
        )

        qubit_parameters = result.tasks["extract_qubit_parameters"].output

        old_qb_pars = qubit_parameters["old_parameter_values"]["q0"]
        assert old_qb_pars["readout_integration_kernels_type"] == "default"
        assert old_qb_pars["readout_integration_kernels"] is None
        assert old_qb_pars["readout_integration_discrimination_thresholds"] is None

        new_qb_pars = qubit_parameters["new_parameter_values"]["q0"]
        assert new_qb_pars["readout_integration_kernels_type"] == "optimal"
        assert (
            new_qb_pars["readout_integration_kernels"][0]["function"] == "sampled_pulse"
        )
        np.testing.assert_array_almost_equal(
            new_qb_pars["readout_integration_kernels"][0]["samples"],
            integration_kernels_ge,
        )

        np.testing.assert_array_almost_equal(
            new_qb_pars["readout_integration_discrimination_thresholds"],
            discrimination_thresholds_ge,
        )

    def test_create_and_run_no_fitting(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_ge,  # noqa: F811
    ):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        options = time_traces.analysis_workflow.options()
        options.do_fitting(False)

        result = time_traces.analysis_workflow(
            result=results_single_qubit_ge,
            qubits=q0,
            states="ge",
            options=options,
        ).run()

        assert len(result.tasks) == 4
        task_names = [t.name for t in result.tasks]
        assert "truncate_time_traces" in task_names
        assert "extract_kernels_thresholds" in task_names
        assert "filter_integration_kernels" not in task_names
        assert "extract_qubit_parameters" in task_names
        assert "plot_time_traces" in task_names
        assert "plot_kernels_traces" not in task_names
        assert "plot_kernels_fft" not in task_names

        qb_params = result.tasks["extract_qubit_parameters"]
        assert qb_params.output["new_parameter_values"]["q0"] == {}

        kernels, threhsolds = result.tasks["extract_kernels_thresholds"].output
        assert kernels is None
        assert threhsolds is None

    def test_create_and_run_no_plotting_time_traces(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_ge,  # noqa: F811
    ):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        options = time_traces.analysis_workflow.options()

        options.do_plotting_time_traces(False)
        result = time_traces.analysis_workflow(
            result=results_single_qubit_ge,
            qubits=q0,
            states="ge",
            options=options,
        ).run()
        assert len(result.tasks) == 5
        task_names = [t.name for t in result.tasks]
        assert "truncate_time_traces" in task_names
        assert "extract_kernels_thresholds" in task_names
        assert "filter_integration_kernels" not in task_names
        assert "extract_qubit_parameters" in task_names
        assert "plot_time_traces" not in task_names
        assert "plot_kernels_traces" in task_names
        assert "plot_kernels_fft" in task_names

    def test_create_and_run_no_plotting_kernels_traces(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_ge,  # noqa: F811
    ):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        options = time_traces.analysis_workflow.options()

        options.do_plotting_kernels_traces(False)
        result = time_traces.analysis_workflow(
            result=results_single_qubit_ge,
            qubits=q0,
            states="ge",
            options=options,
        ).run()
        assert len(result.tasks) == 5
        task_names = [t.name for t in result.tasks]
        assert "truncate_time_traces" in task_names
        assert "extract_kernels_thresholds" in task_names
        assert "filter_integration_kernels" not in task_names
        assert "extract_qubit_parameters" in task_names
        assert "plot_time_traces" in task_names
        assert "plot_kernels_traces" not in task_names
        assert "plot_kernels_fft" in task_names

    def test_create_and_run_no_plotting_kernels_fft(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_ge,  # noqa: F811
    ):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        options = time_traces.analysis_workflow.options()

        options.do_plotting_kernels_fft(False)
        result = time_traces.analysis_workflow(
            result=results_single_qubit_ge,
            qubits=q0,
            states="ge",
            options=options,
        ).run()
        assert len(result.tasks) == 5
        task_names = [t.name for t in result.tasks]
        assert "truncate_time_traces" in task_names
        assert "extract_kernels_thresholds" in task_names
        assert "filter_integration_kernels" not in task_names
        assert "extract_qubit_parameters" in task_names
        assert "plot_time_traces" in task_names
        assert "plot_kernels_traces" in task_names
        assert "plot_kernels_fft" not in task_names

    def test_create_and_run_no_plotting_at_all(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_ge,  # noqa: F811
    ):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        options = time_traces.analysis_workflow.options()

        options.do_plotting(False)
        result = time_traces.analysis_workflow(
            result=results_single_qubit_ge,
            qubits=q0,
            states="ge",
            options=options,
        ).run()
        assert len(result.tasks) == 3
        task_names = [t.name for t in result.tasks]
        assert "truncate_time_traces" in task_names
        assert "extract_kernels_thresholds" in task_names
        assert "filter_integration_kernels" not in task_names
        assert "extract_qubit_parameters" in task_names
        assert "plot_time_traces" not in task_names
        assert "plot_kernels_traces" not in task_names
        assert "plot_kernels_fft" not in task_names

    def test_create_and_run_close_figures(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_ge,  # noqa: F811
    ):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        options = time_traces.analysis_workflow.options()

        options.close_figures(True)
        result = time_traces.analysis_workflow(
            result=results_single_qubit_ge,
            qubits=q0,
            states="ge",
            options=options,
        ).run()

        assert isinstance(
            result.tasks["plot_time_traces"].output["q0"], mpl.figure.Figure
        )
        assert isinstance(
            result.tasks["plot_kernels_traces"].output["q0"], mpl.figure.Figure
        )
        assert isinstance(
            result.tasks["plot_kernels_fft"].output["q0"], mpl.figure.Figure
        )

        options.close_figures(False)
        result = time_traces.analysis_workflow(
            result=results_single_qubit_ge,
            qubits=q0,
            states="ge",
            options=options,
        ).run()

        assert isinstance(
            result.tasks["plot_time_traces"].output["q0"], mpl.figure.Figure
        )
        assert isinstance(
            result.tasks["plot_kernels_traces"].output["q0"], mpl.figure.Figure
        )
        assert isinstance(
            result.tasks["plot_kernels_fft"].output["q0"], mpl.figure.Figure
        )
        # close figures
        for tskn in ["plot_time_traces", "plot_kernels_traces", "plot_kernels_fft"]:
            plt.close(result.tasks[tskn].output["q0"])


class TestTimeTracesAnalysisSingleQubitGEF:
    def test_create_and_run(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_gef,  # noqa: F811
        integration_kernels_gef,  # noqa: F811
        discrimination_thresholds_gef,  # noqa: F811
    ):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        options = time_traces.analysis_workflow.options()
        states = ["g", "e", "f"]
        result = time_traces.analysis_workflow(
            result=results_single_qubit_gef,
            qubits=q0,
            states=states,
            options=options,
        ).run()

        assert len(result.tasks) == 6
        for task in result.tasks:
            if task.name not in [
                "extract_kernels_thresholds",
                "extract_qubit_parameters",
            ]:
                assert "q0" in task.output
        task_names = [t.name for t in result.tasks]
        assert "filter_integration_kernels" not in task_names
        assert len(result.tasks["truncate_time_traces"].output["q0"]) == 3
        data_g = results_single_qubit_gef[handles.result_handle("q0", suffix="g")].data
        assert len(result.tasks["truncate_time_traces"].output["q0"][0]) == len(
            data_g[: (len(data_g) // 16) * 16]
        )

        qubit_parameters = result.tasks["extract_qubit_parameters"].output

        old_qb_pars = qubit_parameters["old_parameter_values"]["q0"]
        assert old_qb_pars["readout_integration_kernels_type"] == "default"
        assert old_qb_pars["readout_integration_kernels"] is None
        assert old_qb_pars["readout_integration_discrimination_thresholds"] is None

        new_qb_pars = qubit_parameters["new_parameter_values"]["q0"]
        assert new_qb_pars["readout_integration_kernels_type"] == "optimal"
        assert (
            new_qb_pars["readout_integration_kernels"][0]["function"] == "sampled_pulse"
        )
        np.testing.assert_array_almost_equal(
            new_qb_pars["readout_integration_kernels"][0]["samples"],
            integration_kernels_gef[0],
        )
        assert (
            new_qb_pars["readout_integration_kernels"][1]["function"] == "sampled_pulse"
        )
        np.testing.assert_array_almost_equal(
            new_qb_pars["readout_integration_kernels"][1]["samples"],
            integration_kernels_gef[1],
        )

        np.testing.assert_array_almost_equal(
            new_qb_pars["readout_integration_discrimination_thresholds"],
            discrimination_thresholds_gef,
        )

    def test_create_and_run_filter_kernels(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_gef,  # noqa: F811
        integration_kernels_filtered_gef,  # noqa: F811
        discrimination_thresholds_gef,  # noqa: F811
    ):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        options = time_traces.analysis_workflow.options()
        options.filter_kernels(True)
        options.granularity(16)
        options.filter_cutoff_frequency(350e6)
        options.sampling_rate(2e9)

        states = ["g", "e", "f"]
        result = time_traces.analysis_workflow(
            result=results_single_qubit_gef,
            qubits=q0,
            states=states,
            options=options,
        ).run()

        assert len(result.tasks) == 7

        for task in result.tasks:
            if task.name not in [
                "extract_kernels_thresholds",
                "extract_qubit_parameters",
            ]:
                assert "q0" in task.output
        task_names = [t.name for t in result.tasks]
        assert "filter_integration_kernels" in task_names
        qubit_parameters = result.tasks["extract_qubit_parameters"].output

        old_qb_pars = qubit_parameters["old_parameter_values"]["q0"]
        assert old_qb_pars["readout_integration_kernels"] is None
        assert old_qb_pars["readout_integration_discrimination_thresholds"] is None

        new_qb_pars = qubit_parameters["new_parameter_values"]["q0"]
        assert (
            new_qb_pars["readout_integration_kernels"][0]["function"] == "sampled_pulse"
        )
        np.testing.assert_array_almost_equal(
            new_qb_pars["readout_integration_kernels"][0]["samples"],
            integration_kernels_filtered_gef[0],
        )
        assert (
            new_qb_pars["readout_integration_kernels"][1]["function"] == "sampled_pulse"
        )
        np.testing.assert_array_almost_equal(
            new_qb_pars["readout_integration_kernels"][1]["samples"],
            integration_kernels_filtered_gef[1],
        )

        np.testing.assert_array_almost_equal(
            new_qb_pars["readout_integration_discrimination_thresholds"],
            discrimination_thresholds_gef,
        )


class TestTimeTracesAnalysisSingleQubitG:
    def test_create_and_run(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_g,  # noqa: F811
    ):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        options = time_traces.analysis_workflow.options()
        states = ["g"]
        result = time_traces.analysis_workflow(
            result=results_single_qubit_g,
            qubits=q0,
            states=states,
            options=options,
        ).run()

        assert len(result.tasks) == 6
        task_names = [t.name for t in result.tasks]
        assert "filter_integration_kernels" not in task_names
        assert len(result.tasks["truncate_time_traces"].output["q0"]) == 1
        data_g = results_single_qubit_g[handles.result_handle("q0", suffix="g")].data
        assert len(result.tasks["truncate_time_traces"].output["q0"][0]) == len(
            data_g[: (len(data_g) // 16) * 16]
        )

        kernels, thresholds = result.tasks["extract_kernels_thresholds"].output
        assert len(kernels) == 0
        assert len(thresholds) == 0

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        old_qb_pars = qubit_parameters["old_parameter_values"]["q0"]
        assert len(old_qb_pars) == 0
        new_qb_pars = qubit_parameters["new_parameter_values"]["q0"]
        assert len(new_qb_pars) == 0

    def test_create_and_run_filter_kernels(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_g,  # noqa: F811
    ):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        options = time_traces.analysis_workflow.options()
        options.filter_kernels(True)
        options.granularity(16)
        options.filter_cutoff_frequency(350e6)
        options.sampling_rate(2e9)

        states = ["g"]
        result = time_traces.analysis_workflow(
            result=results_single_qubit_g,
            qubits=q0,
            states=states,
            options=options,
        ).run()

        assert len(result.tasks) == 7
        task_names = [t.name for t in result.tasks]
        assert "filter_integration_kernels" in task_names

        kernels, thresholds = result.tasks["extract_kernels_thresholds"].output
        assert len(kernels) == 0
        assert len(thresholds) == 0

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        old_qb_pars = qubit_parameters["old_parameter_values"]["q0"]
        assert len(old_qb_pars) == 0
        new_qb_pars = qubit_parameters["new_parameter_values"]["q0"]
        assert len(new_qb_pars) == 0
