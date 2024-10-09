"""Tests for the IQ-Blobs analysis using the testing utilities."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from test_iq_blobs_data import (  # noqa: F401
    results_single_qubit_gef,
    results_two_qubit_ge,
)

from laboneq_applications.analysis import iq_blobs


class TestIQBlobsAnalysisSingleQubitGEF:
    def test_create_and_run(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_gef,  # noqa: F811
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = iq_blobs.analysis_workflow.options()
        states = ["g", "e", "f"]
        result = iq_blobs.analysis_workflow(
            result=results_single_qubit_gef,
            qubits=q0,
            states=states,
            options=options,
        ).run()

        assert len(result.tasks) == 6
        for task in result.tasks:
            assert "q0" in task.output
        proc_data_dict = result.tasks["collect_shots"].output
        assert list(proc_data_dict["q0"]["shots_per_state"]) == states
        len_sps = [len(sps) for sps in proc_data_dict["q0"]["shots_per_state"].values()]
        len_raw_data = [len(results_single_qubit_gef[f"result/q0/{s}"]) for s in states]
        assert len_sps == len_raw_data
        assert proc_data_dict["q0"]["shots_combined"].shape == (sum(len_raw_data), 2)
        assert proc_data_dict["q0"]["ideal_states_shots"].size == sum(len_raw_data)

        fit_res = result.tasks["fit_data"].output["q0"]
        assert isinstance(fit_res, LinearDiscriminantAnalysis)
        np.testing.assert_array_almost_equal(
            fit_res.means_,
            np.array(
                [
                    [0.96392208, -0.2706722],
                    [-0.26109146, -0.83566163],
                    [-0.14423255, -0.1312243],
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            fit_res.intercept_,
            np.array([-12.66487943, -11.56969744, 4.13378278]),
        )

        assgn_mtx = result.tasks["calculate_assignment_matrices"].output["q0"]
        np.testing.assert_array_almost_equal(
            assgn_mtx,
            np.array([[0.975, 0.025, 0.0], [0.02, 0.975, 0.005], [0.015, 0.035, 0.95]]),
        )

        assgn_fid = result.tasks["calculate_assignment_fidelities"].output["q0"]
        np.testing.assert_allclose(assgn_fid, 0.96666666666, rtol=1e-4)

    def test_create_and_run_no_fitting(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_gef,  # noqa: F811
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = iq_blobs.analysis_workflow.options()
        options.do_fitting(False)

        result = iq_blobs.analysis_workflow(
            result=results_single_qubit_gef,
            qubits=q0,
            states="gef",
            options=options,
        ).run()

        assert len(result.tasks) == 2
        task_names = [t.name for t in result.tasks]
        assert "collect_shots" in task_names
        assert "fit_data" not in task_names
        assert "calculate_assignment_matrices" not in task_names
        assert "calculate_assignment_fidelities" not in task_names
        assert "plot_iq_blobs" in task_names
        assert "plot_assignment_matrices" not in task_names

    def test_create_and_run_no_plotting(
        self,
        single_tunable_transmon_platform,
        results_single_qubit_gef,  # noqa: F811
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = iq_blobs.analysis_workflow.options()

        # No plotting of assignment matrices
        options.do_plotting_assignment_matrices(False)
        result = iq_blobs.analysis_workflow(
            result=results_single_qubit_gef,
            qubits=q0,
            states="gef",
            options=options,
        ).run()
        assert len(result.tasks) == 5
        task_names = [t.name for t in result.tasks]
        assert "collect_shots" in task_names
        assert "fit_data" in task_names
        assert "calculate_assignment_matrices" in task_names
        assert "calculate_assignment_fidelities" in task_names
        assert "plot_iq_blobs" in task_names
        assert "plot_assignment_matrices" not in task_names

        # No plotting of the iq blobs (ssro data)
        options.do_plotting_assignment_matrices(True)
        options.do_plotting_iq_blobs(False)
        result = iq_blobs.analysis_workflow(
            result=results_single_qubit_gef,
            qubits=q0,
            states="gef",
            options=options,
        ).run()
        assert len(result.tasks) == 5
        task_names = [t.name for t in result.tasks]
        assert "collect_shots" in task_names
        assert "fit_data" in task_names
        assert "calculate_assignment_matrices" in task_names
        assert "calculate_assignment_fidelities" in task_names
        assert "plot_iq_blobs" not in task_names
        assert "plot_assignment_matrices" in task_names

        # Disable plotting entirely
        options.do_plotting_assignment_matrices(True)
        options.do_plotting_iq_blobs(True)
        options.do_plotting(False)
        result = iq_blobs.analysis_workflow(
            result=results_single_qubit_gef,
            qubits=q0,
            states="gef",
            options=options,
        ).run()
        assert len(result.tasks) == 4
        task_names = [t.name for t in result.tasks]
        assert "collect_shots" in task_names
        assert "fit_data" in task_names
        assert "calculate_assignment_matrices" in task_names
        assert "calculate_assignment_fidelities" in task_names
        assert "plot_iq_blobs" not in task_names
        assert "plot_assignment_matrices" not in task_names


class TestIQBlobsAnalysisTwoQubitGE:
    def test_create_and_run(self, two_tunable_transmon_platform, results_two_qubit_ge):  # noqa: F811
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = iq_blobs.analysis_workflow.options()
        states = ["g", "e"]
        result = iq_blobs.analysis_workflow(
            result=results_two_qubit_ge,
            qubits=qubits,
            states=states,
            options=options,
        ).run()

        assert len(result.tasks) == 6
        for task in result.tasks:
            assert "q0" in task.output
            assert "q1" in task.output
        proc_data_dict = result.tasks["collect_shots"].output
        for qid in ["q0", "q1"]:
            assert list(proc_data_dict[qid]["shots_per_state"]) == states
            len_sps = [
                len(sps) for sps in proc_data_dict[qid]["shots_per_state"].values()
            ]
            len_raw_data = [
                len(results_two_qubit_ge[f"result/{qid}/{s}"]) for s in states
            ]
            assert len_sps == len_raw_data
            assert proc_data_dict[qid]["shots_combined"].shape == (
                sum(len_raw_data),
                2,
            )
            assert proc_data_dict[qid]["ideal_states_shots"].size == sum(len_raw_data)

        # test fit results - q0
        fit_res = result.tasks["fit_data"].output["q0"]
        assert isinstance(fit_res, LinearDiscriminantAnalysis)
        np.testing.assert_array_almost_equal(
            fit_res.means_,
            np.array([[-0.1276838, 0.74826288], [0.36805776, 0.07299178]]),
        )
        np.testing.assert_array_almost_equal(
            fit_res.intercept_,
            np.array([5.83011479]),
        )

        assgn_mtx = result.tasks["calculate_assignment_matrices"].output["q0"]
        np.testing.assert_array_almost_equal(
            assgn_mtx,
            np.array([[0.96, 0.04], [0.065, 0.935]]),
        )

        assgn_fid = result.tasks["calculate_assignment_fidelities"].output["q0"]
        np.testing.assert_allclose(assgn_fid, 0.9475, rtol=1e-4)

        # test fit results - q1
        fit_res = result.tasks["fit_data"].output["q1"]
        assert isinstance(fit_res, LinearDiscriminantAnalysis)
        np.testing.assert_array_almost_equal(
            fit_res.means_,
            np.array([[0.15538387, 0.31928364], [0.91968368, -0.22334999]]),
        )
        np.testing.assert_array_almost_equal(
            fit_res.intercept_,
            np.array([-4.91741867]),
        )

        assgn_mtx = result.tasks["calculate_assignment_matrices"].output["q1"]
        np.testing.assert_array_almost_equal(
            assgn_mtx,
            np.array([[0.965, 0.035], [0.1, 0.9]]),
        )

        assgn_fid = result.tasks["calculate_assignment_fidelities"].output["q1"]
        np.testing.assert_allclose(assgn_fid, 0.9325, rtol=1e-4)

    def test_create_and_run_close_figures(
        self,
        two_tunable_transmon_platform,
        results_two_qubit_ge,  # noqa: F811
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = iq_blobs.analysis_workflow.options()

        options.close_figures(True)
        result = iq_blobs.analysis_workflow(
            result=results_two_qubit_ge,
            qubits=qubits,
            states="ge",
            options=options,
        ).run()

        assert isinstance(result.tasks["plot_iq_blobs"].output["q0"], mpl.figure.Figure)
        assert isinstance(
            result.tasks["plot_assignment_matrices"].output["q1"], mpl.figure.Figure
        )

        options.close_figures(False)
        result = iq_blobs.analysis_workflow(
            result=results_two_qubit_ge,
            qubits=qubits,
            states="ge",
            options=options,
        ).run()

        assert isinstance(result.tasks["plot_iq_blobs"].output["q0"], mpl.figure.Figure)
        assert isinstance(
            result.tasks["plot_assignment_matrices"].output["q1"], mpl.figure.Figure
        )
        # close figures
        for task_name in ["plot_iq_blobs", "plot_assignment_matrices"]:
            for qid in ["q0", "q1"]:
                plt.close(result.tasks[task_name].output[qid])
