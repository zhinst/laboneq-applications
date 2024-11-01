"""Tests for the DRAG quadrature-scaling analysis using the testing utilities."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from laboneq.workflow.tasks import handles
from laboneq.workflow.tasks.run_experiment import (
    AcquiredResult,
    RunExperimentResults,
)

from laboneq_applications.analysis import drag_q_scaling


@pytest.fixture()
def results_single_qubit():
    """Results from a DRAG quadrature-scaling calibration experiment.

    In the AcquiredResults below, and the data is the raw acquisition result obtained
    in integrated-average mode for each of the three pairs of preparation pulses,
    x90-x180 (xx), x90-y180 (xy), x90-ym180 (xmy). The sweep points correspond to the
    DRAG quadrature scaling factor (beta).
    """
    data = {}
    data[handles.result_handle("q0", suffix="xy")] = AcquiredResult(
        data=np.array(
            [
                -0.185234 - 0.03658602j,
                -0.20503443 - 0.04504895j,
                -0.21791076 - 0.05444159j,
                -0.23116526 - 0.06416006j,
                -0.25795109 - 0.0805195j,
                -0.26594044 - 0.08610988j,
                -0.28922453 - 0.10309644j,
                -0.3089145 - 0.11294363j,
                -0.32459566 - 0.12207225j,
                -0.33419394 - 0.13268631j,
                -0.35452839 - 0.14296266j,
            ]
        )
    )
    data[handles.result_handle("q0", suffix="xmy")] = AcquiredResult(
        data=np.array(
            [
                -0.35557612 - 0.14272263j,
                -0.34289465 - 0.13394558j,
                -0.32718833 - 0.12176565j,
                -0.30646699 - 0.11429455j,
                -0.29724482 - 0.10560441j,
                -0.27855107 - 0.0919168j,
                -0.26091608 - 0.08512063j,
                -0.24748801 - 0.07564026j,
                -0.22364471 - 0.05653496j,
                -0.20401136 - 0.04885739j,
                -0.18738762 - 0.03566131j,
            ]
        )
    )
    data[handles.result_handle("q0", suffix="xx")] = AcquiredResult(
        data=np.array(
            [
                -0.27028461 - 0.09548818j,
                -0.27859307 - 0.09192484j,
                -0.28101732 - 0.09405002j,
                -0.27760232 - 0.0910527j,
                -0.27500593 - 0.08806072j,
                -0.28156977 - 0.09513699j,
                -0.28374517 - 0.0935586j,
                -0.27487533 - 0.09483633j,
                -0.27794626 - 0.09659119j,
                -0.28568239 - 0.09430552j,
                -0.2854309 - 0.09919256j,
            ]
        )
    )
    data[handles.calibration_trace_handle("q0", "g")] = AcquiredResult(
        data=(-0.43671994930357777 - 0.196165214208241j),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q0", "e")] = AcquiredResult(
        data=(-0.10701396376997645 + 0.01605024387665138j),
        axis_name=[],
        axis=[],
    )
    sweep_points = np.array(
        [
            -0.04228073,
            -0.03228073,
            -0.02228073,
            -0.01228073,
            -0.00228073,
            0.00771927,
            0.01771927,
            0.02771927,
            0.03771927,
            0.04771927,
            0.05771927,
        ]
    )
    return RunExperimentResults(data=data), sweep_points


class TestDRAGQScalingAnalysisSingleQubit:
    def test_create_and_run_no_pca(
        self, single_tunable_transmon_platform, results_single_qubit
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = drag_q_scaling.analysis_workflow.options()
        options.do_pca(False)

        result = drag_q_scaling.analysis_workflow(
            result=results_single_qubit[0],
            qubits=q0,
            q_scalings=results_single_qubit[1],
            options=options,
        ).run()

        assert len(result.tasks) == 5

        for task in result.tasks:
            if task.name != "extract_qubit_parameters":
                assert "q0" in task.output
        assert "new_parameter_values" in result.tasks["extract_qubit_parameters"].output

        proc_data_dict = result.tasks["calculate_qubit_population_for_pulse_ids"].output
        assert proc_data_dict["q0"]["xy"]["num_cal_traces"] == 2
        np.testing.assert_array_almost_equal(
            proc_data_dict["q0"]["xmy"]["population"],
            np.array(
                [
                    0.24778582,
                    0.28709716,
                    0.33759257,
                    0.39234316,
                    0.42411588,
                    0.48309911,
                    0.53029924,
                    0.5721825,
                    0.64968744,
                    0.70238977,
                    0.75625526,
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            proc_data_dict["q0"]["xx"]["population"],
            np.array(
                [
                    0.4958972,
                    0.48299792,
                    0.47486553,
                    0.48632649,
                    0.49602454,
                    0.4721804,
                    0.46969386,
                    0.48695196,
                    0.47794389,
                    0.4645084,
                    0.45830197,
                ]
            ),
        )

        fit_values = result.tasks["fit_data"].output["q0"]["xy"].best_values
        np.testing.assert_allclose(
            fit_values["gradient"], -5.158296958933983, rtol=1e-4
        )
        np.testing.assert_allclose(
            fit_values["intercept"], 0.5441626303305539, rtol=1e-4
        )
        fit_values = result.tasks["fit_data"].output["q0"]["xx"].best_values
        np.testing.assert_allclose(fit_values["gradient"], 0)
        np.testing.assert_allclose(
            fit_values["intercept"], 0.47869928742414075, rtol=1e-4
        )

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["ge_drive_pulse.beta"],
            0.01,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "ge_drive_pulse.beta"
            ].nominal_value,
            0.009181531700468494,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "ge_drive_pulse.beta"
            ].std_dev,
            0.0004944111397285312,
            rtol=1e-4,
        )

    def test_create_and_run_pca(
        self, single_tunable_transmon_platform, results_single_qubit
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = drag_q_scaling.analysis_workflow.options()
        options.do_pca(True)

        result = drag_q_scaling.analysis_workflow(
            result=results_single_qubit[0],
            qubits=q0,
            q_scalings=results_single_qubit[1],
            options=options,
        ).run()

        proc_data_dict = result.tasks["calculate_qubit_population_for_pulse_ids"].output
        assert proc_data_dict["q0"]["xmy"]["num_cal_traces"] == 2
        np.testing.assert_array_almost_equal(
            proc_data_dict["q0"]["xy"]["population"],
            np.array(
                [
                    0.10034619,
                    0.07911665,
                    0.06320556,
                    0.04680011,
                    0.01542247,
                    0.0056787,
                    -0.02309419,
                    -0.04498023,
                    -0.06310668,
                    -0.07692282,
                    -0.09958307,
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            proc_data_dict["q0"]["xx"]["population"],
            np.array(
                [
                    0.00546271,
                    0.00039947,
                    -0.00278888,
                    0.00170446,
                    0.00550624,
                    -0.00384127,
                    -0.00481806,
                    0.00195309,
                    -0.0015791,
                    -0.00685158,
                    -0.00928203,
                ]
            ),
        )

        fit_values = result.tasks["fit_data"].output["q0"]["xmy"].best_values
        np.testing.assert_allclose(
            fit_values["gradient"], 1.9982301846461958, rtol=1e-4
        )
        np.testing.assert_allclose(
            fit_values["intercept"], -0.016067556464128677, rtol=1e-4
        )
        fit_values = result.tasks["fit_data"].output["q0"]["xx"].best_values
        np.testing.assert_allclose(fit_values["gradient"], 0)
        np.testing.assert_allclose(
            fit_values["intercept"], -0.0012849945692198729, rtol=1e-4
        )

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "ge_drive_pulse.beta"
            ].nominal_value,
            0.007944287108763201,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "ge_drive_pulse.beta"
            ].std_dev,
            0.0004897434559751095,
            rtol=1e-4,
        )

    def test_create_and_run_no_fitting(
        self, single_tunable_transmon_platform, results_single_qubit
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = drag_q_scaling.analysis_workflow.options()
        options.do_fitting(False)

        result = drag_q_scaling.analysis_workflow(
            result=results_single_qubit[0],
            qubits=q0,
            q_scalings=results_single_qubit[1],
            options=options,
        ).run()

        assert len(result.tasks) == 5

        assert result.tasks["fit_data"].output == {}
        qb_params = result.tasks["extract_qubit_parameters"]
        assert len(qb_params.output["old_parameter_values"]["q0"]) > 0
        assert len(qb_params.output["new_parameter_values"]["q0"]) == 0

    def test_create_and_run_no_plotting(
        self, single_tunable_transmon_platform, results_single_qubit
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = drag_q_scaling.analysis_workflow.options()

        options.do_qubit_population_plotting(False)
        result = drag_q_scaling.analysis_workflow(
            result=results_single_qubit[0],
            qubits=q0,
            q_scalings=results_single_qubit[1],
            options=options,
        ).run()
        assert len(result.tasks) == 4
        task_names = [t.name for t in result.tasks]
        assert "fit_data" in task_names
        assert "extract_qubit_parameters" in task_names
        assert "plot_raw_complex_data_1d" in task_names
        assert "plot_population" not in task_names

        options.do_qubit_population_plotting(True)
        options.do_raw_data_plotting(False)
        result = drag_q_scaling.analysis_workflow(
            result=results_single_qubit[0],
            qubits=q0,
            q_scalings=results_single_qubit[1],
            options=options,
        ).run()
        assert len(result.tasks) == 4
        task_names = [t.name for t in result.tasks]
        assert "fit_data" in task_names
        assert "extract_qubit_parameters" in task_names
        assert "plot_raw_complex_data_1d" not in task_names
        assert "plot_population" in task_names

        options.do_plotting(False)
        result = drag_q_scaling.analysis_workflow(
            result=results_single_qubit[0],
            qubits=q0,
            q_scalings=results_single_qubit[1],
            options=options,
        ).run()
        assert len(result.tasks) == 3
        task_names = [t.name for t in result.tasks]
        assert "fit_data" in task_names
        assert "extract_qubit_parameters" in task_names
        assert "plot_raw_complex_data_1d" not in task_names
        assert "plot_population" not in task_names


@pytest.fixture()
def results_two_qubit():
    """Results from a DRAG quadrature-scaling calibration experiment.

    In the AcquiredResults below, and the data is the raw acquisition result obtained
    in integrated-average mode for each of the three pairs of preparation pulses,
    x90-x180 (xx), x90-y180 (xy), x90-ym180 (xmy). The sweep points correspond to the
    DRAG quadrature scaling factor (beta).
    """
    data = {}
    # q0
    data[handles.result_handle("q0", suffix="xy")] = AcquiredResult(
        data=np.array(
            [
                -0.185234 - 0.03658602j,
                -0.20503443 - 0.04504895j,
                -0.21791076 - 0.05444159j,
                -0.23116526 - 0.06416006j,
                -0.25795109 - 0.0805195j,
                -0.26594044 - 0.08610988j,
                -0.28922453 - 0.10309644j,
                -0.3089145 - 0.11294363j,
                -0.32459566 - 0.12207225j,
                -0.33419394 - 0.13268631j,
                -0.35452839 - 0.14296266j,
            ]
        )
    )
    data[handles.result_handle("q0", suffix="xmy")] = AcquiredResult(
        data=np.array(
            [
                -0.35557612 - 0.14272263j,
                -0.34289465 - 0.13394558j,
                -0.32718833 - 0.12176565j,
                -0.30646699 - 0.11429455j,
                -0.29724482 - 0.10560441j,
                -0.27855107 - 0.0919168j,
                -0.26091608 - 0.08512063j,
                -0.24748801 - 0.07564026j,
                -0.22364471 - 0.05653496j,
                -0.20401136 - 0.04885739j,
                -0.18738762 - 0.03566131j,
            ]
        )
    )
    data[handles.result_handle("q0", suffix="xx")] = AcquiredResult(
        data=np.array(
            [
                -0.27028461 - 0.09548818j,
                -0.27859307 - 0.09192484j,
                -0.28101732 - 0.09405002j,
                -0.27760232 - 0.0910527j,
                -0.27500593 - 0.08806072j,
                -0.28156977 - 0.09513699j,
                -0.28374517 - 0.0935586j,
                -0.27487533 - 0.09483633j,
                -0.27794626 - 0.09659119j,
                -0.28568239 - 0.09430552j,
                -0.2854309 - 0.09919256j,
            ]
        )
    )
    data[handles.calibration_trace_handle("q0", "g")] = AcquiredResult(
        data=(-0.43671994930357777 - 0.196165214208241j),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q0", "e")] = AcquiredResult(
        data=(-0.10701396376997645 + 0.01605024387665138j),
        axis_name=[],
        axis=[],
    )
    # q1
    data[handles.result_handle("q1", suffix="xy")] = AcquiredResult(
        data=np.array(
            [
                -0.27916464 + 0.16131101j,
                -0.27944504 + 0.14822675j,
                -0.27904407 + 0.13043341j,
                -0.27049865 + 0.11701805j,
                -0.27472951 + 0.09904274j,
                -0.26542473 + 0.07764735j,
                -0.26481589 + 0.06395056j,
                -0.26362381 + 0.05200055j,
                -0.26288722 + 0.02930015j,
                -0.26321846 + 0.0173421j,
                -0.25374957 - 0.0026386j,
            ]
        )
    )
    data[handles.result_handle("q1", suffix="xmy")] = AcquiredResult(
        data=np.array(
            [
                -0.25472766 + 0.00399406j,
                -0.257671 + 0.01452789j,
                -0.2609928 + 0.03492065j,
                -0.2644978 + 0.0468058j,
                -0.26268044 + 0.05972829j,
                -0.26734248 + 0.07688341j,
                -0.2715577 + 0.0931445j,
                -0.27415986 + 0.11033044j,
                -0.27243309 + 0.12772586j,
                -0.2772246 + 0.14239275j,
                -0.27713368 + 0.15997861j,
            ]
        )
    )
    data[handles.result_handle("q1", suffix="xx")] = AcquiredResult(
        data=np.array(
            [
                -0.26896 + 0.08376001j,
                -0.27239422 + 0.08920608j,
                -0.26742344 + 0.08325086j,
                -0.27029414 + 0.08166594j,
                -0.26673585 + 0.07675461j,
                -0.26533978 + 0.07679418j,
                -0.26901224 + 0.07599635j,
                -0.27278774 + 0.07383619j,
                -0.26791945 + 0.0781864j,
                -0.26800879 + 0.0670004j,
                -0.26794289 + 0.06813104j,
            ]
        )
    )
    data[handles.calibration_trace_handle("q1", "g")] = AcquiredResult(
        data=(-0.24472940477003619 - 0.08540858424191179j),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q1", "e")] = AcquiredResult(
        data=(-0.2888925769156423 + 0.22843360534572796j),
        axis_name=[],
        axis=[],
    )

    sweep_points = [
        np.array(
            [
                -0.04228073,
                -0.03228073,
                -0.02228073,
                -0.01228073,
                -0.00228073,
                0.00771927,
                0.01771927,
                0.02771927,
                0.03771927,
                0.04771927,
                0.05771927,
            ]
        ),
        np.array(
            [
                -0.03688166,
                -0.02688166,
                -0.01688166,
                -0.00688166,
                0.00311834,
                0.01311834,
                0.02311834,
                0.03311834,
                0.04311834,
                0.05311834,
                0.06311834,
            ]
        ),
    ]
    return RunExperimentResults(data=data), sweep_points


class TestDRAGQScalingAnalysisTwoQubit:
    def test_create_and_run_no_pca(
        self, two_tunable_transmon_platform, results_two_qubit
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = drag_q_scaling.analysis_workflow.options()
        options.do_pca(False)

        result = drag_q_scaling.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            q_scalings=results_two_qubit[1],
            options=options,
        ).run()

        assert len(result.tasks) == 5

        for task in result.tasks:
            if task.name != "extract_qubit_parameters":
                assert "q0" in task.output
                assert "q1" in task.output
        assert "new_parameter_values" in result.tasks["extract_qubit_parameters"].output

        proc_data_dict = result.tasks["calculate_qubit_population_for_pulse_ids"].output
        assert proc_data_dict["q0"]["xy"]["num_cal_traces"] == 2
        assert proc_data_dict["q1"]["xy"]["num_cal_traces"] == 2

        # q0
        np.testing.assert_array_almost_equal(
            proc_data_dict["q0"]["xmy"]["population"],
            np.array(
                [
                    0.24778582,
                    0.28709716,
                    0.33759257,
                    0.39234316,
                    0.42411588,
                    0.48309911,
                    0.53029924,
                    0.5721825,
                    0.64968744,
                    0.70238977,
                    0.75625526,
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            proc_data_dict["q0"]["xx"]["population"],
            np.array(
                [
                    0.4958972,
                    0.48299792,
                    0.47486553,
                    0.48632649,
                    0.49602454,
                    0.4721804,
                    0.46969386,
                    0.48695196,
                    0.47794389,
                    0.4645084,
                    0.45830197,
                ]
            ),
        )
        fit_values = result.tasks["fit_data"].output["q0"]["xy"].best_values
        np.testing.assert_allclose(
            fit_values["gradient"], -5.158296958933983, rtol=1e-4
        )
        np.testing.assert_allclose(
            fit_values["intercept"], 0.5441626303305539, rtol=1e-4
        )
        fit_values = result.tasks["fit_data"].output["q0"]["xx"].best_values
        np.testing.assert_allclose(fit_values["gradient"], 0)
        np.testing.assert_allclose(
            fit_values["intercept"], 0.47869928742414075, rtol=1e-4
        )

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["ge_drive_pulse.beta"],
            0.01,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "ge_drive_pulse.beta"
            ].nominal_value,
            0.009181531700468494,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "ge_drive_pulse.beta"
            ].std_dev,
            0.0004944111397285312,
            rtol=1e-4,
        )

        # q1
        np.testing.assert_array_almost_equal(
            proc_data_dict["q1"]["xmy"]["population"],
            np.array(
                [
                    0.28372962,
                    0.31793608,
                    0.38311265,
                    0.42178818,
                    0.46136478,
                    0.51701476,
                    0.56967494,
                    0.62451556,
                    0.67810742,
                    0.72603998,
                    0.78094608,
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            proc_data_dict["q1"]["xx"]["population"],
            np.array(
                [
                    0.53921148,
                    0.55773736,
                    0.53694512,
                    0.53325525,
                    0.51634562,
                    0.51585544,
                    0.51497732,
                    0.50988799,
                    0.52133957,
                    0.48642878,
                    0.48993244,
                ]
            ),
        )
        fit_values = result.tasks["fit_data"].output["q1"]["xy"].best_values
        np.testing.assert_allclose(fit_values["gradient"], -5.25185110574011, rtol=1e-4)
        np.testing.assert_allclose(
            fit_values["intercept"], 0.6001534640446579, rtol=1e-4
        )
        fit_values = result.tasks["fit_data"].output["q1"]["xx"].best_values
        np.testing.assert_allclose(fit_values["gradient"], 0)
        np.testing.assert_allclose(
            fit_values["intercept"], 0.5201742151646731, rtol=1e-4
        )

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q1"]["ge_drive_pulse.beta"],
            0.01,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q1"][
                "ge_drive_pulse.beta"
            ].nominal_value,
            0.013823181227748789,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q1"][
                "ge_drive_pulse.beta"
            ].std_dev,
            0.0003852317805923965,
            rtol=1e-4,
        )

    def test_create_and_run_pca(self, two_tunable_transmon_platform, results_two_qubit):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = drag_q_scaling.analysis_workflow.options()
        options.do_pca(True)

        result = drag_q_scaling.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            q_scalings=results_two_qubit[1],
            options=options,
        ).run()

        assert len(result.tasks) == 5
        proc_data_dict = result.tasks["calculate_qubit_population_for_pulse_ids"].output
        assert proc_data_dict["q0"]["xmy"]["num_cal_traces"] == 2
        assert proc_data_dict["q1"]["xmy"]["num_cal_traces"] == 2

        # q1
        np.testing.assert_array_almost_equal(
            proc_data_dict["q0"]["xy"]["population"],
            np.array(
                [
                    0.10034619,
                    0.07911665,
                    0.06320556,
                    0.04680011,
                    0.01542247,
                    0.0056787,
                    -0.02309419,
                    -0.04498023,
                    -0.06310668,
                    -0.07692282,
                    -0.09958307,
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            proc_data_dict["q0"]["xx"]["population"],
            np.array(
                [
                    0.00546271,
                    0.00039947,
                    -0.00278888,
                    0.00170446,
                    0.00550624,
                    -0.00384127,
                    -0.00481806,
                    0.00195309,
                    -0.0015791,
                    -0.00685158,
                    -0.00928203,
                ]
            ),
        )

        fit_values = result.tasks["fit_data"].output["q0"]["xmy"].best_values
        np.testing.assert_allclose(
            fit_values["gradient"], 1.9982301846461958, rtol=1e-4
        )
        np.testing.assert_allclose(
            fit_values["intercept"], -0.016067556464128677, rtol=1e-4
        )
        fit_values = result.tasks["fit_data"].output["q0"]["xx"].best_values
        np.testing.assert_allclose(fit_values["gradient"], 0)
        np.testing.assert_allclose(
            fit_values["intercept"], -0.0012849945692198729, rtol=1e-4
        )

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "ge_drive_pulse.beta"
            ].nominal_value,
            0.007944287108763201,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "ge_drive_pulse.beta"
            ].std_dev,
            0.0004897434559751095,
            rtol=1e-4,
        )

        # q1
        np.testing.assert_array_almost_equal(
            proc_data_dict["q1"]["xy"]["population"],
            np.array(
                [
                    -0.08225924,
                    -0.06934653,
                    -0.05167571,
                    -0.03718546,
                    -0.01999039,
                    0.002507,
                    0.01615199,
                    0.02815039,
                    0.05072635,
                    0.06251699,
                    0.08363718,
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            proc_data_dict["q1"]["xx"]["population"],
            np.array(
                [
                    -0.00701721,
                    -0.01288883,
                    -0.00629885,
                    -0.00512956,
                    0.00022982,
                    0.00038525,
                    0.00066337,
                    0.00227614,
                    -0.00135298,
                    0.00971136,
                    0.00860094,
                ]
            ),
        )

        fit_values = result.tasks["fit_data"].output["q1"]["xmy"].best_values
        np.testing.assert_allclose(
            fit_values["gradient"], -1.589652207580708, rtol=1e-4
        )
        np.testing.assert_allclose(
            fit_values["intercept"], 0.019682506366657283, rtol=1e-4
        )
        fit_values = result.tasks["fit_data"].output["q1"]["xx"].best_values
        np.testing.assert_allclose(fit_values["gradient"], 0)
        np.testing.assert_allclose(
            fit_values["intercept"], -0.0009836854549006538, rtol=1e-4
        )

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q1"][
                "ge_drive_pulse.beta"
            ].nominal_value,
            0.013226884153893623,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q1"][
                "ge_drive_pulse.beta"
            ].std_dev,
            0.00038307777797754036,
            rtol=1e-4,
        )

    def test_create_and_run_no_fitting(
        self, two_tunable_transmon_platform, results_two_qubit
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = drag_q_scaling.analysis_workflow.options()
        options.do_fitting(False)

        result = drag_q_scaling.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            q_scalings=results_two_qubit[1],
            options=options,
        ).run()

        assert len(result.tasks) == 5
        assert result.tasks["fit_data"].output == {}
        qb_params = result.tasks["extract_qubit_parameters"]
        assert len(qb_params.output["old_parameter_values"]["q0"]) > 0
        assert len(qb_params.output["new_parameter_values"]["q0"]) == 0
        assert len(qb_params.output["old_parameter_values"]["q1"]) > 0
        assert len(qb_params.output["new_parameter_values"]["q1"]) == 0

    def test_create_and_run_close_figures(
        self, two_tunable_transmon_platform, results_two_qubit
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = drag_q_scaling.analysis_workflow.options()
        options.close_figures(True)

        result = drag_q_scaling.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            q_scalings=results_two_qubit[1],
            options=options,
        ).run()

        assert isinstance(
            result.tasks["plot_population"].output["q0"], mpl.figure.Figure
        )
        assert isinstance(
            result.tasks["plot_population"].output["q1"], mpl.figure.Figure
        )

        options.close_figures(False)
        result = drag_q_scaling.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            q_scalings=results_two_qubit[1],
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
