"""Tests for the lifetime_measurement analysis using the testing utilities."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from laboneq_applications.analysis import lifetime_measurement
from laboneq_applications.tasks.run_experiment import (
    AcquiredResult,
    RunExperimentResults,
)


@pytest.fixture()
def results_single_qubit():
    """Results from a lifetime_measurement experiment.

    In the AcquiredResults below, the axis corresponds to the time-delay after the x180
    pulse in the lifetime_measurement experiment, and the data is the raw acquisition
    result obtained in integrated-average mode.
    """
    data = {
        "result": {
            "q0": AcquiredResult(
                data=np.array(
                    [
                        -0.54653355 - 1.17900678j,
                        -0.48369549 - 1.02114036j,
                        -0.42631291 - 0.89184827j,
                        -0.38948615 - 0.79238741j,
                        -0.34899969 - 0.70901739j,
                        -0.32238488 - 0.63723581j,
                        -0.29655117 - 0.56829491j,
                        -0.27073651 - 0.51895399j,
                        -0.2378248 - 0.4451091j,
                        -0.2235502 - 0.40985343j,
                        -0.20836014 - 0.36324451j,
                        -0.19136864 - 0.33351472j,
                        -0.18146525 - 0.29707788j,
                        -0.16225064 - 0.26498941j,
                        -0.14831841 - 0.23149716j,
                        -0.1330217 - 0.18506588j,
                        -0.1307423 - 0.1867978j,
                        -0.12233241 - 0.16564238j,
                        -0.11069799 - 0.14770174j,
                        -0.10880689 - 0.12932072j,
                        -0.10240398 - 0.12374628j,
                        -0.10462664 - 0.12401976j,
                        -0.09051118 - 0.09979015j,
                        -0.08690288 - 0.08633465j,
                        -0.08404272 - 0.07923303j,
                        -0.08276078 - 0.06772339j,
                        -0.07825265 - 0.06516248j,
                        -0.07552934 - 0.06284701j,
                        -0.07286601 - 0.0442911j,
                        -0.06431914 - 0.03795923j,
                        -0.06447172 - 0.03919567j,
                    ]
                ),
                axis_name=["Pulse Delay"],
                axis=[
                    np.array(
                        [
                            0.00000000e00,
                            1.66666667e-06,
                            3.33333333e-06,
                            5.00000000e-06,
                            6.66666667e-06,
                            8.33333333e-06,
                            1.00000000e-05,
                            1.16666667e-05,
                            1.33333333e-05,
                            1.50000000e-05,
                            1.66666667e-05,
                            1.83333333e-05,
                            2.00000000e-05,
                            2.16666667e-05,
                            2.33333333e-05,
                            2.50000000e-05,
                            2.66666667e-05,
                            2.83333333e-05,
                            3.00000000e-05,
                            3.16666667e-05,
                            3.33333333e-05,
                            3.50000000e-05,
                            3.66666667e-05,
                            3.83333333e-05,
                            4.00000000e-05,
                            4.16666667e-05,
                            4.33333333e-05,
                            4.50000000e-05,
                            4.66666667e-05,
                            4.83333333e-05,
                            5.00000000e-05,
                        ]
                    )
                ],
            )
        },
        "cal_trace": {
            "q0": {
                "g": AcquiredResult(
                    data=(-0.03565303063567659 + 0.013210466140904797j),
                    axis_name=[],
                    axis=[],
                ),
                "e": AcquiredResult(
                    data=(-0.5430036272699941 - 1.1694973614508577j),
                    axis_name=[],
                    axis=[],
                ),
            }
        },
    }
    sweep_points = data["result"]["q0"].axis[0]
    return RunExperimentResults(data=data), sweep_points


class TestT1AnalysisSingleQubit:
    def test_create_and_run_no_pca(
        self, single_tunable_transmon_platform, results_single_qubit
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = lifetime_measurement.analysis_workflow.options()
        options.do_pca(False)

        result = lifetime_measurement.analysis_workflow(
            result=results_single_qubit[0],
            qubits=q0,
            delays=results_single_qubit[1],
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
            proc_data_dict["q0"]["population"],
            np.array(
                [
                    1.00787209,
                    0.87588896,
                    0.76598219,
                    0.6836751,
                    0.61173751,
                    0.55232468,
                    0.49517968,
                    0.45203702,
                    0.38922181,
                    0.35967265,
                    0.32173561,
                    0.29530025,
                    0.26624666,
                    0.23744594,
                    0.20926094,
                    0.17141808,
                    0.1719566,
                    0.15427311,
                    0.13789753,
                    0.12419218,
                    0.11825001,
                    0.11912618,
                    0.09749957,
                    0.08678555,
                    0.08083805,
                    0.07222622,
                    0.06901647,
                    0.06652873,
                    0.05246193,
                    0.0453221,
                    0.04625179,
                ]
            ),
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
        self, single_tunable_transmon_platform, results_single_qubit
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = lifetime_measurement.analysis_workflow.options()
        options.do_pca(True)

        result = lifetime_measurement.analysis_workflow(
            result=results_single_qubit[0],
            qubits=q0,
            delays=results_single_qubit[1],
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
            proc_data_dict["q0"]["population"],
            np.array(
                [
                    0.90174388,
                    0.7318704,
                    0.5904377,
                    0.48448906,
                    0.39193213,
                    0.31545397,
                    0.24189628,
                    0.1863958,
                    0.10556313,
                    0.06753195,
                    0.01868863,
                    -0.01531271,
                    -0.05272784,
                    -0.08976783,
                    -0.12604162,
                    -0.17476291,
                    -0.17405644,
                    -0.19681663,
                    -0.21787318,
                    -0.23553753,
                    -0.24316679,
                    -0.24204859,
                    -0.26986376,
                    -0.28366148,
                    -0.29131626,
                    -0.30241525,
                    -0.30653079,
                    -0.30972461,
                    -0.32785102,
                    -0.33701348,
                    -0.33581536,
                ]
            ),
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
        self, single_tunable_transmon_platform, results_single_qubit
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = lifetime_measurement.analysis_workflow.options()
        options.do_fitting(False)

        result = lifetime_measurement.analysis_workflow(
            result=results_single_qubit[0],
            qubits=q0,
            delays=results_single_qubit[1],
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
        options = lifetime_measurement.analysis_workflow.options()
        options.do_plotting(False)

        result = lifetime_measurement.analysis_workflow(
            result=results_single_qubit[0],
            qubits=q0,
            delays=results_single_qubit[1],
            options=options,
        ).run()

        assert len(result.tasks) == 3

        task_names = [t.name for t in result.tasks]
        assert "plot_raw_complex_data_1d" not in task_names
        assert "plot_population" not in task_names
        assert "fit_data" in task_names
        assert "extract_qubit_parameters" in task_names


@pytest.fixture()
def results_two_qubit():
    """Results from a lifetime_measurement experiment.

    In the AcquiredResults below, the axis corresponds to the time-delay after the x180
    pulse in the lifetime_measurement experiment, and the data is the raw acquisition
    result obtained in integrated-average mode.
    """
    data = {
        "result": {
            "q0": AcquiredResult(
                data=np.array(
                    [
                        -0.54653355 - 1.17900678j,
                        -0.48369549 - 1.02114036j,
                        -0.42631291 - 0.89184827j,
                        -0.38948615 - 0.79238741j,
                        -0.34899969 - 0.70901739j,
                        -0.32238488 - 0.63723581j,
                        -0.29655117 - 0.56829491j,
                        -0.27073651 - 0.51895399j,
                        -0.2378248 - 0.4451091j,
                        -0.2235502 - 0.40985343j,
                        -0.20836014 - 0.36324451j,
                        -0.19136864 - 0.33351472j,
                        -0.18146525 - 0.29707788j,
                        -0.16225064 - 0.26498941j,
                        -0.14831841 - 0.23149716j,
                        -0.1330217 - 0.18506588j,
                        -0.1307423 - 0.1867978j,
                        -0.12233241 - 0.16564238j,
                        -0.11069799 - 0.14770174j,
                        -0.10880689 - 0.12932072j,
                        -0.10240398 - 0.12374628j,
                        -0.10462664 - 0.12401976j,
                        -0.09051118 - 0.09979015j,
                        -0.08690288 - 0.08633465j,
                        -0.08404272 - 0.07923303j,
                        -0.08276078 - 0.06772339j,
                        -0.07825265 - 0.06516248j,
                        -0.07552934 - 0.06284701j,
                        -0.07286601 - 0.0442911j,
                        -0.06431914 - 0.03795923j,
                        -0.06447172 - 0.03919567j,
                    ]
                ),
                axis_name=["Pulse Delay"],
                axis=[
                    np.array(
                        [
                            0.00000000e00,
                            1.66666667e-06,
                            3.33333333e-06,
                            5.00000000e-06,
                            6.66666667e-06,
                            8.33333333e-06,
                            1.00000000e-05,
                            1.16666667e-05,
                            1.33333333e-05,
                            1.50000000e-05,
                            1.66666667e-05,
                            1.83333333e-05,
                            2.00000000e-05,
                            2.16666667e-05,
                            2.33333333e-05,
                            2.50000000e-05,
                            2.66666667e-05,
                            2.83333333e-05,
                            3.00000000e-05,
                            3.16666667e-05,
                            3.33333333e-05,
                            3.50000000e-05,
                            3.66666667e-05,
                            3.83333333e-05,
                            4.00000000e-05,
                            4.16666667e-05,
                            4.33333333e-05,
                            4.50000000e-05,
                            4.66666667e-05,
                            4.83333333e-05,
                            5.00000000e-05,
                        ]
                    )
                ],
            ),
            "q1": AcquiredResult(
                data=np.array(
                    [
                        1.3490129 - 1.56692922j,
                        1.31563952 - 1.43939693j,
                        1.29771864 - 1.3299734j,
                        1.28892214 - 1.26335408j,
                        1.26399334 - 1.16856345j,
                        1.24884258 - 1.0964593j,
                        1.23855481 - 1.02965055j,
                        1.22586358 - 0.90662414j,
                        1.20138236 - 0.83645464j,
                        1.19803825 - 0.77275828j,
                        1.19423044 - 0.74596497j,
                        1.17856339 - 0.67606153j,
                        1.16849348 - 0.60745522j,
                        1.15412074 - 0.55858912j,
                        1.14680115 - 0.50797351j,
                        1.149896 - 0.4892067j,
                        1.12431104 - 0.41527088j,
                        1.12575395 - 0.38628374j,
                        1.12121596 - 0.33752466j,
                        1.12008699 - 0.33403167j,
                        1.11137737 - 0.26562835j,
                        1.09952577 - 0.24865748j,
                        1.09887772 - 0.21013059j,
                        1.10099079 - 0.20620871j,
                        1.09337096 - 0.14165318j,
                        1.09672665 - 0.1566198j,
                        1.08921433 - 0.12015613j,
                        1.07816431 - 0.05157708j,
                        1.08422192 - 0.05796172j,
                        1.08046267 - 0.01304732j,
                        1.07439737 - 0.03371405j,
                    ]
                ),
                axis_name=["Pulse Delay"],
                axis=[
                    np.array(
                        [
                            0.00000000e00,
                            1.66666667e-06,
                            3.33333333e-06,
                            5.00000000e-06,
                            6.66666667e-06,
                            8.33333333e-06,
                            1.00000000e-05,
                            1.16666667e-05,
                            1.33333333e-05,
                            1.50000000e-05,
                            1.66666667e-05,
                            1.83333333e-05,
                            2.00000000e-05,
                            2.16666667e-05,
                            2.33333333e-05,
                            2.50000000e-05,
                            2.66666667e-05,
                            2.83333333e-05,
                            3.00000000e-05,
                            3.16666667e-05,
                            3.33333333e-05,
                            3.50000000e-05,
                            3.66666667e-05,
                            3.83333333e-05,
                            4.00000000e-05,
                            4.16666667e-05,
                            4.33333333e-05,
                            4.50000000e-05,
                            4.66666667e-05,
                            4.83333333e-05,
                            5.00000000e-05,
                        ]
                    )
                ],
            ),
        },
        "cal_trace": {
            "q0": {
                "g": AcquiredResult(
                    data=(-0.03565303063567659 + 0.013210466140904797j),
                    axis_name=[],
                    axis=[],
                ),
                "e": AcquiredResult(
                    data=(-0.5430036272699941 - 1.1694973614508577j),
                    axis_name=[],
                    axis=[],
                ),
            },
            "q1": {
                "g": AcquiredResult(
                    data=(1.0407949643102572 + 0.31812771722279526j),
                    axis_name=[],
                    axis=[],
                ),
                "e": AcquiredResult(
                    data=(1.3433557761058659 - 1.5500701510144579j),
                    axis_name=[],
                    axis=[],
                ),
            },
        },
    }
    sweep_points = [data["result"]["q0"].axis[0], data["result"]["q1"].axis[0]]
    return RunExperimentResults(data=data), sweep_points


class TestT1AnalysisTwoQubit:
    def test_create_and_run_no_pca(
        self, two_tunable_transmon_platform, results_two_qubit
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = lifetime_measurement.analysis_workflow.options()
        options.do_pca(False)

        result = lifetime_measurement.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            delays=results_two_qubit[1],
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
            proc_data_dict["q0"]["population"],
            np.array(
                [
                    1.00787209,
                    0.87588896,
                    0.76598219,
                    0.6836751,
                    0.61173751,
                    0.55232468,
                    0.49517968,
                    0.45203702,
                    0.38922181,
                    0.35967265,
                    0.32173561,
                    0.29530025,
                    0.26624666,
                    0.23744594,
                    0.20926094,
                    0.17141808,
                    0.1719566,
                    0.15427311,
                    0.13789753,
                    0.12419218,
                    0.11825001,
                    0.11912618,
                    0.09749957,
                    0.08678555,
                    0.08083805,
                    0.07222622,
                    0.06901647,
                    0.06652873,
                    0.05246193,
                    0.0453221,
                    0.04625179,
                ]
            ),
        )
        assert_array_almost_equal(
            proc_data_dict["q1"]["population"],
            np.array(
                [
                    1.00927147,
                    0.93993217,
                    0.88134363,
                    0.84585228,
                    0.79430419,
                    0.75541523,
                    0.71969912,
                    0.65445715,
                    0.61578911,
                    0.58228296,
                    0.56798606,
                    0.53020136,
                    0.49356605,
                    0.46686364,
                    0.43984451,
                    0.43031728,
                    0.38959151,
                    0.37459387,
                    0.34877807,
                    0.34686078,
                    0.31044625,
                    0.30059319,
                    0.28044304,
                    0.27857591,
                    0.24426043,
                    0.25235041,
                    0.23269656,
                    0.19599268,
                    0.19983458,
                    0.17608993,
                    0.18635721,
                ]
            ),
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

    def test_create_and_run_pca(self, two_tunable_transmon_platform, results_two_qubit):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = lifetime_measurement.analysis_workflow.options()
        options.do_pca(True)

        result = lifetime_measurement.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            delays=results_two_qubit[1],
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
            proc_data_dict["q0"]["population"],
            np.array(
                [
                    0.90174388,
                    0.7318704,
                    0.5904377,
                    0.48448906,
                    0.39193213,
                    0.31545397,
                    0.24189628,
                    0.1863958,
                    0.10556313,
                    0.06753195,
                    0.01868863,
                    -0.01531271,
                    -0.05272784,
                    -0.08976783,
                    -0.12604162,
                    -0.17476291,
                    -0.17405644,
                    -0.19681663,
                    -0.21787318,
                    -0.23553753,
                    -0.24316679,
                    -0.24204859,
                    -0.26986376,
                    -0.28366148,
                    -0.29131626,
                    -0.30241525,
                    -0.30653079,
                    -0.30972461,
                    -0.32785102,
                    -0.33701348,
                    -0.33581536,
                ]
            ),
        )
        assert_array_almost_equal(
            proc_data_dict["q1"]["population"],
            np.array(
                [
                    1.00155742,
                    0.87022906,
                    0.75935018,
                    0.69220034,
                    0.59456744,
                    0.5209423,
                    0.45335495,
                    0.32994639,
                    0.25665924,
                    0.19330767,
                    0.16625558,
                    0.09471298,
                    0.02539031,
                    -0.0251969,
                    -0.07632259,
                    -0.09430166,
                    -0.1714871,
                    -0.19981876,
                    -0.24864656,
                    -0.25227966,
                    -0.32117349,
                    -0.33989574,
                    -0.37798291,
                    -0.38149358,
                    -0.44641115,
                    -0.43109335,
                    -0.46830105,
                    -0.53776168,
                    -0.53044926,
                    -0.57535619,
                    -0.55600379,
                ]
            ),
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
        self, two_tunable_transmon_platform, results_two_qubit
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = lifetime_measurement.analysis_workflow.options()
        options.do_fitting(False)

        result = lifetime_measurement.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            delays=results_two_qubit[1],
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
        self, two_tunable_transmon_platform, results_two_qubit
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = lifetime_measurement.analysis_workflow.options()
        options.do_plotting(False)

        result = lifetime_measurement.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            delays=results_two_qubit[1],
            options=options,
        ).run()

        assert len(result.tasks) == 3

        task_names = [t.name for t in result.tasks]
        assert "plot_raw_complex_data_1d" not in task_names
        assert "plot_population" not in task_names
        assert "fit_data" in task_names
        assert "extract_qubit_parameters" in task_names

    def test_create_and_run_close_figures(
        self, two_tunable_transmon_platform, results_two_qubit
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = lifetime_measurement.analysis_workflow.options()
        options.close_figures(True)

        result = lifetime_measurement.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            delays=results_two_qubit[1],
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
            result=results_two_qubit[0],
            qubits=qubits,
            delays=results_two_qubit[1],
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
