"""Tests for the Rmsey analysis using the testing utilities."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from laboneq_applications.analysis import ramsey
from laboneq_applications.tasks.run_experiment import (
    AcquiredResult,
    RunExperimentResults,
)


@pytest.fixture()
def results_single_qubit():
    """Results from a Ramsey experiment.

    In the AcquiredResults below, the axis corresponds to the time-separation between
    the two x90 pulses in the Ramsey experiment, and the data is the raw acquisition
    result obtained in integrated-average mode.
    """
    data = {}
    data["result/q0"] = AcquiredResult(
        data=np.array(
            [
                -0.41226907 - 0.23724557j,
                -0.75385551 - 1.8598465j,
                -0.46281302 - 0.47365712j,
                -0.69077439 - 1.55162374j,
                -0.53665409 - 0.81242755j,
                -0.61542435 - 1.18807677j,
                -0.60997641 - 1.19778293j,
                -0.51583823 - 0.75198013j,
                -0.68631143 - 1.55106611j,
                -0.46757106 - 0.52384261j,
                -0.71540125 - 1.68984323j,
                -0.46559991 - 0.48890644j,
                -0.71074036 - 1.66832332j,
                -0.49210168 - 0.62910852j,
                -0.65303434 - 1.3879671j,
                -0.56044572 - 0.95590255j,
                -0.60095449 - 1.14913627j,
                -0.61519585 - 1.2105019j,
                -0.53415959 - 0.84759682j,
                -0.66618334 - 1.4577474j,
                -0.49180478 - 0.63857139j,
                -0.6845682 - 1.53714178j,
                -0.4929213 - 0.6358012j,
                -0.67203595 - 1.47718335j,
                -0.51922963 - 0.77509854j,
                -0.63387641 - 1.30319433j,
                -0.56672214 - 0.99294965j,
                -0.5853124 - 1.08259647j,
                -0.61732432 - 1.24854965j,
                -0.54548262 - 0.8940879j,
                -0.65211728 - 1.38617325j,
                -0.52224713 - 0.80173621j,
                -0.6586447 - 1.4388047j,
                -0.51634287 - 0.78281146j,
                -0.65612822 - 1.40612921j,
                -0.54485717 - 0.89232884j,
                -0.61589464 - 1.2442815j,
                -0.57644989 - 1.04338838j,
                -0.58006123 - 1.07135888j,
                -0.60668187 - 1.18829886j,
                -0.55594966 - 0.92351248j,
            ]
        ),
        axis_name=["Pulse Delay"],
        axis=[
            np.array(
                [
                    0.0000e00,
                    6.7500e-07,
                    1.3500e-06,
                    2.0250e-06,
                    2.7000e-06,
                    3.3750e-06,
                    4.0500e-06,
                    4.7250e-06,
                    5.4000e-06,
                    6.0750e-06,
                    6.7500e-06,
                    7.4250e-06,
                    8.1000e-06,
                    8.7750e-06,
                    9.4500e-06,
                    1.0125e-05,
                    1.0800e-05,
                    1.1475e-05,
                    1.2150e-05,
                    1.2825e-05,
                    1.3500e-05,
                    1.4175e-05,
                    1.4850e-05,
                    1.5525e-05,
                    1.6200e-05,
                    1.6875e-05,
                    1.7550e-05,
                    1.8225e-05,
                    1.8900e-05,
                    1.9575e-05,
                    2.0250e-05,
                    2.0925e-05,
                    2.1600e-05,
                    2.2275e-05,
                    2.2950e-05,
                    2.3625e-05,
                    2.4300e-05,
                    2.4975e-05,
                    2.5650e-05,
                    2.6325e-05,
                    2.7000e-05,
                ]
            )
        ],
    )
    data["cal_trace/q0/g"] = AcquiredResult(
        data=(-0.772875386725562 - 1.9347566625390387j),
        axis_name=[],
        axis=[],
    )
    data["cal_trace/q0/e"] = AcquiredResult(
        data=(-0.4094606327325466 - 0.24116128694103414j),
        axis_name=[],
        axis=[],
    )
    sweep_points = data["result/q0"].axis[0]
    return RunExperimentResults(data=data), sweep_points


class TestRamseyAnalysisSingleQubit:
    def test_create_and_run_no_pca(
        self, single_tunable_transmon_platform, results_single_qubit
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = ramsey.analysis_workflow.options()
        options.do_pca(False)

        result = ramsey.analysis_workflow(
            result=results_single_qubit[0],
            qubits=q0,
            delays=results_single_qubit[1],
            detunings=0.67e6,
            options=options,
        ).run()

        assert len(result.tasks) == 5

        for task in result.tasks:
            if task.name != "extract_qubit_parameters":
                assert "q0" in task.output
        assert "new_parameter_values" in result.tasks["extract_qubit_parameters"].output

        proc_data_dict = result.tasks["calculate_qubit_population"].output
        assert proc_data_dict["q0"]["num_cal_traces"] == 2
        np.testing.assert_array_almost_equal(
            proc_data_dict["q0"]["population"],
            np.array(
                [
                    1.00187013,
                    0.04458822,
                    0.8623011,
                    0.22621098,
                    0.66213183,
                    0.4405486,
                    0.43572966,
                    0.69877382,
                    0.22706632,
                    0.83339665,
                    0.14520749,
                    0.85335577,
                    0.15791936,
                    0.77100608,
                    0.32316128,
                    0.57826294,
                    0.46428195,
                    0.427918,
                    0.64258203,
                    0.28217982,
                    0.76570055,
                    0.23513732,
                    0.76712899,
                    0.27049994,
                    0.68531342,
                    0.37333335,
                    0.55659078,
                    0.50373623,
                    0.40618343,
                    0.61496778,
                    0.32428493,
                    0.6699118,
                    0.29378549,
                    0.68130938,
                    0.31253459,
                    0.61603647,
                    0.40876583,
                    0.52694143,
                    0.51071554,
                    0.44148217,
                    0.5970907,
                ]
            ),
        )

        fit_values = result.tasks["fit_data"].output["q0"].best_values
        np.testing.assert_allclose(
            fit_values["frequency"], 672104.6105811436, rtol=1e-4
        )
        np.testing.assert_allclose(
            fit_values["decay_time"], 2.315185344525794e-05, rtol=1e-4
        )
        np.testing.assert_allclose(fit_values["phase"], 6.260565448213821, rtol=1e-4)

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["resonance_frequency_ge"],
            6.5e9,
        )
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["ge_T2_star"],
            0.0,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "resonance_frequency_ge"
            ].nominal_value,
            6499997895.389419,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "resonance_frequency_ge"
            ].std_dev,
            197.39542633510794,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["ge_T2_star"].nominal_value,
            2.315185344525794e-05,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["ge_T2_star"].std_dev,
            4.317526247580677e-07,
            rtol=1e-4,
        )

    def test_create_and_run_pca(
        self, single_tunable_transmon_platform, results_single_qubit
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = ramsey.analysis_workflow.options()
        options.do_pca(True)

        result = ramsey.analysis_workflow(
            result=results_single_qubit[0],
            qubits=q0,
            delays=results_single_qubit[1],
            detunings=0.67e6,
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
        np.testing.assert_array_almost_equal(
            proc_data_dict["q0"]["population"],
            np.array(
                [
                    -8.63689126e-01,
                    7.94475126e-01,
                    -6.21935193e-01,
                    4.79871489e-01,
                    -2.75216089e-01,
                    1.08601964e-01,
                    1.16966465e-01,
                    -3.38667120e-01,
                    4.78399951e-01,
                    -5.71854672e-01,
                    6.20192732e-01,
                    -6.06439493e-01,
                    5.98174064e-01,
                    -4.63789777e-01,
                    3.11945787e-01,
                    -1.29927018e-01,
                    6.75065487e-02,
                    1.30491626e-01,
                    -2.41329833e-01,
                    3.82935742e-01,
                    -4.54594461e-01,
                    4.64416955e-01,
                    -4.57072692e-01,
                    4.03163080e-01,
                    -3.15348153e-01,
                    2.25042828e-01,
                    -9.23838800e-02,
                    -8.30743236e-04,
                    1.68152957e-01,
                    -1.93501110e-01,
                    3.10000694e-01,
                    -2.88664106e-01,
                    3.62841089e-01,
                    -3.08402084e-01,
                    3.30354581e-01,
                    -1.95351661e-01,
                    1.63681049e-01,
                    -4.10244401e-02,
                    -1.29133461e-02,
                    1.07005213e-01,
                    -1.62545062e-01,
                ]
            ),
        )

        fit_values = result.tasks["fit_data"].output["q0"].best_values
        np.testing.assert_allclose(
            fit_values["frequency"], 672036.0251008251, rtol=1e-4
        )
        np.testing.assert_allclose(
            fit_values["decay_time"], 2.407338911414314e-05, rtol=1e-4
        )
        np.testing.assert_allclose(fit_values["phase"], 3.125414097958747, rtol=1e-4)

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["resonance_frequency_ge"],
            6.5e9,
        )
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["ge_T2_star"],
            0.0,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "resonance_frequency_ge"
            ].nominal_value,
            6499997963.974899,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "resonance_frequency_ge"
            ].std_dev,
            194.85970420477435,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["ge_T2_star"].nominal_value,
            2.407338911414314e-05,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["ge_T2_star"].std_dev,
            6.942122673142818e-07,
            rtol=1e-4,
        )

    def test_create_and_run_no_fitting(
        self, single_tunable_transmon_platform, results_single_qubit
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = ramsey.analysis_workflow.options()
        options.do_fitting(False)

        result = ramsey.analysis_workflow(
            result=results_single_qubit[0],
            qubits=q0,
            delays=results_single_qubit[1],
            detunings=0.67e6,
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
        options = ramsey.analysis_workflow.options()
        options.do_plotting(False)

        result = ramsey.analysis_workflow(
            result=results_single_qubit[0],
            qubits=q0,
            delays=results_single_qubit[1],
            detunings=0.67e6,
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
    """Results from a Ramsey experiment.

    In the AcquiredResults below, the axis corresponds to the time-separation between
    the two x90 pulses in the Ramsey experiment, and the data is the raw acquisition
    result obtained in integrated-average mode.
    """
    data = {}
    # q0
    data["result/q0"] = AcquiredResult(
        data=np.array(
            [
                -0.41226907 - 0.23724557j,
                -0.75385551 - 1.8598465j,
                -0.46281302 - 0.47365712j,
                -0.69077439 - 1.55162374j,
                -0.53665409 - 0.81242755j,
                -0.61542435 - 1.18807677j,
                -0.60997641 - 1.19778293j,
                -0.51583823 - 0.75198013j,
                -0.68631143 - 1.55106611j,
                -0.46757106 - 0.52384261j,
                -0.71540125 - 1.68984323j,
                -0.46559991 - 0.48890644j,
                -0.71074036 - 1.66832332j,
                -0.49210168 - 0.62910852j,
                -0.65303434 - 1.3879671j,
                -0.56044572 - 0.95590255j,
                -0.60095449 - 1.14913627j,
                -0.61519585 - 1.2105019j,
                -0.53415959 - 0.84759682j,
                -0.66618334 - 1.4577474j,
                -0.49180478 - 0.63857139j,
                -0.6845682 - 1.53714178j,
                -0.4929213 - 0.6358012j,
                -0.67203595 - 1.47718335j,
                -0.51922963 - 0.77509854j,
                -0.63387641 - 1.30319433j,
                -0.56672214 - 0.99294965j,
                -0.5853124 - 1.08259647j,
                -0.61732432 - 1.24854965j,
                -0.54548262 - 0.8940879j,
                -0.65211728 - 1.38617325j,
                -0.52224713 - 0.80173621j,
                -0.6586447 - 1.4388047j,
                -0.51634287 - 0.78281146j,
                -0.65612822 - 1.40612921j,
                -0.54485717 - 0.89232884j,
                -0.61589464 - 1.2442815j,
                -0.57644989 - 1.04338838j,
                -0.58006123 - 1.07135888j,
                -0.60668187 - 1.18829886j,
                -0.55594966 - 0.92351248j,
            ]
        ),
        axis_name=["Pulse Delay"],
        axis=[
            np.array(
                [
                    0.0000e00,
                    6.7500e-07,
                    1.3500e-06,
                    2.0250e-06,
                    2.7000e-06,
                    3.3750e-06,
                    4.0500e-06,
                    4.7250e-06,
                    5.4000e-06,
                    6.0750e-06,
                    6.7500e-06,
                    7.4250e-06,
                    8.1000e-06,
                    8.7750e-06,
                    9.4500e-06,
                    1.0125e-05,
                    1.0800e-05,
                    1.1475e-05,
                    1.2150e-05,
                    1.2825e-05,
                    1.3500e-05,
                    1.4175e-05,
                    1.4850e-05,
                    1.5525e-05,
                    1.6200e-05,
                    1.6875e-05,
                    1.7550e-05,
                    1.8225e-05,
                    1.8900e-05,
                    1.9575e-05,
                    2.0250e-05,
                    2.0925e-05,
                    2.1600e-05,
                    2.2275e-05,
                    2.2950e-05,
                    2.3625e-05,
                    2.4300e-05,
                    2.4975e-05,
                    2.5650e-05,
                    2.6325e-05,
                    2.7000e-05,
                ]
            )
        ],
    )
    data["cal_trace/q0/g"] = AcquiredResult(
        data=(-0.772875386725562 - 1.9347566625390387j),
        axis_name=[],
        axis=[],
    )
    data["cal_trace/q0/e"] = AcquiredResult(
        data=(-0.4094606327325466 - 0.24116128694103414j),
        axis_name=[],
        axis=[],
    )
    # q1
    data["result/q1"] = AcquiredResult(
        data=np.array(
            [
                -0.5869649 - 1.27564237j,
                -0.06979067 - 0.04651793j,
                -0.52140313 - 1.13092777j,
                -0.15530429 - 0.24786556j,
                -0.41618941 - 0.87664418j,
                -0.28232346 - 0.53313041j,
                -0.30380519 - 0.60418823j,
                -0.37326421 - 0.77317024j,
                -0.21679671 - 0.37867545j,
                -0.45246275 - 0.95257119j,
                -0.15223185 - 0.23645036j,
                -0.47587615 - 1.01152641j,
                -0.164902 - 0.25813813j,
                -0.45537203 - 0.95003236j,
                -0.20382703 - 0.37091566j,
                -0.39665562 - 0.80739756j,
                -0.28535791 - 0.55007901j,
                -0.31854509 - 0.63487318j,
                -0.34464685 - 0.69619619j,
                -0.26711436 - 0.48180142j,
                -0.40037765 - 0.83618287j,
                -0.22242498 - 0.40178687j,
                -0.40301845 - 0.84332809j,
                -0.21649162 - 0.40308903j,
                -0.40343382 - 0.84457504j,
                -0.25659611 - 0.47227615j,
                -0.37110368 - 0.75313587j,
                -0.27835112 - 0.54641035j,
                -0.31799437 - 0.64284319j,
                -0.32466994 - 0.65934115j,
                -0.27930893 - 0.5426351j,
                -0.36734748 - 0.72820048j,
                -0.25463495 - 0.47372374j,
                -0.37628542 - 0.77412822j,
                -0.25860092 - 0.48829962j,
                -0.37038367 - 0.75312633j,
                -0.26882928 - 0.51313816j,
                -0.34823198 - 0.7036106j,
                -0.29279998 - 0.57328701j,
                -0.32880129 - 0.64904714j,
                -0.32547872 - 0.64817809j,
            ]
        ),
        axis_name=["Pulse Delay"],
        axis=[
            np.array(
                [
                    0.0000e00,
                    6.7500e-07,
                    1.3500e-06,
                    2.0250e-06,
                    2.7000e-06,
                    3.3750e-06,
                    4.0500e-06,
                    4.7250e-06,
                    5.4000e-06,
                    6.0750e-06,
                    6.7500e-06,
                    7.4250e-06,
                    8.1000e-06,
                    8.7750e-06,
                    9.4500e-06,
                    1.0125e-05,
                    1.0800e-05,
                    1.1475e-05,
                    1.2150e-05,
                    1.2825e-05,
                    1.3500e-05,
                    1.4175e-05,
                    1.4850e-05,
                    1.5525e-05,
                    1.6200e-05,
                    1.6875e-05,
                    1.7550e-05,
                    1.8225e-05,
                    1.8900e-05,
                    1.9575e-05,
                    2.0250e-05,
                    2.0925e-05,
                    2.1600e-05,
                    2.2275e-05,
                    2.2950e-05,
                    2.3625e-05,
                    2.4300e-05,
                    2.4975e-05,
                    2.5650e-05,
                    2.6325e-05,
                    2.7000e-05,
                ]
            )
        ],
    )
    data["cal_trace/q1/g"] = AcquiredResult(
        data=(-0.047990956067688245 + 0.01885716311907388j),
        axis_name=[],
        axis=[],
    )
    data["cal_trace/q1/e"] = AcquiredResult(
        data=(-0.5862257200659212 - 1.2763478584265495j),
        axis_name=[],
        axis=[],
    )
    sweep_points = [data["result/q0"].axis[0], data["result/q1"].axis[0]]
    return RunExperimentResults(data=data), sweep_points


class TestRamseyAnalysisTwoQubit:
    def test_create_and_run_no_pca(
        self, two_tunable_transmon_platform, results_two_qubit
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = ramsey.analysis_workflow.options()
        options.do_pca(False)

        result = ramsey.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            delays=results_two_qubit[1],
            detunings=[0.67e6, 0.67e6],
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
        np.testing.assert_array_almost_equal(
            proc_data_dict["q0"]["population"],
            np.array(
                [
                    1.00187013,
                    0.04458822,
                    0.8623011,
                    0.22621098,
                    0.66213183,
                    0.4405486,
                    0.43572966,
                    0.69877382,
                    0.22706632,
                    0.83339665,
                    0.14520749,
                    0.85335577,
                    0.15791936,
                    0.77100608,
                    0.32316128,
                    0.57826294,
                    0.46428195,
                    0.427918,
                    0.64258203,
                    0.28217982,
                    0.76570055,
                    0.23513732,
                    0.76712899,
                    0.27049994,
                    0.68531342,
                    0.37333335,
                    0.55659078,
                    0.50373623,
                    0.40618343,
                    0.61496778,
                    0.32428493,
                    0.6699118,
                    0.29378549,
                    0.68130938,
                    0.31253459,
                    0.61603647,
                    0.40876583,
                    0.52694143,
                    0.51071554,
                    0.44148217,
                    0.5970907,
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            proc_data_dict["q1"]["population"],
            np.array(
                [
                    0.99973776,
                    0.04900617,
                    0.88652266,
                    0.20496623,
                    0.69032059,
                    0.42753174,
                    0.48019231,
                    0.61045092,
                    0.30791334,
                    0.75023389,
                    0.19661005,
                    0.79545483,
                    0.21435542,
                    0.74935835,
                    0.29925596,
                    0.63938542,
                    0.43952062,
                    0.50442755,
                    0.55194292,
                    0.38957646,
                    0.65935551,
                    0.32466938,
                    0.66478231,
                    0.32390335,
                    0.66571693,
                    0.38042742,
                    0.59666953,
                    0.4351882,
                    0.50952419,
                    0.52221258,
                    0.4329647,
                    0.57922482,
                    0.38084392,
                    0.61190824,
                    0.39152551,
                    0.59646626,
                    0.41067722,
                    0.55780532,
                    0.4568365,
                    0.51656551,
                    0.51508429,
                ]
            ),
        )

        fit_values = result.tasks["fit_data"].output["q0"].best_values
        np.testing.assert_allclose(
            fit_values["frequency"], 672104.6105811436, rtol=1e-4
        )
        np.testing.assert_allclose(
            fit_values["decay_time"], 2.315185344525794e-05, rtol=1e-4
        )
        np.testing.assert_allclose(fit_values["phase"], 6.260565448213821, rtol=1e-4)

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["resonance_frequency_ge"],
            6.5e9,
        )
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["ge_T2_star"],
            0.0,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "resonance_frequency_ge"
            ].nominal_value,
            6499997895.389419,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "resonance_frequency_ge"
            ].std_dev,
            197.39542633510794,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["ge_T2_star"].nominal_value,
            2.315185344525794e-05,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["ge_T2_star"].std_dev,
            4.317526247580677e-07,
            rtol=1e-4,
        )

        fit_values = result.tasks["fit_data"].output["q1"].best_values
        np.testing.assert_allclose(
            fit_values["frequency"], 675168.9305114065, rtol=1e-4
        )
        np.testing.assert_allclose(
            fit_values["decay_time"], 1.4922738963196736e-05, rtol=1e-4
        )
        np.testing.assert_allclose(fit_values["phase"], 6.3101229841169, rtol=1e-4)

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q1"]["resonance_frequency_ge"],
            6.51e9,
        )
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q1"]["ge_T2_star"],
            0.0,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q1"][
                "resonance_frequency_ge"
            ].nominal_value,
            6509994831.069489,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q1"][
                "resonance_frequency_ge"
            ].std_dev,
            196.14146118058264,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q1"]["ge_T2_star"].nominal_value,
            1.4922738963196736e-05,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q1"]["ge_T2_star"].std_dev,
            1.7488402080301885e-07,
            rtol=1e-4,
        )

    def test_create_and_run_pca(self, two_tunable_transmon_platform, results_two_qubit):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = ramsey.analysis_workflow.options()
        options.do_pca(True)

        result = ramsey.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            delays=results_two_qubit[1],
            detunings=[0.67e6, 0.67e6],
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
        np.testing.assert_array_almost_equal(
            proc_data_dict["q0"]["population"],
            np.array(
                [
                    -8.63689126e-01,
                    7.94475126e-01,
                    -6.21935193e-01,
                    4.79871489e-01,
                    -2.75216089e-01,
                    1.08601964e-01,
                    1.16966465e-01,
                    -3.38667120e-01,
                    4.78399951e-01,
                    -5.71854672e-01,
                    6.20192732e-01,
                    -6.06439493e-01,
                    5.98174064e-01,
                    -4.63789777e-01,
                    3.11945787e-01,
                    -1.29927018e-01,
                    6.75065487e-02,
                    1.30491626e-01,
                    -2.41329833e-01,
                    3.82935742e-01,
                    -4.54594461e-01,
                    4.64416955e-01,
                    -4.57072692e-01,
                    4.03163080e-01,
                    -3.15348153e-01,
                    2.25042828e-01,
                    -9.23838800e-02,
                    -8.30743236e-04,
                    1.68152957e-01,
                    -1.93501110e-01,
                    3.10000694e-01,
                    -2.88664106e-01,
                    3.62841089e-01,
                    -3.08402084e-01,
                    3.30354581e-01,
                    -1.95351661e-01,
                    1.63681049e-01,
                    -4.10244401e-02,
                    -1.29133461e-02,
                    1.07005213e-01,
                    -1.62545062e-01,
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            proc_data_dict["q1"]["population"],
            np.array(
                [
                    0.69377974,
                    -0.63970926,
                    0.53498155,
                    -0.42096024,
                    0.25979134,
                    -0.10878623,
                    -0.03493129,
                    0.14776722,
                    -0.2765625,
                    0.34382858,
                    -0.43267924,
                    0.40725406,
                    -0.40778703,
                    0.34260356,
                    -0.28871264,
                    0.18835735,
                    -0.09197382,
                    -0.00093774,
                    0.06570719,
                    -0.16201718,
                    0.21636088,
                    -0.25306372,
                    0.22397219,
                    -0.25414306,
                    0.22528299,
                    -0.17485449,
                    0.1284423,
                    -0.09805458,
                    0.00620781,
                    0.02400424,
                    -0.10117132,
                    0.10397957,
                    -0.17427226,
                    0.14981322,
                    -0.15929199,
                    0.12815664,
                    -0.13243011,
                    0.0739301,
                    -0.06768847,
                    0.01609013,
                    0.01401034,
                ]
            ),
        )

        fit_values = result.tasks["fit_data"].output["q0"].best_values
        np.testing.assert_allclose(
            fit_values["frequency"], 672036.0251008251, rtol=1e-4
        )
        np.testing.assert_allclose(
            fit_values["decay_time"], 2.407338911414314e-05, rtol=1e-4
        )
        np.testing.assert_allclose(fit_values["phase"], 3.125414097958747, rtol=1e-4)

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["resonance_frequency_ge"],
            6.5e9,
        )
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["ge_T2_star"],
            0.0,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "resonance_frequency_ge"
            ].nominal_value,
            6499997963.974899,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "resonance_frequency_ge"
            ].std_dev,
            194.85970420477435,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["ge_T2_star"].nominal_value,
            2.407338911414314e-05,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"]["ge_T2_star"].std_dev,
            6.942122673142818e-07,
            rtol=1e-4,
        )

        fit_values = result.tasks["fit_data"].output["q1"].best_values
        np.testing.assert_allclose(
            fit_values["frequency"], 675046.9161659325, rtol=1e-4
        )
        np.testing.assert_allclose(
            fit_values["decay_time"], 1.536118748335369e-05, rtol=1e-4
        )
        np.testing.assert_allclose(fit_values["phase"], 6.319147951061621, rtol=1e-4)

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q1"]["resonance_frequency_ge"],
            6.51e9,
        )
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q1"]["ge_T2_star"],
            0.0,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q1"][
                "resonance_frequency_ge"
            ].nominal_value,
            6509994953.083834,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q1"][
                "resonance_frequency_ge"
            ].std_dev,
            187.21824130045195,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q1"]["ge_T2_star"].nominal_value,
            1.536118748335369e-05,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q1"]["ge_T2_star"].std_dev,
            2.4293140974304043e-07,
            rtol=1e-4,
        )

    def test_create_and_run_no_fitting(
        self, two_tunable_transmon_platform, results_two_qubit
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = ramsey.analysis_workflow.options()
        options.do_fitting(False)

        result = ramsey.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            delays=results_two_qubit[1],
            detunings=[0.67e6, 0.67e6],
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
        options = ramsey.analysis_workflow.options()
        options.do_plotting(False)

        result = ramsey.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            delays=results_two_qubit[1],
            detunings=[0.67e6, 0.67e6],
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
        options = ramsey.analysis_workflow.options()
        options.close_figures(True)

        result = ramsey.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            delays=results_two_qubit[1],
            detunings=[0.67e6, 0.67e6],
            options=options,
        ).run()

        assert isinstance(
            result.tasks["plot_population"].output["q0"], mpl.figure.Figure
        )
        assert isinstance(
            result.tasks["plot_population"].output["q1"], mpl.figure.Figure
        )

        options.close_figures(False)
        result = ramsey.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            delays=results_two_qubit[1],
            detunings=[0.67e6, 0.67e6],
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
