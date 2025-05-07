# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from freezegun import freeze_time
from laboneq.workflow import handles
from laboneq.workflow.tasks.run_experiment import (
    AcquiredResult,
    RunExperimentResults,
)
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from laboneq_applications.analysis import amplitude_rabi


@pytest.fixture
def results_single_qubit():
    """Results from AmplitudeRabi experiment."""
    data = {}
    data[handles.result_handle("q0")] = AcquiredResult(
        data=np.array(
            [
                0.05290302 - 0.13215136j,
                0.06067577 - 0.12907117j,
                0.05849071 - 0.09401458j,
                0.0683788 - 0.04265771j,
                0.07369121 + 0.0238058j,
                0.08271086 + 0.10077513j,
                0.09092848 + 0.1884216j,
                0.1063583 + 0.28337206j,
                0.11472132 + 0.38879551j,
                0.13147716 + 0.49203866j,
                0.13378882 + 0.59027211j,
                0.15108762 + 0.70302525j,
                0.16102455 + 0.77474721j,
                0.16483135 + 0.83853894j,
                0.17209631 + 0.88743935j,
                0.17435144 + 0.90659384j,
                0.17877636 + 0.92026812j,
                0.17153804 + 0.90921755j,
                0.17243493 + 0.88099388j,
                0.164842 + 0.82561295j,
                0.15646681 + 0.76574749j,
            ]
        ),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q0", "g")] = AcquiredResult(
        data=(0.05745863888207082 - 0.13026141779382786j),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q0", "e")] = AcquiredResult(
        data=(0.1770431406621688 + 0.91612948998106j),
        axis_name=[],
        axis=[],
    )
    sweep_points = np.array(
        [
            0.0,
            0.0238155,
            0.04763101,
            0.07144651,
            0.09526201,
            0.11907752,
            0.14289302,
            0.16670852,
            0.19052403,
            0.21433953,
            0.23815503,
            0.26197054,
            0.28578604,
            0.30960154,
            0.33341705,
            0.35723255,
            0.38104805,
            0.40486356,
            0.42867906,
            0.45249456,
            0.47631007,
        ]
    )
    return RunExperimentResults(data=data), sweep_points


@pytest.fixture
def monkeypatch_tzlocal(monkeypatch):
    import dateutil.tz
    import laboneq.workflow.timestamps

    fake_local_tz = dateutil.tz.tzoffset("GMT+2", 7200)

    def fake_tzlocal():
        return fake_local_tz

    monkeypatch.setattr(laboneq.workflow.timestamps, "tzlocal", fake_tzlocal)

    return fake_local_tz


@freeze_time("2024-07-28 17:55:00", tz_offset=0)
@pytest.mark.usefixtures("monkeypatch_tzlocal")
class TestRabiAnalysisSingleQubit:
    def test_create_and_run_no_pca(
        self, single_tunable_transmon_platform, results_single_qubit
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = amplitude_rabi.analysis_workflow.options()
        options.do_pca(value=False)

        result = amplitude_rabi.analysis_workflow(
            result=results_single_qubit[0],
            qubits=q0,
            amplitudes=results_single_qubit[1],
            options=options,
        ).run()

        assert len(result.tasks) == 5

        for task in result.tasks:
            if task.name in ["calculate_qubit_population", "fit_data"]:
                assert "q0" in task.output
        assert "new_parameter_values" in result.tasks["extract_qubit_parameters"].output

        proc_data_dict = result.tasks["calculate_qubit_population"].output
        assert proc_data_dict["q0"]["num_cal_traces"] == 2
        assert_array_almost_equal(
            proc_data_dict["q0"]["population"],
            np.array(
                [
                    -0.002274,
                    0.00146965,
                    0.03430455,
                    0.08381782,
                    0.14708857,
                    0.22066961,
                    0.30423641,
                    0.39547093,
                    0.49582323,
                    0.59502358,
                    0.68794084,
                    0.79617092,
                    0.86490077,
                    0.9254888,
                    0.97240199,
                    0.99071441,
                    1.00409101,
                    0.99288616,
                    0.96635819,
                    0.91329628,
                    0.85591957,
                ]
            ),
        )

        fit_values = result.tasks["fit_data"].output["q0"].best_values
        assert_almost_equal(fit_values["frequency"], 8.3203418993638)
        assert_almost_equal(fit_values["phase"], 3.1237529536013273)
        assert_almost_equal(fit_values["amplitude"], 0.5041511237243046)
        assert_almost_equal(fit_values["offset"], 0.502026090780613)

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["ge_drive_amplitude_pi"],
            0.8,
        )
        assert_almost_equal(
            qubit_parameters["new_parameter_values"]["q0"][
                "ge_drive_amplitude_pi"
            ].nominal_value,
            0.3797238613259197,
        )
        assert_almost_equal(
            qubit_parameters["new_parameter_values"]["q0"][
                "ge_drive_amplitude_pi"
            ].std_dev,
            0.002875704340756095,
        )

    def test_create_and_run_pca(
        self,
        single_tunable_transmon_platform,
        results_single_qubit,
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = amplitude_rabi.analysis_workflow.options()
        options.do_pca(True)

        result = amplitude_rabi.analysis_workflow(
            result=results_single_qubit[0],
            qubits=q0,
            amplitudes=results_single_qubit[1],
            options=options,
        ).run()

        assert len(result.tasks) == 5

        for task in result.tasks:
            if task.name in ["calculate_qubit_population", "fit_data"]:
                assert "q0" in task.output
        assert "new_parameter_values" in result.tasks["extract_qubit_parameters"].output

        proc_data_dict = result.tasks["calculate_qubit_population"].output
        assert proc_data_dict["q0"]["num_cal_traces"] == 2
        assert_array_almost_equal(
            proc_data_dict["q0"]["population"],
            np.array(
                [
                    0.60893583,
                    0.60499912,
                    0.57041226,
                    0.5182681,
                    0.45162937,
                    0.37413388,
                    0.28611972,
                    0.19003514,
                    0.0843409,
                    -0.02013298,
                    -0.11800093,
                    -0.23198539,
                    -0.30437035,
                    -0.36818462,
                    -0.41759229,
                    -0.4368789,
                    -0.45096481,
                    -0.43916876,
                    -0.41122607,
                    -0.35534221,
                    -0.29491423,
                ]
            ),
        )

        fit_values = result.tasks["fit_data"].output["q0"].best_values
        assert_almost_equal(fit_values["frequency"], 8.32034466074661)
        assert_almost_equal(fit_values["phase"], 6.265343866673316)
        assert_almost_equal(fit_values["amplitude"], 0.530972990501319)
        assert_almost_equal(fit_values["offset"], 0.07780831562589109)

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["ge_drive_amplitude_pi"],
            0.8,
        )
        assert_almost_equal(
            qubit_parameters["new_parameter_values"]["q0"][
                "ge_drive_amplitude_pi"
            ].nominal_value,
            0.3797239444901262,
        )
        assert_almost_equal(
            qubit_parameters["new_parameter_values"]["q0"][
                "ge_drive_amplitude_pi"
            ].std_dev,
            0.002875751217783006,
        )

        population_plot = result.tasks["plot_population"].output["q0"]
        [ax] = population_plot.axes
        assert ax.title.get_text() == "20240728T195500 - Amplitude Rabi q0"

    def test_create_and_run_no_fitting(
        self, single_tunable_transmon_platform, results_single_qubit
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = amplitude_rabi.analysis_workflow.options()
        options.do_fitting(value=False)

        result = amplitude_rabi.analysis_workflow(
            result=results_single_qubit[0],
            qubits=q0,
            amplitudes=results_single_qubit[1],
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
        options = amplitude_rabi.analysis_workflow.options()
        options.do_plotting(value=False)

        result = amplitude_rabi.analysis_workflow(
            result=results_single_qubit[0],
            qubits=q0,
            amplitudes=results_single_qubit[1],
            options=options,
        ).run()

        assert len(result.tasks) == 3

        task_names = [t.name for t in result.tasks]
        assert "plot_raw_complex_data_1d" not in task_names
        assert "plot_population" not in task_names
        assert "fit_data" in task_names
        assert "extract_qubit_parameters" in task_names


@pytest.fixture
def results_two_qubit():
    """Results from AmplitudeRabi experiment."""
    data = {}
    data[handles.result_handle("q0")] = AcquiredResult(
        data=np.array(
            [
                0.05290302 - 0.13215136j,
                0.06067577 - 0.12907117j,
                0.05849071 - 0.09401458j,
                0.0683788 - 0.04265771j,
                0.07369121 + 0.0238058j,
                0.08271086 + 0.10077513j,
                0.09092848 + 0.1884216j,
                0.1063583 + 0.28337206j,
                0.11472132 + 0.38879551j,
                0.13147716 + 0.49203866j,
                0.13378882 + 0.59027211j,
                0.15108762 + 0.70302525j,
                0.16102455 + 0.77474721j,
                0.16483135 + 0.83853894j,
                0.17209631 + 0.88743935j,
                0.17435144 + 0.90659384j,
                0.17877636 + 0.92026812j,
                0.17153804 + 0.90921755j,
                0.17243493 + 0.88099388j,
                0.164842 + 0.82561295j,
                0.15646681 + 0.76574749j,
            ]
        ),
        axis_name=[],
        axis=[],
    )
    data[handles.result_handle("q1")] = AcquiredResult(
        np.array(
            [
                -0.00223629 + 2.51237327j,
                -0.00306557 + 2.48804924j,
                0.01106652 + 2.42217034j,
                0.02728497 + 2.31121781j,
                0.04834255 + 2.17814482j,
                0.08530416 + 1.9752334j,
                0.11996124 + 1.76145403j,
                0.15104249 + 1.57310709j,
                0.19803634 + 1.32573559j,
                0.22609813 + 1.11031412j,
                0.25675565 + 0.92663821j,
                0.29063937 + 0.7483506j,
                0.31398193 + 0.57160351j,
                0.32026664 + 0.51355002j,
                0.33606578 + 0.42037339j,
                0.34222361 + 0.40457712j,
                0.33488714 + 0.44717097j,
                0.32479511 + 0.5031597j,
                0.30801348 + 0.62862901j,
                0.28395935 + 0.78102526j,
                0.25708916 + 0.93078653j,
            ]
        ),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q0", "g")] = AcquiredResult(
        data=(0.05745863888207082 - 0.13026141779382786j),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q0", "e")] = AcquiredResult(
        data=(0.1770431406621688 + 0.91612948998106j),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q1", "g")] = AcquiredResult(
        data=(0.0033944563323902097 + 2.509301287477822j),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q1", "e")] = AcquiredResult(
        data=(0.3364214235541073 + 0.4244829181581308j),
        axis_name=[],
        axis=[],
    )

    sweep_points = [
        np.array(
            [
                0.0,
                0.0238155,
                0.04763101,
                0.07144651,
                0.09526201,
                0.11907752,
                0.14289302,
                0.16670852,
                0.19052403,
                0.21433953,
                0.23815503,
                0.26197054,
                0.28578604,
                0.30960154,
                0.33341705,
                0.35723255,
                0.38104805,
                0.40486356,
                0.42867906,
                0.45249456,
                0.47631007,
            ]
        ),
        np.array(
            [
                0.0,
                0.02345217,
                0.04690434,
                0.07035651,
                0.09380869,
                0.11726086,
                0.14071303,
                0.1641652,
                0.18761737,
                0.21106954,
                0.23452172,
                0.25797389,
                0.28142606,
                0.30487823,
                0.3283304,
                0.35178257,
                0.37523475,
                0.39868692,
                0.42213909,
                0.44559126,
                0.46904343,
            ]
        ),
    ]
    return RunExperimentResults(data=data), sweep_points


class TestRabiAnalysisTwoQubit:
    def test_create_and_run_no_pca(
        self, two_tunable_transmon_platform, results_two_qubit
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = amplitude_rabi.analysis_workflow.options()
        options.do_pca(value=False)

        result = amplitude_rabi.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            amplitudes=results_two_qubit[1],
            options=options,
        ).run()

        assert len(result.tasks) == 5

        for task in result.tasks:
            if task.name in ["calculate_qubit_population", "fit_data"]:
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
                    -0.002274,
                    0.00146965,
                    0.03430455,
                    0.08381782,
                    0.14708857,
                    0.22066961,
                    0.30423641,
                    0.39547093,
                    0.49582323,
                    0.59502358,
                    0.68794084,
                    0.79617092,
                    0.86490077,
                    0.9254888,
                    0.97240199,
                    0.99071441,
                    1.00409101,
                    0.99288616,
                    0.96635819,
                    0.91329628,
                    0.85591957,
                ]
            ),
        )
        assert_array_almost_equal(
            proc_data_dict["q1"]["population"],
            np.array(
                [
                    -0.00185753,
                    0.00945743,
                    0.04132639,
                    0.09443323,
                    0.15824786,
                    0.25591582,
                    0.3584948,
                    0.44891126,
                    0.56812378,
                    0.67097804,
                    0.7591781,
                    0.84509896,
                    0.92951172,
                    0.95713425,
                    1.00189555,
                    1.00974391,
                    0.98927363,
                    0.96233238,
                    0.90239364,
                    0.82931717,
                    0.75726275,
                ]
            ),
        )

        fit_values = result.tasks["fit_data"].output["q0"].best_values
        assert_almost_equal(fit_values["frequency"], 8.3203418993638)
        assert_almost_equal(fit_values["phase"], 3.1237529536013273)
        assert_almost_equal(fit_values["amplitude"], 0.5041511237243046)
        assert_almost_equal(fit_values["offset"], 0.502026090780613)
        fit_values = result.tasks["fit_data"].output["q1"].best_values
        assert_almost_equal(
            fit_values["frequency"],
            8.944201329352525,
        )
        assert_almost_equal(fit_values["phase"], 3.1536281498777274)
        assert_almost_equal(fit_values["amplitude"], 0.5065553919929595)
        assert_almost_equal(fit_values["offset"], 0.5020640553309232)

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["ge_drive_amplitude_pi"],
            0.8,
        )
        assert_almost_equal(
            qubit_parameters["new_parameter_values"]["q0"][
                "ge_drive_amplitude_pi"
            ].nominal_value,
            0.3797238613259197,
        )
        assert_almost_equal(
            qubit_parameters["new_parameter_values"]["q0"][
                "ge_drive_amplitude_pi"
            ].std_dev,
            0.002875704340756095,
        )
        assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q1"]["ge_drive_amplitude_pi"],
            0.81,
        )
        assert_almost_equal(
            qubit_parameters["new_parameter_values"]["q1"][
                "ge_drive_amplitude_pi"
            ].nominal_value,
            0.34989788825878415,
        )
        assert_almost_equal(
            qubit_parameters["new_parameter_values"]["q1"][
                "ge_drive_amplitude_pi"
            ].std_dev,
            0.0030524911744699897,
        )

    def test_create_and_run_pca(self, two_tunable_transmon_platform, results_two_qubit):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = amplitude_rabi.analysis_workflow.options()
        options.do_pca(value=True)

        result = amplitude_rabi.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            amplitudes=results_two_qubit[1],
            options=options,
        ).run()

        assert len(result.tasks) == 5

        for task in result.tasks:
            if task.name in ["calculate_qubit_population", "fit_data"]:
                assert "q0" in task.output
        assert "new_parameter_values" in result.tasks["extract_qubit_parameters"].output

        proc_data_dict = result.tasks["calculate_qubit_population"].output
        assert proc_data_dict["q0"]["num_cal_traces"] == 2
        assert proc_data_dict["q1"]["num_cal_traces"] == 2
        assert_array_almost_equal(
            proc_data_dict["q0"]["population"],
            np.array(
                [
                    0.60893583,
                    0.60499912,
                    0.57041226,
                    0.5182681,
                    0.45162937,
                    0.37413388,
                    0.28611972,
                    0.19003514,
                    0.0843409,
                    -0.02013298,
                    -0.11800093,
                    -0.23198539,
                    -0.30437035,
                    -0.36818462,
                    -0.41759229,
                    -0.4368789,
                    -0.45096481,
                    -0.43916876,
                    -0.41122607,
                    -0.35534221,
                    -0.29491423,
                ]
            ),
        )
        assert_array_almost_equal(
            proc_data_dict["q1"]["population"],
            np.array(
                [
                    -1.2474706,
                    -1.22359998,
                    -1.15630344,
                    -1.0441882,
                    -0.90946133,
                    -0.70324416,
                    -0.48667402,
                    -0.29577994,
                    -0.04406603,
                    0.17305928,
                    0.35927523,
                    0.54069479,
                    0.71889131,
                    0.77719758,
                    0.87170263,
                    0.88828616,
                    0.84506659,
                    0.78818295,
                    0.66165061,
                    0.50737019,
                    0.35523526,
                ]
            ),
        )

        fit_values = result.tasks["fit_data"].output["q0"].best_values
        assert_almost_equal(fit_values["frequency"], 8.32034466074661)
        assert_almost_equal(fit_values["phase"], 6.265343866673316)
        assert_almost_equal(fit_values["amplitude"], 0.530972990501319)
        assert_almost_equal(fit_values["offset"], 0.07780831562589109)
        fit_values = result.tasks["fit_data"].output["q1"].best_values
        assert_almost_equal(fit_values["frequency"], 8.944142729731128)
        assert_almost_equal(fit_values["phase"], 3.1536457220232963)
        assert_almost_equal(fit_values["amplitude"], 1.069478377275622)
        assert_almost_equal(fit_values["offset"], -0.1835670924745826)

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"]["ge_drive_amplitude_pi"],
            0.8,
        )
        assert_almost_equal(
            qubit_parameters["new_parameter_values"]["q0"][
                "ge_drive_amplitude_pi"
            ].nominal_value,
            0.3797239444901262,
        )
        assert_almost_equal(
            qubit_parameters["new_parameter_values"]["q0"][
                "ge_drive_amplitude_pi"
            ].std_dev,
            0.002875751217783006,
        )
        assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q1"]["ge_drive_amplitude_pi"],
            0.81,
        )
        assert_almost_equal(
            qubit_parameters["new_parameter_values"]["q1"][
                "ge_drive_amplitude_pi"
            ].nominal_value,
            0.34989821604181487,
        )
        assert_almost_equal(
            qubit_parameters["new_parameter_values"]["q1"][
                "ge_drive_amplitude_pi"
            ].std_dev,
            0.003053048005281863,
        )

    def test_create_and_run_no_fitting(
        self, two_tunable_transmon_platform, results_two_qubit
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = amplitude_rabi.analysis_workflow.options()
        options.do_fitting(value=False)

        result = amplitude_rabi.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            amplitudes=results_two_qubit[1],
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
        options = amplitude_rabi.analysis_workflow.options()
        options.do_plotting(value=False)

        result = amplitude_rabi.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            amplitudes=results_two_qubit[1],
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
        options = amplitude_rabi.analysis_workflow.options()
        options.close_figures(value=True)

        result = amplitude_rabi.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            amplitudes=results_two_qubit[1],
            options=options,
        ).run()

        assert isinstance(
            result.tasks["plot_population"].output["q0"], mpl.figure.Figure
        )
        assert isinstance(
            result.tasks["plot_population"].output["q1"], mpl.figure.Figure
        )

        options.close_figures(value=False)

        result = amplitude_rabi.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            amplitudes=results_two_qubit[1],
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
