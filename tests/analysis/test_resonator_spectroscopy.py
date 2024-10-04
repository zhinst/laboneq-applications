"""Tests for the resonator spectroscopy analysis using the testing utilities."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from laboneq_applications.analysis import resonator_spectroscopy
from laboneq_applications.tasks.run_experiment import (
    AcquiredResult,
    RunExperimentResults,
)


@pytest.fixture()
def results_single_qubit():
    """Results from a resonator spectroscopy experiment.

    In the AcquiredResults below, the axis corresponds to the qubit readout-resonator
    frequency, and the data is the magnitude of the acquired complex signal, which has
    a double-dipped Fano line-shape.
    """
    data = {
        "result": {
            "q0": AcquiredResult(
                data=np.array(
                    [
                        -1.03017225e-04 + 2.16266844e-05j,
                        -7.56883254e-05 + 6.78858755e-05j,
                        -3.03757712e-05 + 9.30859560e-05j,
                        2.28454259e-05 + 9.44941968e-05j,
                        6.71934205e-05 + 6.66653353e-05j,
                        9.00867419e-05 + 2.27344337e-05j,
                        8.88190024e-05 - 2.62385697e-05j,
                        5.92831434e-05 - 6.29519669e-05j,
                        2.00131915e-05 - 8.33086777e-05j,
                        -2.49858492e-05 - 7.58859488e-05j,
                        -5.86424917e-05 - 5.19642711e-05j,
                        -7.35343659e-05 - 1.17339067e-05j,
                        -6.49782982e-05 + 2.58213066e-05j,
                        -3.88790259e-05 + 5.08291964e-05j,
                        -6.40578233e-06 + 5.89518113e-05j,
                        2.24683042e-05 + 4.92454041e-05j,
                        4.30172064e-05 + 2.68939068e-05j,
                        4.45564613e-05 + 2.58179972e-06j,
                        3.56395257e-05 - 1.95259401e-05j,
                        1.79794437e-05 - 2.81481457e-05j,
                        -1.17841499e-06 - 2.92846223e-05j,
                        -9.43096439e-06 - 1.61753868e-05j,
                        -9.87435896e-06 - 2.75339216e-06j,
                        -3.80908907e-06 + 4.16346386e-07j,
                        5.51819421e-06 - 5.15361408e-06j,
                        3.59088842e-06 - 1.59574896e-05j,
                        -5.74680274e-06 - 2.63108314e-05j,
                        -2.62034111e-05 - 2.62031490e-05j,
                        -4.85629828e-05 - 1.31494261e-05j,
                        -5.93415578e-05 + 1.92031377e-05j,
                        -4.49270383e-05 + 5.88877650e-05j,
                        -9.21814466e-06 + 8.82821213e-05j,
                        4.68664294e-05 + 9.03668156e-05j,
                        9.89805732e-05 + 5.97281591e-05j,
                        1.34199080e-04 - 6.58228878e-06j,
                        1.19589144e-04 - 8.64405801e-05j,
                        6.31635459e-05 - 1.54523227e-04j,
                        -3.48933167e-05 - 1.82832859e-04j,
                        -1.39566660e-04 - 1.50769082e-04j,
                        -2.17300674e-04 - 5.42987378e-05j,
                        -2.29580968e-04 + 7.97726298e-05j,
                        -1.61737748e-04 + 2.12163598e-04j,
                        -2.25261070e-05 + 2.91213644e-04j,
                        1.52380641e-04 + 2.77667295e-04j,
                        3.07119824e-04 + 1.60375738e-04j,
                        3.73799982e-04 - 4.38283750e-05j,
                        3.09671705e-04 - 2.71866707e-04j,
                        1.02975691e-04 - 4.31964242e-04j,
                        -1.79663383e-04 - 4.49660476e-04j,
                        -4.36248749e-04 - 2.70974479e-04j,
                        -5.47251475e-04 + 4.96272930e-05j,
                        -4.20720949e-04 + 3.91670247e-04j,
                        -9.47983282e-05 + 5.87651571e-04j,
                        2.92645511e-04 + 5.25435103e-04j,
                        5.51323387e-04 + 2.32331506e-04j,
                        5.66761957e-04 - 1.52967197e-04j,
                        3.44882256e-04 - 4.49104615e-04j,
                        6.00916287e-06 - 5.39749878e-04j,
                        -2.92836743e-04 - 4.10093178e-04j,
                        -4.46399297e-04 - 1.52060466e-04j,
                        -4.15910429e-04 + 1.22189811e-04j,
                        -2.49761576e-04 + 3.10596777e-04j,
                        -3.05384336e-05 + 3.61366925e-04j,
                        1.60111321e-04 + 2.84708536e-04j,
                        2.58670293e-04 + 1.34631812e-04j,
                        2.58971749e-04 - 2.95973367e-05j,
                        1.75825946e-04 - 1.43660198e-04j,
                        5.89566154e-05 - 1.84948615e-04j,
                        -4.28663880e-05 - 1.54956287e-04j,
                        -9.49673206e-05 - 8.36147064e-05j,
                        -8.95451513e-05 - 1.40095600e-05j,
                        -4.84806772e-05 + 2.16560782e-05j,
                        -6.34506760e-06 + 7.26550116e-06j,
                        -7.14452202e-06 - 3.61524482e-05j,
                        -5.78226700e-05 - 7.04475709e-05j,
                        -1.44167377e-04 - 4.15728484e-05j,
                        -2.03374879e-04 + 7.36497983e-05j,
                        -1.60179732e-04 + 2.42057944e-04j,
                        2.98596848e-05 + 3.65975306e-04j,
                        3.00424040e-04 + 3.26702797e-04j,
                        5.10392313e-04 + 7.56232954e-05j,
                        4.94210204e-04 - 2.96439786e-04j,
                        2.15433328e-04 - 5.84809256e-04j,
                        -2.08614448e-04 - 6.19139815e-04j,
                        -5.56709074e-04 - 3.70355873e-04j,
                        -6.71516816e-04 + 4.34377567e-05j,
                        -5.06529439e-04 + 4.32437563e-04j,
                        -1.54411720e-04 + 6.34901570e-04j,
                        2.34068449e-04 + 5.93507762e-04j,
                        5.11515314e-04 + 3.46393381e-04j,
                        5.98184674e-04 + 1.94460474e-07j,
                        4.84555629e-04 - 3.16269207e-04j,
                        2.25257816e-04 - 5.11713870e-04j,
                        -8.17789993e-05 - 5.37264905e-04j,
                        -3.44584921e-04 - 4.01282894e-04j,
                        -4.89944116e-04 - 1.55657803e-04j,
                        -4.86297358e-04 + 1.24181987e-04j,
                        -3.45667135e-04 + 3.51187085e-04j,
                        -1.09106591e-04 + 4.70964195e-04j,
                        1.49077786e-04 + 4.52455123e-04j,
                        3.53786568e-04 + 3.04725050e-04j,
                    ]
                ),
                axis_name=["RR Frequency"],
                axis=[
                    np.array(
                        [
                            6.690e09,
                            6.691e09,
                            6.692e09,
                            6.693e09,
                            6.694e09,
                            6.695e09,
                            6.696e09,
                            6.697e09,
                            6.698e09,
                            6.699e09,
                            6.700e09,
                            6.701e09,
                            6.702e09,
                            6.703e09,
                            6.704e09,
                            6.705e09,
                            6.706e09,
                            6.707e09,
                            6.708e09,
                            6.709e09,
                            6.710e09,
                            6.711e09,
                            6.712e09,
                            6.713e09,
                            6.714e09,
                            6.715e09,
                            6.716e09,
                            6.717e09,
                            6.718e09,
                            6.719e09,
                            6.720e09,
                            6.721e09,
                            6.722e09,
                            6.723e09,
                            6.724e09,
                            6.725e09,
                            6.726e09,
                            6.727e09,
                            6.728e09,
                            6.729e09,
                            6.730e09,
                            6.731e09,
                            6.732e09,
                            6.733e09,
                            6.734e09,
                            6.735e09,
                            6.736e09,
                            6.737e09,
                            6.738e09,
                            6.739e09,
                            6.740e09,
                            6.741e09,
                            6.742e09,
                            6.743e09,
                            6.744e09,
                            6.745e09,
                            6.746e09,
                            6.747e09,
                            6.748e09,
                            6.749e09,
                            6.750e09,
                            6.751e09,
                            6.752e09,
                            6.753e09,
                            6.754e09,
                            6.755e09,
                            6.756e09,
                            6.757e09,
                            6.758e09,
                            6.759e09,
                            6.760e09,
                            6.761e09,
                            6.762e09,
                            6.763e09,
                            6.764e09,
                            6.765e09,
                            6.766e09,
                            6.767e09,
                            6.768e09,
                            6.769e09,
                            6.770e09,
                            6.771e09,
                            6.772e09,
                            6.773e09,
                            6.774e09,
                            6.775e09,
                            6.776e09,
                            6.777e09,
                            6.778e09,
                            6.779e09,
                            6.780e09,
                            6.781e09,
                            6.782e09,
                            6.783e09,
                            6.784e09,
                            6.785e09,
                            6.786e09,
                            6.787e09,
                            6.788e09,
                            6.789e09,
                            6.790e09,
                        ]
                    )
                ],
            )
        },
    }
    sweep_points = data["result"]["q0"].axis[0]
    return RunExperimentResults(data=data), sweep_points


class TestResonatorSpectroscopyAnalysisSingleQubit:
    def test_create_and_run_find_dips(
        self, single_tunable_transmon_platform, results_single_qubit
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = resonator_spectroscopy.analysis_workflow.options()
        options.find_peaks(False)

        result = resonator_spectroscopy.analysis_workflow(
            result=results_single_qubit[0],
            qubit=q0,
            frequencies=results_single_qubit[1],
            options=options,
        ).run()

        assert len(result.tasks) == 6

        proc_data_dict = result.tasks["calculate_signal_magnitude_and_phase"].output
        np.testing.assert_array_almost_equal(
            proc_data_dict["magnitude"],
            np.array(
                [
                    1.05262824e-04,
                    1.01672094e-04,
                    9.79167130e-05,
                    9.72165969e-05,
                    9.46531705e-05,
                    9.29111164e-05,
                    9.26135937e-05,
                    8.64721992e-05,
                    8.56788399e-05,
                    7.98934909e-05,
                    7.83532214e-05,
                    7.44646730e-05,
                    6.99208060e-05,
                    6.39936392e-05,
                    5.92988204e-05,
                    5.41288696e-05,
                    5.07322607e-05,
                    4.46311991e-05,
                    4.06378903e-05,
                    3.34002770e-05,
                    2.93083225e-05,
                    1.87239480e-05,
                    1.02510552e-05,
                    3.83177555e-06,
                    7.55051027e-06,
                    1.63565263e-05,
                    2.69311268e-05,
                    3.70570340e-05,
                    5.03117352e-05,
                    6.23713153e-05,
                    7.40689384e-05,
                    8.87620817e-05,
                    1.01796972e-04,
                    1.15605393e-04,
                    1.34360409e-04,
                    1.47558589e-04,
                    1.66934302e-04,
                    1.86132743e-04,
                    2.05451135e-04,
                    2.23981999e-04,
                    2.43045455e-04,
                    2.66781730e-04,
                    2.92083570e-04,
                    3.16731727e-04,
                    3.46472169e-04,
                    3.76360668e-04,
                    4.12077749e-04,
                    4.44068800e-04,
                    4.84224612e-04,
                    5.13556364e-04,
                    5.49497083e-04,
                    5.74814491e-04,
                    5.95248765e-04,
                    6.01434487e-04,
                    5.98277031e-04,
                    5.87041804e-04,
                    5.66249703e-04,
                    5.39783328e-04,
                    5.03914450e-04,
                    4.71587444e-04,
                    4.33487987e-04,
                    3.98561417e-04,
                    3.62655002e-04,
                    3.26641371e-04,
                    2.91609405e-04,
                    2.60657570e-04,
                    2.27052892e-04,
                    1.94118192e-04,
                    1.60776174e-04,
                    1.26531463e-04,
                    9.06344410e-05,
                    5.30976627e-05,
                    9.64610750e-06,
                    3.68516446e-05,
                    9.11390224e-05,
                    1.50041776e-04,
                    2.16299871e-04,
                    2.90257807e-04,
                    3.67191401e-04,
                    4.43834791e-04,
                    5.15964336e-04,
                    5.76298770e-04,
                    6.23228197e-04,
                    6.53340721e-04,
                    6.68646742e-04,
                    6.72920258e-04,
                    6.66013752e-04,
                    6.53408741e-04,
                    6.37996475e-04,
                    6.17767182e-04,
                    5.98184706e-04,
                    5.78636647e-04,
                    5.59099427e-04,
                    5.43453202e-04,
                    5.28929796e-04,
                    5.14076442e-04,
                    5.01902666e-04,
                    4.92765804e-04,
                    4.83437195e-04,
                    4.76382015e-04,
                    4.66928573e-04,
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            proc_data_dict["phase"],
            np.array(
                [
                    2.93466492e00,
                    2.41048546e00,
                    1.88622129e00,
                    1.33358278e00,
                    7.81453091e-01,
                    2.47200101e-01,
                    -2.87246161e-01,
                    -8.15403637e-01,
                    -1.33503448e00,
                    -1.88887215e00,
                    -2.41649933e00,
                    -2.98335632e00,
                    2.76334387e00,
                    2.22376262e00,
                    1.67903300e00,
                    1.14275558e00,
                    5.58735591e-01,
                    5.78797314e-02,
                    -5.01208865e-01,
                    -1.00237507e00,
                    -1.61101469e00,
                    -2.09865497e00,
                    -2.86965751e00,
                    3.03272145e00,
                    -7.51248525e-01,
                    -1.34945485e00,
                    -1.78583875e00,
                    -2.35619949e00,
                    -2.87716275e00,
                    2.82862433e00,
                    2.22251921e00,
                    1.67483620e00,
                    1.09236051e00,
                    5.42940053e-01,
                    -4.90094112e-02,
                    -6.25873147e-01,
                    -1.18275761e00,
                    -1.75937676e00,
                    -2.31762925e00,
                    -2.89672847e00,
                    2.80717300e00,
                    2.22213928e00,
                    1.64799511e00,
                    1.06888366e00,
                    4.81243755e-01,
                    -1.16717952e-01,
                    -7.20480910e-01,
                    -1.33677479e00,
                    -1.95091767e00,
                    -2.58576904e00,
                    3.05115540e00,
                    2.39193876e00,
                    1.73073568e00,
                    1.06262645e00,
                    3.98823369e-01,
                    -2.63615575e-01,
                    -9.15916537e-01,
                    -1.55966355e00,
                    -2.19090539e00,
                    -2.81328256e00,
                    2.85584394e00,
                    2.24805321e00,
                    1.65510411e00,
                    1.05850621e00,
                    4.79894308e-01,
                    -1.13794162e-01,
                    -6.85056652e-01,
                    -1.26220685e00,
                    -1.84068228e00,
                    -2.41967993e00,
                    -2.98639827e00,
                    2.72149053e00,
                    2.28867092e00,
                    -1.76590438e00,
                    -2.25808624e00,
                    -2.86084391e00,
                    2.79414554e00,
                    2.15538129e00,
                    1.48938731e00,
                    8.27277079e-01,
                    1.47096798e-01,
                    -5.40291037e-01,
                    -1.21784014e00,
                    -1.89579151e00,
                    -2.55456503e00,
                    3.07699662e00,
                    2.43493919e00,
                    1.80937026e00,
                    1.19514290e00,
                    5.95253084e-01,
                    3.25084335e-04,
                    -5.78270582e-01,
                    -1.15611966e00,
                    -1.72185041e00,
                    -2.28032429e00,
                    -2.83397272e00,
                    2.89157357e00,
                    2.34827342e00,
                    1.79844684e00,
                    1.25251205e00,
                    7.11031910e-01,
                ]
            ),
        )

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"][
                "readout_resonator_frequency"
            ],
            7.1e9,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "readout_resonator_frequency"
            ].nominal_value,
            6713000000.0,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "readout_resonator_frequency"
            ].std_dev,
            0,
        )

    def test_create_and_run_find_peaks(
        self, single_tunable_transmon_platform, results_single_qubit
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = resonator_spectroscopy.analysis_workflow.options()
        options.find_peaks(True)

        result = resonator_spectroscopy.analysis_workflow(
            result=results_single_qubit[0],
            qubit=q0,
            frequencies=results_single_qubit[1],
            options=options,
        ).run()

        qubit_parameters = result.tasks["extract_qubit_parameters"].output
        np.testing.assert_almost_equal(
            qubit_parameters["old_parameter_values"]["q0"][
                "readout_resonator_frequency"
            ],
            7.1e9,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "readout_resonator_frequency"
            ].nominal_value,
            6.775e9,
        )
        np.testing.assert_allclose(
            qubit_parameters["new_parameter_values"]["q0"][
                "readout_resonator_frequency"
            ].std_dev,
            0,
        )

    def test_create_and_run_no_plotting(
        self, single_tunable_transmon_platform, results_single_qubit
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = resonator_spectroscopy.analysis_workflow.options()
        options.do_plotting(False)

        result = resonator_spectroscopy.analysis_workflow(
            result=results_single_qubit[0],
            qubit=q0,
            frequencies=results_single_qubit[1],
            options=options,
        ).run()

        assert len(result.tasks) == 3

        task_names = [t.name for t in result.tasks]
        assert "plot_raw_complex_data_1d" not in task_names
        assert "plot_magnitude_phase" not in task_names
        assert "plot_real_imaginary" not in task_names
        assert "fit_data" in task_names
        assert "extract_qubit_parameters" in task_names

    def test_create_and_run_close_figures(
        self, single_tunable_transmon_platform, results_single_qubit
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = resonator_spectroscopy.analysis_workflow.options()
        options.close_figures(True)

        result = resonator_spectroscopy.analysis_workflow(
            result=results_single_qubit[0],
            qubit=q0,
            frequencies=results_single_qubit[1],
            options=options,
        ).run()

        assert isinstance(
            result.tasks["plot_magnitude_phase"].output, mpl.figure.Figure
        )
        assert isinstance(result.tasks["plot_real_imaginary"].output, mpl.figure.Figure)

        options.close_figures(False)
        result = resonator_spectroscopy.analysis_workflow(
            result=results_single_qubit[0],
            qubit=q0,
            frequencies=results_single_qubit[1],
            options=options,
        ).run()

        assert isinstance(
            result.tasks["plot_magnitude_phase"].output, mpl.figure.Figure
        )
        assert isinstance(result.tasks["plot_real_imaginary"].output, mpl.figure.Figure)
        plt.close(result.tasks["plot_magnitude_phase"].output)
        plt.close(result.tasks["plot_real_imaginary"].output)
