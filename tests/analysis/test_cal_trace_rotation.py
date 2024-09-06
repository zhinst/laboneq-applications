import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from laboneq_applications.analysis.cal_trace_rotation import (
    calculate_population_1d,
    principal_component_analysis,
    rotate_data_to_cal_trace_results,
)


@pytest.mark.parametrize(
    ("pca_input", "expected_output"),
    [
        (np.array([1 + 5j, 2 + 2j, 5 + 2j]), np.array([-2.54012, 0.080577, 2.459543])),
        (np.array([1, 2, 5]), np.array([-1.666667, -0.666667, 2.333333])),
    ],
)
def test_principal_component_analysis(pca_input, expected_output):
    # TODO: Expand tests
    result = principal_component_analysis(pca_input)
    assert_array_almost_equal(result, expected_output)


def test_rotate_data_to_cal_trace_results():
    # TODO: Expand tests
    raw_data = np.array([1 + 2j, 3 + 2j])
    pts_1 = np.array([0.5 + 2j])
    pts_2 = np.array([0.1 + 2j])
    assert_array_almost_equal(
        rotate_data_to_cal_trace_results(raw_data, pts_1, pts_2),
        [-1.25, -6.25],
    )


@pytest.fixture()
def results():
    """Results from AmplitudeRabi experiment."""
    raw = np.array(
        [
            -0.14181528 + 0.74110792j,
            -0.13743392 + 0.73537577j,
            -0.12167174 + 0.71405756j,
            -0.09305505 + 0.67498465j,
            -0.06360052 + 0.63947572j,
            -0.01566121 + 0.58065239j,
            0.02906186 + 0.52455673j,
            0.08227274 + 0.45812602j,
            0.11847114 + 0.40637085j,
            0.16929275 + 0.33847125j,
            0.22506453 + 0.27189472j,
            0.27475463 + 0.21382596j,
            0.30095408 + 0.17743883j,
            0.33585012 + 0.12845817j,
            0.35314972 + 0.11077295j,
            0.35619345 + 0.10518323j,
            0.35681263 + 0.10114049j,
            0.3439974 + 0.11990758j,
            0.32719295 + 0.14280674j,
            0.28974077 + 0.18369942j,
            0.25877208 + 0.22439374j,
        ],
    )
    sweep_points = np.array(
        [
            0.0,
            0.02133995,
            0.04267991,
            0.06401986,
            0.08535982,
            0.10669977,
            0.12803973,
            0.14937968,
            0.17071964,
            0.19205959,
            0.21339955,
            0.2347395,
            0.25607945,
            0.27741941,
            0.29875936,
            0.32009932,
            0.34143927,
            0.36277923,
            0.38411918,
            0.40545914,
            0.42679909,
        ],
    )
    calibration_traces = [
        -0.14148732183502724 + 0.7385117964730766j,
        0.3582122789731416 + 0.1005765204862131j,
    ]
    return raw, sweep_points, calibration_traces


class TestCalculatePopulation1D:
    @pytest.fixture()
    def analysis(self, results):
        return calculate_population_1d(
            *results,
            do_pca=False,
        )

    def test_sweep_points(self, analysis):
        assert_array_almost_equal(
            analysis["sweep_points"],
            np.array(
                [
                    0.0,
                    0.02133995,
                    0.04267991,
                    0.06401986,
                    0.08535982,
                    0.10669977,
                    0.12803973,
                    0.14937968,
                    0.17071964,
                    0.19205959,
                    0.21339955,
                    0.2347395,
                    0.25607945,
                    0.27741941,
                    0.29875936,
                    0.32009932,
                    0.34143927,
                    0.36277923,
                    0.38411918,
                    0.40545914,
                    0.42679909,
                ],
            ),
        )

    def test_sweep_points_w_cal_traces(self, analysis):
        assert_array_almost_equal(
            analysis["sweep_points_with_cal_traces"],
            np.array(
                [
                    0.0,
                    0.02133995,
                    0.04267991,
                    0.06401986,
                    0.08535982,
                    0.10669977,
                    0.12803973,
                    0.14937968,
                    0.17071964,
                    0.19205959,
                    0.21339955,
                    0.2347395,
                    0.25607945,
                    0.27741941,
                    0.29875936,
                    0.32009932,
                    0.34143927,
                    0.36277923,
                    0.38411918,
                    0.40545914,
                    0.42679909,
                    0.44813904,
                    0.46947899,
                ],
            ),
        )

    def test_sweep_points_cal_traces(self, analysis):
        assert_array_almost_equal(
            analysis["sweep_points_cal_traces"],
            np.array([0.44813904, 0.46947899]),
        )

    def test_data_raw(self, analysis):
        assert_array_almost_equal(
            analysis["data_raw"],
            np.array(
                [
                    -0.14181528 + 0.74110792j,
                    -0.13743392 + 0.73537577j,
                    -0.12167174 + 0.71405756j,
                    -0.09305505 + 0.67498465j,
                    -0.06360052 + 0.63947572j,
                    -0.01566121 + 0.58065239j,
                    0.02906186 + 0.52455673j,
                    0.08227274 + 0.45812602j,
                    0.11847114 + 0.40637085j,
                    0.16929275 + 0.33847125j,
                    0.22506453 + 0.27189472j,
                    0.27475463 + 0.21382596j,
                    0.30095408 + 0.17743883j,
                    0.33585012 + 0.12845817j,
                    0.35314972 + 0.11077295j,
                    0.35619345 + 0.10518323j,
                    0.35681263 + 0.10114049j,
                    0.3439974 + 0.11990758j,
                    0.32719295 + 0.14280674j,
                    0.28974077 + 0.18369942j,
                    0.25877208 + 0.22439374j,
                ],
            ),
        )

    def test_data_raw_w_cal_traces(self, analysis):
        assert_array_almost_equal(
            analysis["data_raw_with_cal_traces"],
            np.array(
                [
                    -0.14181528 + 0.74110792j,
                    -0.13743392 + 0.73537577j,
                    -0.12167174 + 0.71405756j,
                    -0.09305505 + 0.67498465j,
                    -0.06360052 + 0.63947572j,
                    -0.01566121 + 0.58065239j,
                    0.02906186 + 0.52455673j,
                    0.08227274 + 0.45812602j,
                    0.11847114 + 0.40637085j,
                    0.16929275 + 0.33847125j,
                    0.22506453 + 0.27189472j,
                    0.27475463 + 0.21382596j,
                    0.30095408 + 0.17743883j,
                    0.33585012 + 0.12845817j,
                    0.35314972 + 0.11077295j,
                    0.35619345 + 0.10518323j,
                    0.35681263 + 0.10114049j,
                    0.3439974 + 0.11990758j,
                    0.32719295 + 0.14280674j,
                    0.28974077 + 0.18369942j,
                    0.25877208 + 0.22439374j,
                    -0.14148732 + 0.7385118j,
                    0.35821228 + 0.10057652j,
                ],
            ),
        )

    def test_data_raw_cal_traces(self, analysis):
        assert_array_almost_equal(
            analysis["data_raw_cal_traces"],
            np.array([-0.14148732 + 0.7385118j, 0.35821228 + 0.10057652j]),
        )

    def test_data_rotated(self, analysis):
        assert_array_almost_equal(
            analysis["population"],
            np.array(
                [
                    -0.00277166,
                    0.00613112,
                    0.03883595,
                    0.09857108,
                    0.15548143,
                    0.24910771,
                    0.33763663,
                    0.44266485,
                    0.52049004,
                    0.6251271,
                    0.73224575,
                    0.82647127,
                    0.88175776,
                    0.95589651,
                    0.98624188,
                    0.99398839,
                    0.99838702,
                    0.9704031,
                    0.93536926,
                    0.86714271,
                    0.80404261,
                ],
            ),
        )

    def test_data_rotated_w_cal_traces(self, analysis):
        assert_array_almost_equal(
            analysis["population_with_cal_traces"],
            np.array(
                [
                    -0.00277166,
                    0.00613112,
                    0.03883595,
                    0.09857108,
                    0.15548143,
                    0.24910771,
                    0.33763663,
                    0.44266485,
                    0.52049004,
                    0.6251271,
                    0.73224575,
                    0.82647127,
                    0.88175776,
                    0.95589651,
                    0.98624188,
                    0.99398839,
                    0.99838702,
                    0.9704031,
                    0.93536926,
                    0.86714271,
                    0.80404261,
                    0.0,
                    1.0,
                ],
            ),
        )

    def test_data_rotated_cal_traces(self, analysis):
        assert_array_almost_equal(
            analysis["population_cal_traces"],
            np.array([0.0, 1.0]),
        )

    def test_num_cal_traces(self, analysis):
        assert analysis["num_cal_traces"] == 2


class TestCalculatePopulation1DWithPCA:
    @pytest.fixture()
    def analysis(self, results):
        return calculate_population_1d(
            *results,
            do_pca=True,
        )

    def test_sweep_points(self, analysis):
        assert_array_almost_equal(
            analysis["sweep_points"],
            np.array(
                [
                    0.0,
                    0.02133995,
                    0.04267991,
                    0.06401986,
                    0.08535982,
                    0.10669977,
                    0.12803973,
                    0.14937968,
                    0.17071964,
                    0.19205959,
                    0.21339955,
                    0.2347395,
                    0.25607945,
                    0.27741941,
                    0.29875936,
                    0.32009932,
                    0.34143927,
                    0.36277923,
                    0.38411918,
                    0.40545914,
                    0.42679909,
                ],
            ),
        )

    def test_sweep_points_w_cal_traces(self, analysis):
        assert_array_almost_equal(
            analysis["sweep_points_with_cal_traces"],
            np.array(
                [
                    0.0,
                    0.02133995,
                    0.04267991,
                    0.06401986,
                    0.08535982,
                    0.10669977,
                    0.12803973,
                    0.14937968,
                    0.17071964,
                    0.19205959,
                    0.21339955,
                    0.2347395,
                    0.25607945,
                    0.27741941,
                    0.29875936,
                    0.32009932,
                    0.34143927,
                    0.36277923,
                    0.38411918,
                    0.40545914,
                    0.42679909,
                    0.44813904,
                    0.46947899,
                ],
            ),
        )

    def test_sweep_points_cal_traces(self, analysis):
        assert_array_almost_equal(
            analysis["sweep_points_cal_traces"],
            np.array([0.44813904, 0.46947899]),
        )

    def test_data_raw(self, analysis):
        assert_array_almost_equal(
            analysis["data_raw"],
            np.array(
                [
                    -0.14181528 + 0.74110792j,
                    -0.13743392 + 0.73537577j,
                    -0.12167174 + 0.71405756j,
                    -0.09305505 + 0.67498465j,
                    -0.06360052 + 0.63947572j,
                    -0.01566121 + 0.58065239j,
                    0.02906186 + 0.52455673j,
                    0.08227274 + 0.45812602j,
                    0.11847114 + 0.40637085j,
                    0.16929275 + 0.33847125j,
                    0.22506453 + 0.27189472j,
                    0.27475463 + 0.21382596j,
                    0.30095408 + 0.17743883j,
                    0.33585012 + 0.12845817j,
                    0.35314972 + 0.11077295j,
                    0.35619345 + 0.10518323j,
                    0.35681263 + 0.10114049j,
                    0.3439974 + 0.11990758j,
                    0.32719295 + 0.14280674j,
                    0.28974077 + 0.18369942j,
                    0.25877208 + 0.22439374j,
                ],
            ),
        )

    def test_data_raw_w_cal_traces(self, analysis):
        assert_array_almost_equal(
            analysis["data_raw_with_cal_traces"],
            np.array(
                [
                    -0.14181528 + 0.74110792j,
                    -0.13743392 + 0.73537577j,
                    -0.12167174 + 0.71405756j,
                    -0.09305505 + 0.67498465j,
                    -0.06360052 + 0.63947572j,
                    -0.01566121 + 0.58065239j,
                    0.02906186 + 0.52455673j,
                    0.08227274 + 0.45812602j,
                    0.11847114 + 0.40637085j,
                    0.16929275 + 0.33847125j,
                    0.22506453 + 0.27189472j,
                    0.27475463 + 0.21382596j,
                    0.30095408 + 0.17743883j,
                    0.33585012 + 0.12845817j,
                    0.35314972 + 0.11077295j,
                    0.35619345 + 0.10518323j,
                    0.35681263 + 0.10114049j,
                    0.3439974 + 0.11990758j,
                    0.32719295 + 0.14280674j,
                    0.28974077 + 0.18369942j,
                    0.25877208 + 0.22439374j,
                    -0.14148732 + 0.7385118j,
                    0.35821228 + 0.10057652j,
                ],
            ),
        )

    def test_data_raw_cal_traces(self, analysis):
        assert_array_almost_equal(
            analysis["data_raw_cal_traces"],
            np.array([-0.14148732 + 0.7385118j, 0.35821228 + 0.10057652j]),
        )

    def test_data_rotated(self, analysis):
        assert_array_almost_equal(
            analysis["population"],
            np.array(
                [
                    -0.47517906,
                    -0.46796471,
                    -0.4414623,
                    -0.39305581,
                    -0.34693901,
                    -0.27106964,
                    -0.19933069,
                    -0.11422167,
                    -0.05115556,
                    0.03363711,
                    0.12043968,
                    0.19679423,
                    0.24159584,
                    0.3016745,
                    0.32626416,
                    0.33254175,
                    0.33610661,
                    0.31342961,
                    0.28503987,
                    0.2297537,
                    0.1786206,
                ],
            ),
        )

    def test_data_rotated_w_cal_traces(self, analysis):
        assert_array_almost_equal(
            analysis["population_with_cal_traces"],
            np.array(
                [
                    -0.47517906,
                    -0.46796471,
                    -0.4414623,
                    -0.39305581,
                    -0.34693901,
                    -0.27106964,
                    -0.19933069,
                    -0.11422167,
                    -0.05115556,
                    0.03363711,
                    0.12043968,
                    0.19679423,
                    0.24159584,
                    0.3016745,
                    0.32626416,
                    0.33254175,
                    0.33610661,
                    0.31342961,
                    0.28503987,
                    0.2297537,
                    0.1786206,
                    -0.47293275,
                    0.33741351,
                ],
            ),
        )

    def test_data_rotated_cal_traces(self, analysis):
        assert_array_almost_equal(
            analysis["population_cal_traces"],
            np.array([-0.472933, 0.337414]),
        )

    def test_num_cal_traces(self, analysis):
        assert analysis["num_cal_traces"] == 2


@pytest.fixture()
def results_failure():
    """Results from AmplitudeRabi experiment with too many cal states."""
    raw = np.array(
        [
            -0.14181528 + 0.74110792j,
            -0.13743392 + 0.73537577j,
            -0.12167174 + 0.71405756j,
            -0.09305505 + 0.67498465j,
            -0.06360052 + 0.63947572j,
            -0.01566121 + 0.58065239j,
            0.02906186 + 0.52455673j,
            0.08227274 + 0.45812602j,
            0.11847114 + 0.40637085j,
            0.16929275 + 0.33847125j,
            0.22506453 + 0.27189472j,
            0.27475463 + 0.21382596j,
            0.30095408 + 0.17743883j,
            0.33585012 + 0.12845817j,
            0.35314972 + 0.11077295j,
            0.35619345 + 0.10518323j,
            0.35681263 + 0.10114049j,
            0.3439974 + 0.11990758j,
            0.32719295 + 0.14280674j,
            0.28974077 + 0.18369942j,
            0.25877208 + 0.22439374j,
        ],
    )
    sweep_points = np.array(
        [
            0.0,
            0.02133995,
            0.04267991,
            0.06401986,
            0.08535982,
            0.10669977,
            0.12803973,
            0.14937968,
            0.17071964,
            0.19205959,
            0.21339955,
            0.2347395,
            0.25607945,
            0.27741941,
            0.29875936,
            0.32009932,
            0.34143927,
            0.36277923,
            0.38411918,
            0.40545914,
            0.42679909,
        ],
    )
    calibration_traces = [
        -0.14148732183502724 + 0.7385117964730766j,
        0.3582122789731416 + 0.1005765204862131j,
        0.3582122789731416 + 0.1005765204862131j,
    ]
    return raw, sweep_points, calibration_traces


class TestCalculatePopulation1DFailure:
    def test_failure(self, results_failure):
        with pytest.raises(NotImplementedError):
            return calculate_population_1d(
                *results_failure,
                do_pca=False,
            )


@pytest.fixture()
def results_fallback_pca():
    """Results from AmplitudeRabi experiment with too many cal states."""
    raw = np.array(
        [
            -0.14181528 + 0.74110792j,
            -0.13743392 + 0.73537577j,
            -0.12167174 + 0.71405756j,
            -0.09305505 + 0.67498465j,
            -0.06360052 + 0.63947572j,
            -0.01566121 + 0.58065239j,
            0.02906186 + 0.52455673j,
            0.08227274 + 0.45812602j,
            0.11847114 + 0.40637085j,
            0.16929275 + 0.33847125j,
            0.22506453 + 0.27189472j,
            0.27475463 + 0.21382596j,
            0.30095408 + 0.17743883j,
            0.33585012 + 0.12845817j,
            0.35314972 + 0.11077295j,
            0.35619345 + 0.10518323j,
            0.35681263 + 0.10114049j,
            0.3439974 + 0.11990758j,
            0.32719295 + 0.14280674j,
            0.28974077 + 0.18369942j,
            0.25877208 + 0.22439374j,
        ],
    )
    sweep_points = np.array(
        [
            0.0,
            0.02133995,
            0.04267991,
            0.06401986,
            0.08535982,
            0.10669977,
            0.12803973,
            0.14937968,
            0.17071964,
            0.19205959,
            0.21339955,
            0.2347395,
            0.25607945,
            0.27741941,
            0.29875936,
            0.32009932,
            0.34143927,
            0.36277923,
            0.38411918,
            0.40545914,
            0.42679909,
        ],
    )
    calibration_traces = []
    return raw, sweep_points, calibration_traces


class TestCalculatePopulation1DFallbackPca:
    @pytest.fixture()
    def analysis(self, results_fallback_pca):
        return calculate_population_1d(
            *results_fallback_pca,
            do_pca=False,
        )

    def test_sweep_points(self, analysis):
        assert_array_almost_equal(
            analysis["sweep_points"],
            np.array(
                [
                    0.0,
                    0.02133995,
                    0.04267991,
                    0.06401986,
                    0.08535982,
                    0.10669977,
                    0.12803973,
                    0.14937968,
                    0.17071964,
                    0.19205959,
                    0.21339955,
                    0.2347395,
                    0.25607945,
                    0.27741941,
                    0.29875936,
                    0.32009932,
                    0.34143927,
                    0.36277923,
                    0.38411918,
                    0.40545914,
                    0.42679909,
                ],
            ),
        )

    def test_sweep_points_w_cal_traces(self, analysis):
        assert_array_almost_equal(
            analysis["sweep_points_with_cal_traces"],
            np.array(
                [
                    0.0,
                    0.02133995,
                    0.04267991,
                    0.06401986,
                    0.08535982,
                    0.10669977,
                    0.12803973,
                    0.14937968,
                    0.17071964,
                    0.19205959,
                    0.21339955,
                    0.2347395,
                    0.25607945,
                    0.27741941,
                    0.29875936,
                    0.32009932,
                    0.34143927,
                    0.36277923,
                    0.38411918,
                    0.40545914,
                    0.42679909,
                ],
            ),
        )

    def test_sweep_points_cal_traces(self, analysis):
        assert_array_almost_equal(
            analysis["sweep_points_cal_traces"],
            np.array([]),
        )

    def test_data_raw(self, analysis):
        assert_array_almost_equal(
            analysis["data_raw"],
            np.array(
                [
                    -0.14181528 + 0.74110792j,
                    -0.13743392 + 0.73537577j,
                    -0.12167174 + 0.71405756j,
                    -0.09305505 + 0.67498465j,
                    -0.06360052 + 0.63947572j,
                    -0.01566121 + 0.58065239j,
                    0.02906186 + 0.52455673j,
                    0.08227274 + 0.45812602j,
                    0.11847114 + 0.40637085j,
                    0.16929275 + 0.33847125j,
                    0.22506453 + 0.27189472j,
                    0.27475463 + 0.21382596j,
                    0.30095408 + 0.17743883j,
                    0.33585012 + 0.12845817j,
                    0.35314972 + 0.11077295j,
                    0.35619345 + 0.10518323j,
                    0.35681263 + 0.10114049j,
                    0.3439974 + 0.11990758j,
                    0.32719295 + 0.14280674j,
                    0.28974077 + 0.18369942j,
                    0.25877208 + 0.22439374j,
                ],
            ),
        )

    def test_data_raw_w_cal_traces(self, analysis):
        assert_array_almost_equal(
            analysis["data_raw_with_cal_traces"],
            np.array(
                [
                    -0.14181528 + 0.74110792j,
                    -0.13743392 + 0.73537577j,
                    -0.12167174 + 0.71405756j,
                    -0.09305505 + 0.67498465j,
                    -0.06360052 + 0.63947572j,
                    -0.01566121 + 0.58065239j,
                    0.02906186 + 0.52455673j,
                    0.08227274 + 0.45812602j,
                    0.11847114 + 0.40637085j,
                    0.16929275 + 0.33847125j,
                    0.22506453 + 0.27189472j,
                    0.27475463 + 0.21382596j,
                    0.30095408 + 0.17743883j,
                    0.33585012 + 0.12845817j,
                    0.35314972 + 0.11077295j,
                    0.35619345 + 0.10518323j,
                    0.35681263 + 0.10114049j,
                    0.3439974 + 0.11990758j,
                    0.32719295 + 0.14280674j,
                    0.28974077 + 0.18369942j,
                    0.25877208 + 0.22439374j,
                ],
            ),
        )

    def test_data_raw_cal_traces(self, analysis):
        assert_array_almost_equal(
            analysis["data_raw_cal_traces"],
            np.array([]),
        )

    def test_data_rotated(self, analysis):
        assert_array_almost_equal(
            analysis["population"],
            np.array(
                [
                    -0.4816324,
                    -0.47441804,
                    -0.44791557,
                    -0.39950894,
                    -0.35339225,
                    -0.27752302,
                    -0.20578413,
                    -0.1206752,
                    -0.05760877,
                    0.02718407,
                    0.11398637,
                    0.19034061,
                    0.23514239,
                    0.2952213,
                    0.31981071,
                    0.32608839,
                    0.32965344,
                    0.30697631,
                    0.27858648,
                    0.22330071,
                    0.17216755,
                ]
            ),
        )

    def test_data_rotated_w_cal_traces(self, analysis):
        assert_array_almost_equal(
            analysis["population_with_cal_traces"],
            np.array(
                [
                    -0.4816324,
                    -0.47441804,
                    -0.44791557,
                    -0.39950894,
                    -0.35339225,
                    -0.27752302,
                    -0.20578413,
                    -0.1206752,
                    -0.05760877,
                    0.02718407,
                    0.11398637,
                    0.19034061,
                    0.23514239,
                    0.2952213,
                    0.31981071,
                    0.32608839,
                    0.32965344,
                    0.30697631,
                    0.27858648,
                    0.22330071,
                    0.17216755,
                ],
            ),
        )

    def test_data_rotated_cal_traces(self, analysis):
        assert_array_almost_equal(
            analysis["population_cal_traces"],
            np.array([]),
        )

    def test_num_cal_traces(self, analysis):
        assert analysis["num_cal_traces"] == 0
