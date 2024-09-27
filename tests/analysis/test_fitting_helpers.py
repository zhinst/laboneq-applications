import numpy as np
import pytest
from laboneq.analysis import fitting as fit_mods

from laboneq_applications.analysis import fitting_helpers as fit_hlp


@pytest.fixture()
def fit_data():
    """Results from AmplitudeRabi experiment."""
    data = np.array(
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
    )
    x = np.array(
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
    )
    return x, data


def test_fit_data_lmfit(fit_data):
    param_hints = {
        "frequency": {"value": 2 * np.pi * 2.0304751167609485, "min": 0},
        "phase": {"value": 1.7951955468942824},
        "amplitude": {
            "value": abs(max(fit_data[1]) - min(fit_data[1])) / 2,
            "min": 0,
        },
        "offset": {"value": np.mean(fit_data[1])},
    }

    fit_res = fit_hlp.fit_data_lmfit(
        fit_mods.oscillatory,
        *fit_data,
        param_hints=param_hints,
    )
    np.testing.assert_almost_equal(
        fit_res.best_values["frequency"],
        8.944142729731128,
    )
    np.testing.assert_almost_equal(fit_res.best_values["phase"], 3.1536457220232963)
    np.testing.assert_almost_equal(fit_res.best_values["amplitude"], 1.069478377275622)
    np.testing.assert_almost_equal(fit_res.best_values["offset"], -0.1835670924745826)


def test_find_oscillation_frequency_and_phase(fit_data):
    freq_guess, phase_guess = fit_hlp.find_oscillation_frequency_and_phase(*fit_data)
    np.testing.assert_almost_equal(freq_guess, 2.0304751167609485)
    np.testing.assert_almost_equal(phase_guess, 1.7951955468942824)


def test_cosine_oscillatory_fit(fit_data):
    fit_res = fit_hlp.cosine_oscillatory_fit(*fit_data)
    fit_res.model.name = "Model(oscillatory)"
    np.testing.assert_almost_equal(
        fit_res.best_values["frequency"],
        8.944142729731128,
    )
    np.testing.assert_almost_equal(fit_res.best_values["phase"], 3.1536457220232963)
    np.testing.assert_almost_equal(fit_res.best_values["amplitude"], 1.069478377275622)
    np.testing.assert_almost_equal(fit_res.best_values["offset"], -0.1835670924745826)


def test_get_pi_pi2_xvalues_on_cos(fit_data):
    pixv_top, pixv_bottom, pi2xv_rising, pi2xv_falling = (
        fit_hlp.get_pi_pi2_xvalues_on_cos(
            x=fit_data[0],
            frequency=8.944142729731128,
            phase=3.1536457220232963,
        )
    )
    np.testing.assert_array_almost_equal(pixv_top, np.array([0.34989822]))
    np.testing.assert_array_almost_equal(pixv_bottom, np.array([]))
    np.testing.assert_array_almost_equal(pi2xv_rising, np.array([0.17427531]))
    np.testing.assert_array_almost_equal(pi2xv_falling, np.array([]))


@pytest.fixture()
def fit_data_cos_osc_decay_fit():
    """Results from Ramsey experiment.

    Below, x corresponds to the time-separation between the two x90 pulses in the Ramsey
    experiment, and the data is the qubit excited-state population obtained from doing
    rotation and projection on the calibration states (g, e).
    """
    data = np.array(
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
    )
    x = np.array(
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
    return x, data


def test_cosine_oscillatory_decay_fit(fit_data_cos_osc_decay_fit):
    param_hints = {
        "amplitude": {"value": 0.5, "vary": False},
        "oscillation_offset": {"value": 0, "vary": False},
    }
    fit_res = fit_hlp.cosine_oscillatory_decay_fit(
        *fit_data_cos_osc_decay_fit, param_hints=param_hints
    )
    fit_res.model.name = "Model(cosine_oscillatory_decay)"
    np.testing.assert_allclose(
        fit_res.best_values["frequency"], 672104.6105811436, rtol=1e-4
    )
    np.testing.assert_allclose(
        fit_res.best_values["decay_time"], 2.315185344525794e-05, rtol=1e-4
    )
    np.testing.assert_allclose(
        fit_res.best_values["phase"], 6.260565448213821, rtol=1e-4
    )
    np.testing.assert_allclose(fit_res.best_values["amplitude"], 0.5)
    np.testing.assert_allclose(fit_res.best_values["oscillation_offset"], 0.0)
    np.testing.assert_allclose(
        fit_res.best_values["exponential_offset"], 0.49582477545145665, rtol=1e-4
    )
    np.testing.assert_allclose(fit_res.best_values["decay_exponent"], 1)
