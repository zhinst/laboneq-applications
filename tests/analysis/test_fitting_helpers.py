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
