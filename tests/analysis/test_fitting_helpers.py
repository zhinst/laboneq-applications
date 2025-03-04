# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from laboneq.analysis import fitting as fit_mods

from laboneq_applications.analysis import fitting_helpers as fit_hlp


@pytest.fixture
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
    assert fit_res.model.name == "Model(oscillatory)"
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


@pytest.fixture
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
    assert fit_res.model.name == "Model(cosine_oscillatory_decay)"
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


@pytest.fixture
def fit_data_exp_decay():
    """Results from a lifetime_measurement experiment.

    Below, x corresponds to the time-delay after the x180 pulse in the
    lifetime_measurement experiment, and the data is the qubit excited-state
    population obtained from doing rotation and projection on the calibration
    states (g, e).
    """
    data = np.array(
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
    )
    x = np.array(
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
    return x, data


def test_exponential_decay_fit(fit_data_exp_decay):
    param_hints = {
        "offset": {"value": 0, "vary": False},
    }
    fit_res = fit_hlp.exponential_decay_fit(
        *fit_data_exp_decay, param_hints=param_hints
    )
    assert fit_res.model.name == "Model(exponential_decay)"
    np.testing.assert_allclose(
        fit_res.best_values["decay_rate"], 65645.08548221787, rtol=1e-4
    )
    np.testing.assert_allclose(
        fit_res.best_values["amplitude"], 0.971258379771985, rtol=1e-4
    )
    np.testing.assert_almost_equal(fit_res.best_values["offset"], 0)


@pytest.fixture
def fit_data_lorentzian():
    """Results from qubit-spectroscopy experiment.

    Below, x corresponds to the qubit ge frequency, and the data is the magnitude of
    the acquired complex signal, which has a downward-pointing Lorentzian line-shape.
    """

    data = np.array(
        [
            1.96115506,
            1.97167168,
            1.97574724,
            1.97508348,
            1.97027065,
            1.9782153,
            1.97901144,
            1.98068189,
            1.98272098,
            1.97764058,
            1.97731777,
            1.98334076,
            1.98316209,
            1.9842831,
            1.98353034,
            1.98324422,
            1.97663214,
            1.97817882,
            1.98195531,
            1.97946963,
            1.97737341,
            1.97921709,
            1.97869709,
            1.97831869,
            1.98143393,
            1.98041165,
            1.97766483,
            1.9833734,
            1.98579218,
            1.97745856,
            1.97783093,
            1.97778806,
            1.53335673,
            1.63088846,
            1.73985083,
            1.81740783,
            1.86778957,
            1.91063428,
            1.9414293,
            1.95288792,
            1.95651417,
            1.97141211,
            1.97442231,
            1.97626437,
            1.97624834,
            1.97792029,
            1.9829139,
            1.976315,
            1.97279769,
            1.97396463,
            1.97296576,
            1.97856192,
            1.97322007,
            1.97906554,
            1.97910662,
            1.97868223,
            1.97327943,
            1.97568842,
            1.97500094,
            1.9759244,
            1.9765965,
            1.98099916,
            1.98156601,
            1.97859128,
            1.98048876,
            1.97729464,
            1.97736198,
            1.97365126,
            1.97507859,
            1.97953144,
            1.98113901,
            1.97962915,
            1.98130595,
            1.98099216,
            1.98246194,
            1.97651464,
            1.97985085,
            1.97527139,
            1.9771229,
            1.9832143,
            1.98172825,
            1.98003474,
            1.98281759,
            1.98134343,
            1.98105307,
            1.97911766,
            1.97901714,
            1.97795731,
            1.98144341,
            1.98690111,
            1.97926907,
            1.98404887,
            1.98220711,
            1.98176834,
            1.97939118,
            1.97597327,
            1.97530496,
            1.97374594,
            1.97473532,
            1.97392353,
            1.97843611,
            1.97299418,
            1.97310134,
            1.97699442,
            1.97461831,
            1.97435024,
            1.9750972,
            1.9680742,
            1.96605148,
            1.96905762,
            1.97754094,
            1.9817749,
            1.975488,
            1.9741389,
            1.97368691,
            1.97935858,
            1.97723729,
            1.9706961,
            1.97285766,
            1.97524586,
            1.96988841,
            1.96615903,
            1.96562686,
            1.97305376,
            1.96709766,
            1.96496425,
            1.9608229,
            1.95645364,
            1.95658641,
            1.95796649,
            1.95341563,
            1.95207782,
            1.94961041,
            1.95285579,
            1.95005552,
            1.95870723,
            1.9476983,
            1.94518902,
            1.95246765,
            1.94318091,
            1.92933585,
            1.93463406,
            1.93553165,
            1.9293092,
            1.91646137,
            1.91843395,
            1.92416161,
            1.92927675,
            1.9259922,
            1.93575144,
            1.94348412,
            1.94528936,
            1.95035223,
            1.94720806,
            1.92301455,
            1.88494644,
            1.81504866,
            1.76973602,
            1.69720039,
            1.69178306,
            1.76183464,
            1.59769412,
            1.46432595,
            1.51682355,
            1.36014666,
            1.05287849,
            1.14209776,
            1.18865815,
            1.11163754,
            1.27123034,
            1.37744105,
            1.39770971,
            1.5091407,
            1.49213495,
            1.60056557,
            1.69439646,
            1.71654592,
            1.70816567,
            1.7333621,
            1.77719735,
            1.83030981,
            1.8599123,
            1.89905877,
            1.91914962,
            1.93468819,
            1.94261031,
            1.94737284,
            1.95216059,
            1.94493945,
            1.94320228,
            1.95041838,
            1.94711904,
            1.93437796,
            1.9384283,
            1.93764797,
            1.94534969,
            1.95142959,
            1.9467735,
            1.94830007,
            1.94643087,
            1.95364949,
        ]
    )
    x = np.array(
        [
            6.20433221e09,
            6.20493221e09,
            6.20553221e09,
            6.20613221e09,
            6.20673221e09,
            6.20733221e09,
            6.20793221e09,
            6.20853221e09,
            6.20913221e09,
            6.20973221e09,
            6.21033221e09,
            6.21093221e09,
            6.21153221e09,
            6.21213221e09,
            6.21273221e09,
            6.21333221e09,
            6.21393221e09,
            6.21453221e09,
            6.21513221e09,
            6.21573221e09,
            6.21633221e09,
            6.21693221e09,
            6.21753221e09,
            6.21813221e09,
            6.21873221e09,
            6.21933221e09,
            6.21993221e09,
            6.22053221e09,
            6.22113221e09,
            6.22173221e09,
            6.22233221e09,
            6.22293221e09,
            6.22353221e09,
            6.22413221e09,
            6.22473221e09,
            6.22533221e09,
            6.22593221e09,
            6.22653221e09,
            6.22713221e09,
            6.22773221e09,
            6.22833221e09,
            6.22893221e09,
            6.22953221e09,
            6.23013221e09,
            6.23073221e09,
            6.23133221e09,
            6.23193221e09,
            6.23253221e09,
            6.23313221e09,
            6.23373221e09,
            6.23433221e09,
            6.23493221e09,
            6.23553221e09,
            6.23613221e09,
            6.23673221e09,
            6.23733221e09,
            6.23793221e09,
            6.23853221e09,
            6.23913221e09,
            6.23973221e09,
            6.24033221e09,
            6.24093221e09,
            6.24153221e09,
            6.24213221e09,
            6.24273221e09,
            6.24333221e09,
            6.24393221e09,
            6.24453221e09,
            6.24513221e09,
            6.24573221e09,
            6.24633221e09,
            6.24693221e09,
            6.24753221e09,
            6.24813221e09,
            6.24873221e09,
            6.24933221e09,
            6.24993221e09,
            6.25053221e09,
            6.25113221e09,
            6.25173221e09,
            6.25233221e09,
            6.25293221e09,
            6.25353221e09,
            6.25413221e09,
            6.25473221e09,
            6.25533221e09,
            6.25593221e09,
            6.25653221e09,
            6.25713221e09,
            6.25773221e09,
            6.25833221e09,
            6.25893221e09,
            6.25953221e09,
            6.26013221e09,
            6.26073221e09,
            6.26133221e09,
            6.26193221e09,
            6.26253221e09,
            6.26313221e09,
            6.26373221e09,
            6.26433221e09,
            6.26493221e09,
            6.26553221e09,
            6.26613221e09,
            6.26673221e09,
            6.26733221e09,
            6.26793221e09,
            6.26853221e09,
            6.26913221e09,
            6.26973221e09,
            6.27033221e09,
            6.27093221e09,
            6.27153221e09,
            6.27213221e09,
            6.27273221e09,
            6.27333221e09,
            6.27393221e09,
            6.27453221e09,
            6.27513221e09,
            6.27573221e09,
            6.27633221e09,
            6.27693221e09,
            6.27753221e09,
            6.27813221e09,
            6.27873221e09,
            6.27933221e09,
            6.27993221e09,
            6.28053221e09,
            6.28113221e09,
            6.28173221e09,
            6.28233221e09,
            6.28293221e09,
            6.28353221e09,
            6.28413221e09,
            6.28473221e09,
            6.28533221e09,
            6.28593221e09,
            6.28653221e09,
            6.28713221e09,
            6.28773221e09,
            6.28833221e09,
            6.28893221e09,
            6.28953221e09,
            6.29013221e09,
            6.29073221e09,
            6.29133221e09,
            6.29193221e09,
            6.29253221e09,
            6.29313221e09,
            6.29373221e09,
            6.29433221e09,
            6.29493221e09,
            6.29553221e09,
            6.29613221e09,
            6.29673221e09,
            6.29733221e09,
            6.29793221e09,
            6.29853221e09,
            6.29913221e09,
            6.29973221e09,
            6.30033221e09,
            6.30093221e09,
            6.30153221e09,
            6.30213221e09,
            6.30273221e09,
            6.30333221e09,
            6.30393221e09,
            6.30453221e09,
            6.30513221e09,
            6.30573221e09,
            6.30633221e09,
            6.30693221e09,
            6.30753221e09,
            6.30813221e09,
            6.30873221e09,
            6.30933221e09,
            6.30993221e09,
            6.31053221e09,
            6.31113221e09,
            6.31173221e09,
            6.31233221e09,
            6.31293221e09,
            6.31353221e09,
            6.31413221e09,
            6.31473221e09,
            6.31533221e09,
            6.31593221e09,
            6.31653221e09,
            6.31713221e09,
            6.31773221e09,
            6.31833221e09,
            6.31893221e09,
            6.31953221e09,
            6.32013221e09,
            6.32073221e09,
            6.32133221e09,
            6.32193221e09,
            6.32253221e09,
            6.32313221e09,
            6.32373221e09,
            6.32433221e09,
        ]
    )
    return x, data


def test_lorentzian_fit(fit_data_lorentzian):
    fit_res = fit_hlp.lorentzian_fit(*fit_data_lorentzian)
    assert fit_res.model.name == "Model(lorentzian)"
    np.testing.assert_allclose(
        fit_res.best_values["position"], 6304602810.510597, rtol=1e-4
    )
    np.testing.assert_allclose(
        fit_res.best_values["width"], 3200590.950376773, rtol=1e-4
    )
    np.testing.assert_allclose(
        fit_res.best_values["amplitude"], -2744250.8775792723, rtol=1e-4
    )
    np.testing.assert_allclose(
        fit_res.best_values["offset"], 1.9721753641263535, rtol=1e-4
    )


def test_lorentzian_fit_spectral_feature(fit_data_lorentzian):
    fit_res = fit_hlp.lorentzian_fit(*fit_data_lorentzian, spectral_feature="peak")
    assert fit_res.model.name == "Model(lorentzian)"
    np.testing.assert_allclose(
        fit_res.best_values["position"], 6241483923.483863, rtol=1e-4
    )
    np.testing.assert_allclose(
        fit_res.best_values["width"], 67111210.43146636, rtol=1e-4
    )
    np.testing.assert_allclose(
        fit_res.best_values["amplitude"], 27300080.192143552, rtol=1e-4
    )
    np.testing.assert_allclose(
        fit_res.best_values["offset"], 1.587935088800723, rtol=1e-4
    )


def test_lorentzian_fit_spectral_feature_invalid_input(fit_data_lorentzian):
    with pytest.raises(ValueError) as err:
        fit_hlp.lorentzian_fit(*fit_data_lorentzian, spectral_feature="zig-zag")
    error_string = (
        "Unrecognised spectral_feature 'zig-zag'. "
        "This parameter can only be 'auto', 'peak', or 'dip'."
    )
    assert str(err.value) == error_string


def test_linear_fit():
    x = np.linspace(0, 1, 31)
    data = fit_hlp.linear(x, gradient=10, intercept=2.34)
    fit_res = fit_hlp.linear_fit(x, data)

    assert fit_res.model.name == "Model(linear)"
    np.testing.assert_equal(fit_res.best_values["gradient"], 10)
    np.testing.assert_equal(fit_res.best_values["intercept"], 2.34)
