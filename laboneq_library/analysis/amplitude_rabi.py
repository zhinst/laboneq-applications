from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
import uncertainties as unc
from laboneq.analysis import fitting as fit_mods
from laboneq_library.analysis import analysis_helpers


if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    import lmfit


@dataclass
class AmplitudeRabiAnalysisResult:
    """Amplitude Rabi results.

    Attributes:
        pi_amplitude: Pi amplitude.
        pi2_amplitude: PI2 amplitude.
        model: A fitting model results used for the analysis.
    """
    # TODO: Should not probably depend on `uncertainties` data structures
    # TODO: Include std_dev / nominal value as separate attribute for both amplitude?
    pi_amplitude: unc.UFloat
    pi2_amplitude: unc.UFloat
    model: lmfit.model.ModelResult


# extract
def calculate_rabi_amplitude(data: ArrayLike, amplitudes: ArrayLike, param_hints: dict = None) -> AmplitudeRabiAnalysisResult:
    """Calculate Rabi amplitude.

    Arguments:
        data: Input data
        amplitudes: Swept aplitudes
        param_hint: A dictionary of parameter hints for the fitting model.

    Returns:
        Amplitude fitting results.
    """
    # TODO: Turn `param_hints` to an callable to access e.g `freq_guess` / `phase_guess`
    # TODO: Separate fitting / fit analysis for generic usage
    freqs_guess, phase_guess = analysis_helpers.find_oscillation_frequency_and_phase(
        data, amplitudes
    )
    if not param_hints:
        param_hints = {
            "frequency": {"value": 2 * np.pi * freqs_guess, "min": 0},
            "phase": {"value": phase_guess},
            "amplitude": {
                "value": abs(max(data) - min(data)) / 2,
                "min": 0,
            },
            "offset": {"value": np.mean(data)},
        }

    fit_res = analysis_helpers.fit_data_lmfit(
        fit_mods.oscillatory, amplitudes, data, param_hints=param_hints
    )

    freq_fit = unc.ufloat(
        fit_res.params["frequency"].value, fit_res.params["frequency"].stderr
    )
    phase_fit = unc.ufloat(
        fit_res.params["phase"].value, fit_res.params["phase"].stderr
    )
    (
        pi_amps_top,
        pi_amps_bottom,
        pi2_amps_rise,
        pi2_amps_fall,
    ) = analysis_helpers.get_pi_pi2_xvalues_on_cos(amplitudes, freq_fit, phase_fit)
    # if pca is done, it can happen that the pi-pulse amplitude
    # is in pi_amps_bottom and the pi/2-pulse amplitude in pi2_amps_fall
    pi_amps = np.sort(np.concatenate([pi_amps_top, pi_amps_bottom]))
    pi2_amps = np.sort(np.concatenate([pi2_amps_rise, pi2_amps_fall]))
    try:
        pi2_amp = pi2_amps[0]
        pi_amp = pi_amps[pi_amps > pi2_amp][0]
    except IndexError:
        # TODO: Kept to ensure old experiment functionality. Remove later.
        return AmplitudeRabiAnalysisResult(
            pi_amplitude=None,
            pi2_amplitude=None,
            model=fit_res
        )
    return AmplitudeRabiAnalysisResult(
        pi_amplitude=pi_amp,
        pi2_amplitude=pi2_amp,
        model=fit_res
    )
