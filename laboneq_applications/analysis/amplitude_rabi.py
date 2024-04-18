from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
import uncertainties as unc
from laboneq.analysis import fitting as fit_mods
from laboneq_applications.analysis import analysis_helpers


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


def extract_rabi_amplitude(data: ArrayLike, amplitudes: ArrayLike, param_hints: dict = None) -> AmplitudeRabiAnalysisResult:
    """Extract Rabi amplitude.

    Arguments:
        data: Input data
        amplitudes: Swept amplitudes
        param_hint: A dictionary of parameter hints for the fitting model.

    Returns:
        Amplitude fitting results.
    """
    (
        fit_res,
        pi_amps_top,
        pi_amps_bottom,
        pi2_amps_rise,
        pi2_amps_fall,
    ) = analysis_helpers.cosine_oscillatory_fit(data, amplitudes, param_hints=param_hints)
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
