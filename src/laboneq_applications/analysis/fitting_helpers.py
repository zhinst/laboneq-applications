"""This module contains helper function for experiment analyses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import lmfit
import numpy as np
from laboneq.analysis import fitting as fit_mods

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def fit_data_lmfit(
    model: lmfit.model.Model | Callable | str,
    x: ArrayLike,
    y: ArrayLike,
    param_hints: dict,
) -> lmfit.model.ModelResult:
    """Performs a fit of model to the data (x,y).

    Arguments:
        model: the model to fit to
        x: the independent variable
        y: the data to fit
        param_hints: dictionary of guesses for the fit parameters. See the lmfit
            docstring for details on the form of the parameter hints dictionary:
            https://lmfit.github.io/lmfit-py/model.html#lmfit.model.Model.set_param_hint

    Returns:
        the lmfit result
    """
    if isinstance(model, str):
        # string with the name of an lmfit model
        model = lmfit.models.lmfit_models[model]()
    elif not isinstance(model, lmfit.model.Model):
        # a fitting function: needs to be converted to an lmfit model
        model = lmfit.Model(model)

    model.param_hints = param_hints
    fit_res = model.fit(x=x, data=y, params=model.make_params())
    for par in fit_res.params:
        if fit_res.params[par].stderr is None:
            # Stderr for par is None. Setting it to 0.
            fit_res.params[par].stderr = 0
    return fit_res


def find_oscillation_frequency_and_phase(
    time: ArrayLike,
    data: ArrayLike,
) -> tuple[ArrayLike, ArrayLike]:
    """Extracts the frequency and phase of an oscillatory data set.

    The frequency and phase are extracted using a fast-Fourier analysis.

    Arguments:
        data: the data describing the oscillation
        time: the independent variable

    Returns:
        the frequency and phase values
    """
    w = np.fft.fft(data)
    f = np.fft.fftfreq(len(data), time[1] - time[0])
    mask = f > 0
    w, f = w[mask], f[mask]
    abs_w = np.abs(w)
    freq = f[np.argmax(abs_w)]
    phase = 2 * np.pi - (2 * np.pi * time[np.argmax(data)] * freq)
    return freq, phase


def cosine_oscillatory_fit(
    x: ArrayLike,
    data: ArrayLike,
    param_hints: dict | None = None,
) -> lmfit.model.ModelResult:
    """Performs a fit of a cosine model to data.

    Arguments:
        data: the data to be fitted
        x: the independent variable
        param_hints: dictionary of guesses for the fit parameters. See the lmfit
            docstring for details on the form of the parameter hints dictionary:
            https://lmfit.github.io/lmfit-py/model.html#lmfit.model.Model.set_param_hint

    Returns:
        The lmfit result
    """
    if not param_hints:
        freqs_guess, phase_guess = find_oscillation_frequency_and_phase(
            x,
            data,
        )

        param_hints = {
            "frequency": {"value": 2 * np.pi * freqs_guess, "min": 0},
            "phase": {"value": phase_guess},
            "amplitude": {
                "value": abs(max(data) - min(data)) / 2,
                "min": 0,
            },
            "offset": {"value": np.mean(data)},
        }

    return fit_data_lmfit(
        fit_mods.oscillatory,
        x,
        data,
        param_hints=param_hints,
    )


def exponential_decay_fit(
    x: ArrayLike,
    data: ArrayLike,
    param_hints: dict[str, dict[str, float | bool | str]] | None = None,
) -> lmfit.model.ModelResult:
    """Performs a fit of an exponential-decay model to data.

    Arguments:
        data: the data to be fitted
        x: the independent variable
        param_hints: dictionary of guesses for the fit parameters. See the lmfit
            docstring for details on the form of the parameter hints dictionary:
            https://lmfit.github.io/lmfit-py/model.html#lmfit.model.Model.set_param_hint

    Returns:
        The lmfit result
    """
    if param_hints is None:
        param_hints = {}
    param_hints_to_use = {
        "decay_rate": {"value": 2 / (3 * np.max(x))},
        "amplitude": {"value": data[0]},
        "offset": {"value": 0},
    }
    param_hints_to_use.update(param_hints)
    return fit_data_lmfit(
        fit_mods.exponential_decay,
        x,
        data,
        param_hints=param_hints_to_use,
    )


def get_pi_pi2_xvalues_on_cos(
    x: ArrayLike,
    frequency: float | ArrayLike,
    phase: float | ArrayLike,
) -> tuple[float | ArrayLike, float | ArrayLike, float | ArrayLike, float | ArrayLike]:
    """Calculate the useful x-values of a cosine function.

    Calculate the x-values of a cosine function of the form
    cos(x * frequency + phase) at which
    x * frequency + phase = n * pi and x * frequency + phase = n * pi + pi/2.

    Only the values that are inside the interval [min(x), max(x)] are returned.
    Note: To achieve this, we have heuristically chosen n=+/-20 periods of the
    fitted cosine to make sure the entire range spanned by x. Then we use a mask to
    return only the pi and pi/2 values of the cosine within this range.

    Arguments:
        x: array of x-values of the cosine function, corresponding for example
            to:
             - amplitudes in an amplitude-Rabi measurement
             - voltages in a resonator-spectroscopy-vs-dc-voltage measurement
             - phases in a dynamic-phase measurement
        frequency: the frequency of the cosine oscillation in angular units
        phase: the phase of the cosine oscillation in radians

    Returns: the following array in the following order
        - x-values for which cos(x * frequency + phase) = 1
        - x-values for which cos(x * frequency + phase) = - 1
        - x-values for which cos(x * frequency + phase) = 0 on the rising edge
        - x-values for which cos(x * frequency + phase) = 0 on the falling edge

    """
    n = np.arange(-20, 20)
    pi_xvals = (n * np.pi - phase) / frequency
    pi2_xvals = (n * np.pi + np.pi / 2 - phase) / frequency
    pi_xvals_top = pi_xvals[0::2]
    pi_xvals_bottom = pi_xvals[1::2]
    pi2_xvals_rising = pi2_xvals[1::2]
    pi2_xvals_falling = pi2_xvals[0::2]

    def mask_func(cos_xvals: ArrayLike) -> ArrayLike:
        return np.logical_and(
            cos_xvals >= min(x),
            cos_xvals <= max(x),
        )

    pixv_top = pi_xvals_top[mask_func(pi_xvals_top)]
    pixv_bottom = pi_xvals_bottom[mask_func(pi_xvals_bottom)]
    pi2xv_rising = pi2_xvals_rising[mask_func(pi2_xvals_rising)]
    pi2xv_falling = pi2_xvals_falling[mask_func(pi2_xvals_falling)]

    return pixv_top, pixv_bottom, pi2xv_rising, pi2xv_falling


def cosine_oscillatory_decay(
    x: ArrayLike,
    frequency: float,
    phase: float,
    decay_time: float,
    amplitude: float = 1.0,
    oscillation_offset: float = 0.0,
    exponential_offset: float = 0.0,
    decay_exponent: float = 1.0,
) -> ArrayLike:
    """A function for modelling decaying oscillations such as Ramsey and Echo decay.

    Arguments:
        x:
            An array of values to evaluate the function at.
        frequency:
            The frequency of the cosine.
        phase:
            The phase of the cosine.
        decay_time:
            The exponential decay time.
        amplitude:
            The amplitude of the cosine.
        oscillation_offset:
            The offset of the oscillatory part of the function.
        exponential_offset:
            The offset of the exponential-decay part of the function.
        decay_exponent:
            Exponential decay exponent power

    Returns:
        values:
            The values of the decaying oscillation function at the values `x`.
    """
    return (
        amplitude
        * np.exp(-((x / decay_time) ** decay_exponent))
        * (np.cos(2 * np.pi * frequency * x + phase) + oscillation_offset)
        + exponential_offset
    )


def cosine_oscillatory_decay_fit(
    x: ArrayLike,
    data: ArrayLike,
    param_hints: dict[str, dict[str, float | bool | str]] | None = None,
) -> lmfit.model.ModelResult:
    """Performs a fit of an exponentially decaying cosine model to data.

    Arguments:
        data: the data to be fitted
        x: the independent variable
        param_hints: dictionary of guesses for the fit parameters. See the lmfit
            docstring for details on the form of the parameter hints dictionary:
            https://lmfit.github.io/lmfit-py/model.html#lmfit.model.Model.set_param_hint

    Returns:
        The lmfit result
    """
    freqs_guess, phase_guess = find_oscillation_frequency_and_phase(
        x,
        data,
    )
    if param_hints is None:
        param_hints = {}
    param_hints_default = {  # good guesses for fitting a qubit state population
        "frequency": {"value": freqs_guess},
        "phase": {"value": phase_guess},
        "decay_time": {"value": 3 * np.max(x) / 2, "min": 0},
        "amplitude": {"value": 0.5, "vary": True},
        "oscillation_offset": {"value": 0},
        "exponential_offset": {"value": np.mean(data)},
        "decay_exponent": {"value": 1, "vary": False},
    }
    param_hints_default.update(param_hints)

    return fit_data_lmfit(
        cosine_oscillatory_decay,
        x,
        data,
        param_hints=param_hints_default,
    )
