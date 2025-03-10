# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module contains helper function for experiment analyses."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable

import lmfit
import numpy as np
from laboneq.analysis import fitting as fit_mods

if TYPE_CHECKING:
    from typing import Literal

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
    try:
        pi_xvals = (n * np.pi - phase) / frequency
        pi2_xvals = (n * np.pi + np.pi / 2 - phase) / frequency
    except ZeroDivisionError:
        warnings.warn(
            "The frequency of the cosine function is zero. "
            "Returning empty arrays for the pi and pi/2 x-values.",
            stacklevel=2
        )
        return np.array([]), np.array([]), np.array([]), np.array([])
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


def lorentzian_fit(
    x: ArrayLike,
    data: ArrayLike,
    spectral_feature: Literal["peak", "dip", "auto"] = "auto",
    param_hints: dict[str, dict[str, float | bool | str]] | None = None,
) -> lmfit.model.ModelResult:
    """Fit a Lorentzian model to the data as a function of sweep_points.

    This function determines whether the Lorentzian structure has a peak or a
    dip by performing two fits with two guess values, the min and the max of
    the data. To determine whether there is a peak or a dip, the distance
    between the value at the fitted peak/dip is compared to the mean of the
    data array: the larger distance is the true spectroscopy signal.

    Args:
        x: numpy array of the independent variable
        data: numpy array of data to fit
        spectral_feature: whether to perform the fit assuming the Lorentzian is pointing
            upwards ("peak") or downwards ("dip"). By default, this parameter is "auto",
            in which case, the routine tries to work out the orientation of the
            Lorentzian feature.
        param_hints: dict with parameter hints for the fit (see fit_data_lmfit)

    Returns:
        an instance of lmfit.model.ModelResult
    """
    if param_hints is None:
        width_guess = 50e3

        # fit with guess values for a peak
        param_hints = {
            "amplitude": {"value": np.max(data) * width_guess},
            "position": {"value": x[np.argmax(data)]},
            "width": {"value": width_guess},
            "offset": {"value": 0},
        }
        fit_res_peak = fit_data_lmfit(
            fit_mods.lorentzian,
            x,
            data,
            param_hints=param_hints,
        )

        # fit with guess values for a dip
        param_hints["amplitude"]["value"] *= -1
        param_hints["position"]["value"] = x[np.argmin(data)]
        fit_res_dip = fit_data_lmfit(
            fit_mods.lorentzian,
            x,
            data,
            param_hints=param_hints,
        )

        if spectral_feature == "auto":
            # determine whether there is a peak or a dip: compare
            # the distance between the value at the fitted peak/dip
            # to the mean of the data array: the larger distance
            # is the true spectroscopy signal
            dpeak = abs(
                fit_res_peak.model.func(
                    fit_res_peak.best_values["position"],
                    **fit_res_peak.best_values,
                )
                - np.mean(data)
            )
            ddip = abs(
                fit_res_dip.model.func(
                    fit_res_dip.best_values["position"],
                    **fit_res_dip.best_values,
                )
                - np.mean(data)
            )
            fit_res = fit_res_peak if dpeak > ddip else fit_res_dip
        elif spectral_feature == "peak":
            fit_res = fit_res_peak
        elif spectral_feature == "dip":
            fit_res = fit_res_dip
        else:
            raise ValueError(
                f"Unrecognised spectral_feature '{spectral_feature}'. "
                f"This parameter can only be 'auto', 'peak', or 'dip'."
            )
    else:
        # do what the user asked
        fit_res = fit_data_lmfit(
            fit_mods.lorentzian,
            x,
            data,
            param_hints=param_hints,
        )
    return fit_res


def linear(
    x: ArrayLike,
    gradient: float,
    intercept: float,
) -> ArrayLike:
    """A function for modelling linear.

    Args:
        x: An array of values to evaluate the function at.
        gradient: The gradient.
        intercept: The offset.

    Returns:
        ArrayLike: The values of the linear function at the times `x`.
    """
    return gradient * x + intercept


def linear_fit(
    x: ArrayLike,
    data: ArrayLike,
    param_hints: dict[str, dict[str, float | bool | str]] | None = None,
) -> lmfit.model.ModelResult:
    """Performs a fit of a linear model to the data.

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

    gradient = (data[-1] - data[0]) / (x[-1] - x[0])
    param_hints_default = {  # good guesses for fitting a drag q-scaling measurement
        "gradient": {"value": gradient},
        "intercept": {"value": data[-1] - gradient * x[-1]},
    }
    param_hints_default.update(param_hints)

    return fit_data_lmfit(
        linear,
        x,
        data,
        param_hints=param_hints_default,
    )


def is_data_convex(x: ArrayLike, y: ArrayLike) -> bool:
    """Check if a data set is convex.

    The check is done by comparing the data points y to the line
    between the two end points of the data set (x[0], y[0]), (x[-1], y[-1]).

    Args:
        x: x values of the data set
        y: y values of the data set

    Returns:
        True if the data set is convex, else False.
    """
    if len(x) < 2:  # noqa: PLR2004
        raise ValueError("The x array must have at least two entries.")

    if x[-1] - x[0] == 0:
        raise ValueError(
            "Division by zero: the secant gradient cannot be calculated because "
            "x[-1] - x[0] == 0."
        )

    secant_gradient = (y[-1] - y[0]) / (x[-1] - x[0])
    b = y[0] - secant_gradient * x[0]
    data_line = secant_gradient * x + b
    return np.all(y[1:-1] >= data_line[1:-1])
