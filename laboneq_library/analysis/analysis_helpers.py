from scipy.optimize import leastsq
from numpy.typing import ArrayLike
import numpy as np
import logging
import lmfit
import uncertainties as unc

from laboneq.analysis import fitting as fit_mods

log = logging.getLogger(__name__)


def find_oscillation_frequency_and_phase(data, time):
    w = np.fft.fft(data)
    f = np.fft.fftfreq(len(data), time[1] - time[0])
    mask = f > 0
    w, f = w[mask], f[mask]
    abs_w = np.abs(w)
    freq = f[np.argmax(abs_w)]
    phase = 2 * np.pi - (2 * np.pi * time[np.argmax(data)] * freq)
    return freq, phase


def fit_data_lmfit(model, x, y, param_hints):
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
            log.warning(f"Stderr for {par} is None. Setting it to 0.")
            fit_res.params[par].stderr = 0
    return fit_res


def cosine_oscillatory_fit(data: ArrayLike, x: ArrayLike, param_hints: dict = None) -> lmfit.model.ModelResult:
    """Descriptive name"""
    freqs_guess, phase_guess = find_oscillation_frequency_and_phase(
        data, x
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

    fit_result = fit_data_lmfit(
        fit_mods.oscillatory, x, data, param_hints=param_hints
    )
    freq_fit = unc.ufloat(
        fit_result.params["frequency"].value,
        fit_result.params["frequency"].stderr,
    )
    phase_fit = unc.ufloat(
        fit_result.params["phase"].value,
        fit_result.params["phase"].stderr,
    )
    (
        x1_pos,
        x1_neg,
        x0_rising_edge,
        x0_falling_edge,
    ) = get_pi_pi2_xvalues_on_cos(x, freq_fit, phase_fit)
    return fit_result, x1_pos, x1_neg, x0_rising_edge, x0_falling_edge


def flatten_lmfit_modelresult(fit_result):
    # used for saving an lmfit ModelResults object as a dict
    assert isinstance(fit_result, lmfit.model.ModelResult)
    fit_res_dict = dict()
    fit_res_dict["success"] = fit_result.success
    fit_res_dict["message"] = fit_result.message
    fit_res_dict["params"] = {}
    for param_name in fit_result.params:
        fit_res_dict["params"][param_name] = {}
        param = fit_result.params[param_name]
        for k in param.__dict__:
            if k == "_val":
                fit_res_dict["params"][param_name]["value"] = getattr(param, k)
            else:
                if not k.startswith("_") and k not in [
                    "from_internal",
                ]:
                    fit_res_dict["params"][param_name][k] = getattr(param, k)
    return fit_res_dict


def is_data_convex(x, y):
    """
    Check if a data set is convex by comparing the data points y to the line
    between the two end points of the data set (x[0], y[0]), (x[-1], y[-1]).

    Args:
        x: x values of the data set
        y: y values of the data set

    Returns:
        True if the data set is convex, else False
    """
    secant_gradient = (y[-1] - y[0]) / (x[-1] - x[0])
    b = y[0] - secant_gradient * x[0]
    l = secant_gradient * x + b
    return np.all(y[1:-1] >= l[1:-1])


def get_pi_pi2_xvalues_on_cos(x, frequency, phase):
    """
    Calculate the x-values of a cosine function of the form
    cos(x * frequency + phase) at which x * frequency + phase = n * pi
    and x * frequency + phase = n * pi + pi/2.

    Only the values that are inside the interval [min(x), max(x)] are returned.

    Args:
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
    mask_func = lambda cos_xvals: np.logical_and(
        cos_xvals >= min(x), cos_xvals <= max(x)
    )
    pixv_top = pi_xvals_top[mask_func(pi_xvals_top)]
    pixv_bottom = pi_xvals_bottom[mask_func(pi_xvals_bottom)]
    pi2xv_rising = pi2_xvals_rising[mask_func(pi2_xvals_rising)]
    pi2xv_falling = pi2_xvals_falling[mask_func(pi2_xvals_falling)]

    return pixv_top, pixv_bottom, pi2xv_rising, pi2xv_falling


def fit_lorentzian(data, sweep_points, param_hints=None):
    """
    Fit the Lorentzian model fit_mods.lorentzian to the data as a function of
    sweep_points.

    This function determines whether the Lorentzian structure has a peak or a
    dip by performing two fits with two guess values, the min and the max of
    the data. To determine whether there is a peak or a dip, the distance
    between the value at the fitted peak/dip is compared to the mean of the
    data array: the larger distance is the true spectroscopy signal.

    Args:
        data: numpy array of data to fit
        sweep_points: numpy array of the independent variable
        param_hints: dict with parameter hints for the fit (see fit_data_lmfit)

    Returns:
        an instance of lmfit.model.ModelResult
    """
    if param_hints is None:
        width_guess = 50e3
        # fit with guess values for a peak
        param_hints = {
            "amplitude": {"value": np.max(data) * width_guess},
            "position": {"value": sweep_points[np.argmax(data)]},
            "width": {"value": width_guess},
            "offset": {"value": 0},
        }
        fit_res_peak = fit_data_lmfit(
            fit_mods.lorentzian,
            sweep_points,
            data,
            param_hints=param_hints,
        )
        # fit with guess values for a dip
        param_hints["amplitude"]["value"] *= -1
        param_hints["position"]["value"] = sweep_points[np.argmin(data)]
        fit_res_dip = fit_data_lmfit(
            fit_mods.lorentzian,
            sweep_points,
            data,
            param_hints=param_hints,
        )
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
    else:
        # do what the user asked
        fit_res = fit_data_lmfit(
            fit_mods.lorentzian,
            sweep_points,
            data,
            param_hints=param_hints,
        )
    return fit_res


def cavity_complex_fit(fit_func, xData, yData, p0, weights=None):
    """
    Performs a complex fit of the complex signal from a cavity reflection measurement.

    Args:
        fit_func: fitting function
        xData: frequency array
        yData: complex data to be fitted
        p0: scalar or list with initial guesses of the parameters
        weights: fitting weights

    Returns:
        list of fitted parameters
    """

    if np.isscalar(p0):
        p0 = np.array([p0])

    def residuals(params, x, y):
        if weights is not None:
            # TODO Steph 2024.01.16: Are the weights used correctly here?
            diff = weights * fit_func(x, *params) - y
        else:
            diff = fit_func(x, *params) - y

        flatDiff = np.zeros(diff.size * 2, dtype=np.float64)
        flatDiff[0 : flatDiff.size : 2] = diff.real
        flatDiff[1 : flatDiff.size : 2] = diff.imag
        return flatDiff

    popt, bar = leastsq(residuals, p0, args=(xData, yData), maxfev=10000)
    return popt


# Single-mode cavity reflection
def cavity_1p1m_S11(f, f0, kappa_c, kappa_i, a_in, T_delay):
    """
    Fitting function for the complex signal from a cavity reflection measurement.

    Args:
        f: frequency
        f0: resonance frequency
        kappa_c:
        kappa_i:
        a_in:
        T_delay:

    kappa_c and kappa_i are kappas / 2*pi (in Hz)

    Returns:
        data array of the complex model
    """

    Delta = f - f0
    num = 1.0j * Delta + (kappa_i - kappa_c) / 2
    den = 1.0j * Delta + (kappa_i + kappa_c) / 2
    if kappa_c > 0 and kappa_i > 0 and f0 > 0:
        return num / den * a_in * np.exp(-1j * Delta * T_delay)
    else:
        return np.Inf


# Single mode cavity reflection fitting function
def fit_cavity_1p1m_S11(f, a_out, param_hints=None):
    """
    Complex fit of the signal from a cavity reflection measurement.

    The aux function is written for the complex fit and can return a complex result,
    but all its variables must be real. cavity_1p1m_amp receives complex input
    amplitude a_in

    Args:
        f: frequency
        a_out: complex data measured in reflection
        param_hints: dict with parameter hints for the fit (see fit_data_lmfit)

    Returns:
        list of fitted parameters
    """

    if param_hints is None:
        param_hints = {}
    f0 = param_hints.get("f0", {}).get("value", 7.5737e9)
    kappa_c = param_hints.get("kappa_c", {}).get("value", 5e3)
    kappa_i = param_hints.get("kappa_i", {}).get("value", 5e3)
    a_in = param_hints.get("a_in", {}).get("value", None)
    T_delay = param_hints.get("T_delay", {}).get("value", 70e-9)

    def aux_1p1m(f, f0, kappa_c, kappa_i, Re_a_in, Im_a_in, T_delay):
        return cavity_1p1m_S11(
            f, f0, kappa_c, kappa_i, Re_a_in + 1.0j * Im_a_in, T_delay
        )

    if a_in is None:
        a_in = a_out[0]
    popt = cavity_complex_fit(
        aux_1p1m,
        f,
        a_out,
        (f0, kappa_c, kappa_i, np.real(a_in), np.imag(a_in), T_delay),
    )

    return [popt[0], popt[1], popt[2], popt[3] + 1.0j * popt[4], popt[5]]


def sorted_mesh(xvals, yvals, zvals):
    """
    Prepare the x, y, z arrays to be plotted with matplotlib pcolormesh.

    Ensures that the z values are sorted according to the values in xvals and yvals and
    creates np.meshgrid from xvals and yvals.

    Args:
        xvals: array of the values to be plotted on the x-axis: typically the real-time
            sweep points
        yvals: array of the values to be plotted on the y-axis: typically the near-time
            sweep points
        zvals: array of the values to be plotted on the z-axis: typically the data

    Returns:
        the x, y, and z values to be passed directly to pcolormesh

    """
    # First, we need to sort the data as otherwise we get odd plotting
    # artefacts. An example is e.g., plotting a fourier transform
    sorted_x_arguments = xvals.argsort()
    xvals = xvals[sorted_x_arguments]
    sorted_y_arguments = yvals.argsort()
    yvals = yvals[sorted_y_arguments]
    zvals_srt = zvals[:,  sorted_x_arguments]
    zvals_srt = zvals_srt[sorted_y_arguments, :]

    xgrid, ygrid = np.meshgrid(xvals, yvals)

    return xgrid, ygrid, zvals_srt


def oscillatory_decay_flexible(
    x: ArrayLike,
    frequency: float,
    phase: float,
    decay_time: float,
    amplitude: float = 1.0,
    oscillation_offset: float = 0.0,
    exponential_offset: float = 0.0,
    decay_exponent: float = 1.0,
) -> ArrayLike:
    """A function for modelling decaying oscillations such as Ramsey and Echo
    decay.

    The form of the function is a decaying cosine:

    $$
        f(x) = amplitude \\times
               (\\cos(2 \\times \\pi \\times frequency  \\times x + phase) +
               oscillation \\text{\\textunderscore} offset)
               \\exp(- (\\frac{x}{decay \\text{\\textunderscore} time})
               ^ decay \\text{\\textunderscore} exponent) +
               exponential \\text{\\textunderscore} offset
    $$

    Calling this function evaluates it.

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
            The values of the decaying oscillation function at the times `x`.

    Examples:
        Evaluate the function:
        ``` py
        x = np.linspace(0, 10, 100)
        values = oscillatory_decay_flexible(
            x, 1e6, np.pi / 2, 1e-6, 0.5, 0, 0, 1)
        ```

    """
    return (
        amplitude
        * np.exp(-((x / decay_time) ** decay_exponent))
        * (np.cos(2 * np.pi * frequency * x + phase) + oscillation_offset)
        + exponential_offset
    )


def transmon_voltage_dependence_quadratic(
    x: ArrayLike,
    voltage_sweet_spot: float,
    frequency_sweet_spot: float,
    frequency_voltage_scaling: float,
) -> ArrayLike:
    """A function for the quadratic approximation of the transmon frequency
    dependence on an external dc voltage.

    The form of the quadratic approximation function is:

    $$
        f(x) = frequency \\text{\\textunderscore} sweet \\text{\\textunderscore} spot -
            frequency \\text{\\textunderscore} voltage \\text{\\textunderscore} scaling
            \\times (x - voltage \\text{\\textunderscore} sweep \\text{\\textunderscore} spot) ^ 2)
    $$

    Calling this function evaluates it.

    Arguments:
        x:
            An array of values to evaluate the function at.
        voltage_sweet_spot:
            voltage value at sweet spot given by the offset of the quadratic
            function along the voltage axis
        frequency_sweet_spot:
            frequency value at sweet spot given by the offset of the quadratic
            function along the frequency axis
        frequency_voltage_scaling:
            frequency-to-voltage conversion factor

    Returns:
        values:
            The values of the function at the voltage values `x`.

    Examples:
        Evaluate the function:
        ``` py
        x = np.linspace(-5, 5, 100)
        values = transmon_voltage_dependence_quadratic(x, 1, 6e9, 5.5e9)
        ```
    """
    return (
        frequency_sweet_spot - frequency_voltage_scaling * (x - voltage_sweet_spot) ** 2
    )


def linear_dependence(
    x: ArrayLike,
    slope: float,
    intercept: float,
) -> ArrayLike:
    """A function for the quadratic approximation of the transmon frequency
    dependence on an external dc voltage.

    The form of the quadratic approximation function is:

    $$
        f(x) = slope \\times x + intercept
    $$

    Calling this function evaluates it.

    Arguments:
        x:
            An array of values to evaluate the function at.
        slope:
            value of the slope
        intercept:
            value of the intercept

    Returns:
        values:
            The values of the function at the values `x`.

    Examples:
        Evaluate the function:
        ``` py
        x = np.linspace(-5, 5, 100)
        values = linear_dependence(x, 0.5, 1)
        ```
    """

    return slope * x + intercept
