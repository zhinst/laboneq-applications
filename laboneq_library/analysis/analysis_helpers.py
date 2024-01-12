import numpy as np
from copy import deepcopy
import logging

log = logging.getLogger(__name__)

from laboneq_library.analysis import cal_trace_rotation as cal_tr_rot
from laboneq.analysis import fitting as fit_mods


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
    import lmfit

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


def flatten_lmfit_modelresult(fit_result):
    import lmfit

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


def extend_sweep_points_cal_traces(sweep_points, num_cal_traces=0):
    if num_cal_traces == 0:
        return sweep_points

    if len(sweep_points) > 1:
        dsp = sweep_points[1] - sweep_points[0]
    else:
        dsp = 1
    cal_traces_swpts = np.array(
        [sweep_points[-1] + (i + 1) * dsp for i in range(num_cal_traces)]
    )
    return np.concatenate([sweep_points, cal_traces_swpts])


def extract_and_rotate_data_1d(
    results, data_handle, cal_trace_handle_root=None, cal_states="ge", do_pca=False
):
    # extract data
    swpts = deepcopy(results.get_axis(data_handle)[0])
    if isinstance(swpts, list):
        swpts = swpts[0]
    data_raw = deepcopy(results.get_data(data_handle))
    if cal_trace_handle_root is None:
        cal_trace_handle_root = data_handle
    cal_trace_handles = [
        e for e in results.get_handles() if f"{cal_trace_handle_root}_cal_trace" in e
    ]
    num_cal_traces = len(cal_trace_handles)
    if num_cal_traces > 0:
        # rotate data to cal states
        raw_data_cal_pt_0 = results.get_data(
            f"{cal_trace_handle_root}_cal_trace_{cal_states[0]}"
        )
        raw_data_cal_pt_1 = results.get_data(
            f"{cal_trace_handle_root}_cal_trace_{cal_states[1]}"
        )
        cal_traces = np.array([raw_data_cal_pt_0, raw_data_cal_pt_1])
        data_raw_w_cal_tr = np.concatenate([data_raw, cal_traces])
        if do_pca:
            # rotate data using pca
            data_rot = cal_tr_rot.principal_component_analysis(data_raw_w_cal_tr)
        else:
            data_rot = cal_tr_rot.rotate_data_to_cal_trace_results(
                data_raw_w_cal_tr, raw_data_cal_pt_0, raw_data_cal_pt_1
            )
    else:
        # rotate data using pca
        data_rot = cal_tr_rot.principal_component_analysis(data_raw)
        data_raw_w_cal_tr = data_raw
        cal_traces = np.array([])
    swpts_w_cal_tr = extend_sweep_points_cal_traces(swpts, num_cal_traces)

    data_dict = {
        "sweep_points": swpts,
        "sweep_points_w_cal_traces": swpts_w_cal_tr,
        "sweep_points_cal_traces": swpts_w_cal_tr[
            len(swpts_w_cal_tr) - num_cal_traces :
        ],
        "data_raw": data_raw,
        "data_raw_w_cal_traces": data_raw_w_cal_tr,
        "data_raw_cal_traces": data_raw_w_cal_tr[
            len(data_raw_w_cal_tr) - num_cal_traces :
        ],
        "data_rotated": data_rot[: len(data_raw)],
        "data_rotated_w_cal_traces": data_rot,
        "data_rotated_cal_traces": data_rot[len(data_rot) - num_cal_traces :],
        "num_cal_traces": num_cal_traces,
        "do_pca": do_pca,
    }

    return data_dict


def extract_and_rotate_data_2d(
    results, data_handle, cal_trace_handle_root=None, cal_states="ge", do_pca=False
):
    # extract data
    swpts_nt = deepcopy(results.get_axis(data_handle)[0])
    swpts_rt = deepcopy(results.get_axis(data_handle)[1][0])
    data_raw = deepcopy(results.get_data(data_handle))
    if cal_trace_handle_root is None:
        cal_trace_handle_root = data_handle
    cal_trace_handles = [
        e for e in results.get_handles() if f"{cal_trace_handle_root}_cal_trace" in e
    ]
    num_cal_traces = len(cal_trace_handles)
    data_rot = np.zeros(shape=data_raw.shape)
    if num_cal_traces > 0:
        # rotate data to cal states
        raw_data_cal_pt_0 = results.get_data(
            f"{cal_trace_handle_root}_cal_trace_{cal_states[0]}"
        )
        raw_data_cal_pt_1 = results.get_data(
            f"{cal_trace_handle_root}_cal_trace_{cal_states[1]}"
        )
        cal_traces = np.array([raw_data_cal_pt_0, raw_data_cal_pt_1]).T
        data_raw_w_cal_tr = np.concatenate([data_raw, cal_traces], axis=1)
        data_rot = np.zeros(shape=data_raw_w_cal_tr.shape)
        if do_pca:
            # rotate data using pca
            for i in range(data_raw_w_cal_tr.shape[0]):
                data_rot[i, :] = cal_tr_rot.principal_component_analysis(
                    data_raw_w_cal_tr[i, :]
                )
        else:
            for i in range(data_raw_w_cal_tr.shape[0]):
                data_rot[i, :] = cal_tr_rot.rotate_data_to_cal_trace_results(
                    data_raw_w_cal_tr[i, :], raw_data_cal_pt_0[i], raw_data_cal_pt_1[i]
                )
    else:
        # rotate data using pca
        for i in range(data_raw.shape[0]):
            data_rot[i, :] = cal_tr_rot.principal_component_analysis(data_raw[i, :])
        data_raw_w_cal_tr = data_raw

    swpts_rt_w_cal_tr = extend_sweep_points_cal_traces(swpts_rt, num_cal_traces)

    data_dict = {
        "sweep_points": swpts_rt,
        "sweep_points_nt": swpts_nt,
        "sweep_points_w_cal_traces": swpts_rt_w_cal_tr,
        "sweep_points_cal_traces": swpts_rt_w_cal_tr[
            len(swpts_rt_w_cal_tr) - num_cal_traces :
        ],
        "data_raw": data_raw,
        "data_raw_w_cal_traces": data_raw_w_cal_tr,
        "data_raw_cal_traces": data_raw_w_cal_tr[
            :, data_raw_w_cal_tr.shape[1] - num_cal_traces :
        ],
        "data_rotated": data_rot[:, : data_raw.shape[1]],
        "data_rotated_w_cal_traces": data_rot,
        "data_rotated_cal_traces": data_rot[:, data_rot.shape[1] - num_cal_traces :],
        "num_cal_traces": num_cal_traces,
        "do_pca": do_pca,
    }

    return data_dict


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
        p0: list with initial guess of the parameters
        weights: fitting weights

    Returns:
        list of fitted parameters
    """
    from scipy.optimize import leastsq

    if np.isscalar(p0):
        p0 = np.array([p0])

    def residuals(params, x, y):
        if weights is not None:
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
