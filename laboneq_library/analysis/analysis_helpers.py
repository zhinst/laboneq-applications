import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import logging
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("analysis_helpers")

from laboneq_library.analysis import cal_trace_rotation as cal_tr_rot


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
            log.warning(f'Stderr for {par} is None. Setting it to 0.')
            fit_res.params[par].stderr = 0
    return fit_res


def flatten_lmfit_modelresult(fit_result):
    import lmfit
    # used for saving an lmfit ModelResults object as a dict
    assert isinstance(fit_result, lmfit.model.ModelResult)
    fit_res_dict = dict()
    fit_res_dict['success'] = fit_result.success
    fit_res_dict['message'] = fit_result.message
    fit_res_dict['params'] = {}
    for param_name in fit_result.params:
        fit_res_dict['params'][param_name] = {}
        param = fit_result.params[param_name]
        for k in param.__dict__:
            if k == '_val':
                fit_res_dict['params'][param_name]['value'] = getattr(param, k)
            else:
                if not k.startswith('_') and k not in ['from_internal', ]:
                    fit_res_dict['params'][param_name][k] = getattr(param, k)
    return fit_res_dict


def extend_sweep_points_cal_traces(sweep_points, num_cal_traces=0):
    if num_cal_traces == 0:
        return sweep_points

    if len(sweep_points) > 1:
        dsp = sweep_points[1] - sweep_points[0]
    else:
        dsp = 1
    cal_traces_swpts = np.array([sweep_points[-1] + (i+1)*dsp
                                 for i in range(num_cal_traces)])
    return np.concatenate([sweep_points, cal_traces_swpts])


def extract_and_rotate_data_1d(results, data_handle, cal_trace_handle_root=None,
                               cal_states='ge', do_pca=False):
    # extract data
    swpts = deepcopy(results.get_axis(data_handle)[0])
    if isinstance(swpts, list):
        swpts = swpts[0]
    data_raw = deepcopy(results.get_data(data_handle))
    if cal_trace_handle_root is None:
        cal_trace_handle_root = data_handle
    cal_trace_handles = [e for e in list(results.acquired_results)
                         if f"{cal_trace_handle_root}_cal_trace" in e]
    num_cal_traces = len(cal_trace_handles)
    if num_cal_traces > 0:
        # rotate data to cal states
        raw_data_cal_pt_0 = results.get_data(
            f"{cal_trace_handle_root}_cal_trace_{cal_states[0]}")
        raw_data_cal_pt_1 = results.get_data(
            f"{cal_trace_handle_root}_cal_trace_{cal_states[1]}")
        cal_traces = np.array([raw_data_cal_pt_0, raw_data_cal_pt_1])
        data_raw_w_cal_tr = np.concatenate([data_raw, cal_traces])
        if not do_pca:
            data_rot = cal_tr_rot.rotate_data_to_cal_trace_results(
                data_raw_w_cal_tr, raw_data_cal_pt_0, raw_data_cal_pt_1)
        else:
            # rotate data using pca
            data_rot = cal_tr_rot.principal_component_analysis(data_raw_w_cal_tr)
    else:
        # rotate data using pca
        data_rot = cal_tr_rot.principal_component_analysis(data_raw)
        data_raw_w_cal_tr = data_raw
        cal_traces = np.array([])
    swpts_w_cal_tr = extend_sweep_points_cal_traces(swpts, num_cal_traces)

    data_dict = {
        "sweep_points": swpts,
        "sweep_points_w_cal_traces": swpts_w_cal_tr,
        "sweep_points_cal_traces": swpts_w_cal_tr[len(swpts_w_cal_tr) - num_cal_traces:],
        "data_raw": data_raw,
        "data_raw_w_cal_traces": data_raw_w_cal_tr,
        "data_raw_cal_traces": data_raw_w_cal_tr[len(data_raw_w_cal_tr) - num_cal_traces:],
        "data_rotated": data_rot[:len(data_raw)],
        "data_rotated_w_cal_traces": data_rot,
        "data_rotated_cal_traces": data_rot[len(data_rot) - num_cal_traces:],
        "num_cal_traces": num_cal_traces,
        "do_pca": do_pca,
    }

    return data_dict


def extract_and_rotate_data_2d(results, data_handle, cal_trace_handle_root=None,
                               cal_states='ge', do_pca=False):
    # extract data
    swpts_nt = deepcopy(results.get_axis(data_handle)[0])
    swpts_rt = deepcopy(results.get_axis(data_handle)[1][0])
    data_raw = deepcopy(results.get_data(data_handle))
    if cal_trace_handle_root is None:
        cal_trace_handle_root = data_handle
    cal_trace_handles = [e for e in list(results.acquired_results)
                         if f"{cal_trace_handle_root}_cal_trace" in e]
    num_cal_traces = len(cal_trace_handles)
    data_rot = np.zeros(shape=data_raw.shape)
    if num_cal_traces > 0:
        # rotate data to cal states
        raw_data_cal_pt_0 = results.get_data(f"{cal_trace_handle_root}_cal_trace_{cal_states[0]}")
        raw_data_cal_pt_1 = results.get_data(f"{cal_trace_handle_root}_cal_trace_{cal_states[1]}")
        cal_traces = np.array([raw_data_cal_pt_0, raw_data_cal_pt_1]).T
        data_raw_w_cal_tr = np.concatenate([data_raw, cal_traces], axis=1)
        data_rot = np.zeros(shape=data_raw_w_cal_tr.shape)
        if not do_pca > 0:
            for i in range(data_raw_w_cal_tr.shape[0]):
                data_rot[i, :] = cal_tr_rot.rotate_data_to_cal_trace_results(
                    data_raw_w_cal_tr[i, :], raw_data_cal_pt_0[i], raw_data_cal_pt_1[i])
        else:
            # rotate data using pca
            for i in range(data_raw_w_cal_tr.shape[0]):
                data_rot[i, :] = cal_tr_rot.principal_component_analysis(
                    data_raw_w_cal_tr[i, :])
    else:
        # rotate data using pca
        for i in range(data_raw.shape[0]):
            data_rot[i, :] = cal_tr_rot.principal_component_analysis(
                data_raw[i, :])
        data_raw_w_cal_tr = data_raw

    swpts_rt_w_cal_tr = extend_sweep_points_cal_traces(swpts_rt, num_cal_traces)

    data_dict = {
        "sweep_points": swpts_rt,
        "sweep_points_nt": swpts_nt,
        "sweep_points_w_cal_traces": swpts_rt_w_cal_tr,
        "sweep_points_cal_traces": swpts_rt_w_cal_tr[len(swpts_rt_w_cal_tr) - num_cal_traces:],
        "data_raw": data_raw,
        "data_raw_w_cal_traces": data_raw_w_cal_tr,
        "data_raw_cal_traces": data_raw_w_cal_tr[:, data_raw_w_cal_tr.shape[1] - num_cal_traces:],
        "data_rotated": data_rot[:, :data_raw.shape[1]],
        "data_rotated_w_cal_traces": data_rot,
        "data_rotated_cal_traces": data_rot[:, data_rot.shape[1] - num_cal_traces:],
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
    secant_gradient = ((y[-1] - y[0]) / (x[-1] - x[0]))
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
    mask_func = lambda cos_xvals: np.logical_and(cos_xvals >= min(x),
                                                 cos_xvals <= max(x))
    pixv_top = pi_xvals_top[mask_func(pi_xvals_top)]
    pixv_bottom = pi_xvals_bottom[mask_func(pi_xvals_bottom)]
    pi2xv_rising = pi2_xvals_rising[mask_func(pi2_xvals_rising)]
    pi2xv_falling = pi2_xvals_falling[mask_func(pi2_xvals_falling)]

    return pixv_top, pixv_bottom, pi2xv_rising, pi2xv_falling
