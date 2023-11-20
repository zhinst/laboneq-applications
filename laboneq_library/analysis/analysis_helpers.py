import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

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


def fit_data_lmfit(function, x, y, param_hints):
    import lmfit
    model = lmfit.Model(function)
    model.param_hints = param_hints
    return model.fit(x=x, data=y, params=model.make_params())


def flatten_lmfit_modelresult(fit_result):
    import lmfit
    # used for saving an lmfit ModelResults object as a dict
    assert type(fit_result) is lmfit.model.ModelResult
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


def extract_and_rotate_data_1d(results, handle, cal_states='ge'):
    # extract data
    swpts = deepcopy(results.get_axis(handle)[0])
    if isinstance(swpts, list):
        swpts = swpts[0]
    data_raw = deepcopy(results.get_data(handle))
    cal_trace_handles = [e for e in list(results.acquired_results)
                         if f"{handle}_cal_trace" in e]
    num_cal_traces = len(cal_trace_handles)
    if num_cal_traces > 0:
        # rotate data to cal states
        raw_data_cal_pt_0 = results.get_data(f"{handle}_cal_trace_{cal_states[0]}")
        raw_data_cal_pt_1 = results.get_data(f"{handle}_cal_trace_{cal_states[1]}")
        cal_traces = np.array([raw_data_cal_pt_0, raw_data_cal_pt_1])
        data_raw_w_cal_tr = np.concatenate([data_raw, cal_traces])
        data_rot = cal_tr_rot.rotate_data_to_cal_trace_results(
            data_raw_w_cal_tr, raw_data_cal_pt_0, raw_data_cal_pt_1)
    else:
        # rotate data using pca
        data_raw_w_cal_tr = data_raw
        data_rot = cal_tr_rot.principal_component_analysis(data_raw_w_cal_tr)

    swpts_w_cal_tr = extend_sweep_points_cal_traces(swpts, num_cal_traces)

    data_dict = {
        "sweep_points": swpts,
        "data_raw": data_raw,
        "sweep_points_w_cal_tr": swpts_w_cal_tr,
        "data_raw_w_cal_tr": data_raw_w_cal_tr,
        "data_rotated": data_rot[:len(data_raw)],
        "data_rotated_w_cal_tr": data_rot,
        "num_cal_traces": num_cal_traces
    }

    return data_dict

