import matplotlib.pyplot as plt
import numpy as np


def find_oscillation_frequency(data, time):
    w = np.fft.fft(data)
    f = np.fft.fftfreq(len(data), time[1] - time[0])
    mask = f > 0
    w, f = w[mask], f[mask]
    abs_w = np.abs(w)
    return f[np.argmax(abs_w)]


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

