###
# Common helper functions for fitting data - Needed to extract qubit and readout parameters from measurement data
###

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.optimize as opt
from laboneq.simple import *  # noqa: F403
from scipy.optimize import curve_fit


# convenience
def flatten(l):
    # flatten a nested list to a single level
    return [item for sublist in l for item in sublist]


# oscillations - Rabi
def func_osc(x, freq, phase, amp=1, off=0):
    return amp * np.cos(freq * x + phase) + off


# decaying oscillations - Ramsey
def func_decayOsc(x, freq, phase, rate, amp=1, off=-0.5):
    return amp * np.cos(freq * x + phase) * np.exp(-rate * x) + off


# decaying exponent - T1
def func_exp(x, rate, off, amp=1):
    return amp * np.exp(-rate * x) + off


# Lorentzian
def func_lorentz(x, width, pos, amp, off):
    return off + amp * width / (width**2 + (x - pos) ** 2)


# inverted Lorentzian - spectroscopy
def func_invLorentz(x, width, pos, amp, off=1):
    return off - amp * width / (width**2 + (x - pos) ** 2)


# Fano lineshape - spectroscopy
def func_Fano(x, width, pos, amp, fano=0, off=0.5):
    return off + amp * (fano * width + x - pos) ** 2 / (width**2 + (x - pos) ** 2)


## function to fit Rabi oscillations
def fit_Rabi(x, y, freq, phase, amp=None, off=None, plot=False, bounds=None):
    if amp is not None:
        if off is not None:
            if bounds is None:
                popt, pcov = opt.curve_fit(func_osc, x, y, p0=[freq, phase, amp, off])
            else:
                popt, pcov = opt.curve_fit(
                    func_osc, x, y, p0=[freq, phase, amp, off], bounds=bounds
                )
        else:
            if bounds is None:
                popt, pcov = opt.curve_fit(func_osc, x, y, p0=[freq, phase, amp])
            else:
                popt, pcov = opt.curve_fit(
                    func_osc, x, y, p0=[freq, phase, amp], bounds=bounds
                )
    else:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_osc, x, y, p0=[freq, phase])
        else:
            popt, pcov = opt.curve_fit(func_osc, x, y, p0=[freq, phase], bounds=bounds)

    if plot:
        plt.plot(x, y, ".k")
        plt.plot(x, func_osc(x, *popt), "-r")
        plt.show()

    return popt, pcov


## function to fit Ramsey oscillations
def fit_Ramsey(x, y, freq, phase, rate, amp=None, off=None, plot=False, bounds=None):
    if amp is not None:
        if off is not None:
            if bounds is None:
                popt, pcov = opt.curve_fit(
                    func_decayOsc, x, y, p0=[freq, phase, rate, amp, off]
                )
            else:
                popt, pcov = opt.curve_fit(
                    func_decayOsc, x, y, p0=[freq, phase, rate, amp, off], bounds=bounds
                )
        else:
            if bounds is None:
                popt, pcov = opt.curve_fit(
                    func_decayOsc, x, y, p0=[freq, phase, rate, amp]
                )
            else:
                popt, pcov = opt.curve_fit(
                    func_decayOsc, x, y, p0=[freq, phase, rate, amp], bounds=bounds
                )
    else:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_decayOsc, x, y, p0=[freq, phase, rate])
        else:
            popt, pcov = opt.curve_fit(
                func_decayOsc, x, y, p0=[freq, phase, rate], bounds=bounds
            )

    if plot:
        plt.plot(x, y, ".k")
        plt.plot(x, func_decayOsc(x, *popt), "-r")
        plt.show()

    return popt, pcov


## function to fit T1 decay
def fit_T1(x, y, rate, off, amp=None, plot=False, bounds=None):
    if bounds is None:
        if amp is None:
            popt, pcov = opt.curve_fit(func_exp, x, y, p0=[rate, off])
        else:
            popt, pcov = opt.curve_fit(func_exp, x, y, p0=[rate, off, amp])
    else:
        if amp is None:
            popt, pcov = opt.curve_fit(func_exp, x, y, p0=[rate, off], bounds=bounds)
        else:
            popt, pcov = opt.curve_fit(
                func_exp, x, y, p0=[rate, off, amp], bounds=bounds
            )

    if plot:
        plt.plot(x, y, ".k")
        plt.plot(x, func_exp(x, *popt), "-r")
        plt.show()

    return popt, pcov


## function to fit spectroscopy traces
def fit_Spec(x, y, width, pos, amp, off=None, plot=False, bounds=None):
    np.median(y)
    y = y - np.median(y) + 1
    if off is not None:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_invLorentz, x, y, p0=[width, pos, amp, off])
        else:
            popt, pcov = opt.curve_fit(
                func_invLorentz, x, y, p0=[width, pos, amp, off], bounds=bounds
            )
    else:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_invLorentz, x, y, p0=[width, pos, amp])
        else:
            popt, pcov = opt.curve_fit(
                func_invLorentz, x, y, p0=[width, pos, amp], bounds=bounds
            )

    if plot:
        plt.plot(x, y, ".k")
        plt.plot(x, func_invLorentz(x, *popt), "-r")
        plt.show()

    return popt, pcov


## function to fit 3D cavity spectroscopy traces
def fit_3DSpec(x, y, width, pos, amp, off=None, plot=False, bounds=None):
    if off is not None:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_lorentz, x, y, p0=[width, pos, amp, off])
        else:
            popt, pcov = opt.curve_fit(
                func_lorentz, x, y, p0=[width, pos, amp, off], bounds=bounds
            )
    else:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_lorentz, x, y, p0=[width, pos, amp])
        else:
            popt, pcov = opt.curve_fit(
                func_lorentz, x, y, p0=[width, pos, amp], bounds=bounds
            )

    if plot:
        plt.plot(x, y, ".k")
        plt.plot(x, func_lorentz(x, *popt), "-r")
        plt.show()

    return popt, pcov


## function to fit spectroscopy traces with Fano lineshape
def fit_ResSpec(x, y, width, pos, amp, fano, off=None, plot=False, bounds=None):
    if off is not None:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_Fano, x, y, p0=[width, pos, amp, fano, off])
        else:
            popt, pcov = opt.curve_fit(
                func_Fano, x, y, p0=[width, pos, amp, fano, off], bounds=bounds
            )
    else:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_Fano, x, y, p0=[width, pos, amp, fano])
        else:
            popt, pcov = opt.curve_fit(
                func_Fano, x, y, p0=[width, pos, amp, fano], bounds=bounds
            )

    if plot:
        plt.plot(x, y, ".k")
        plt.plot(x, func_Fano(x, *popt), "-r")
        plt.show()

    return popt, pcov


# PSI specific analysis
def rotate_to_real_axis(complex_values):
    # find angle
    slope = np.polyfit(np.real(complex_values), np.imag(complex_values), 1)[0]
    angle = np.arctan(slope)

    res_values = complex_values * np.exp(2 * np.pi * 1j * angle)

    # rotate
    return res_values


def analyze_qspec(
    res, handle, f0=1e9, a=0.01, gamma=3e6, offset=0.04, rotate=False, flip=False
):
    qspec_res = res.get_data(handle)
    qspec_freq = res.get_axis(handle)[0]

    y = np.abs(qspec_res) if not rotate else np.real(rotate_to_real_axis(qspec_res))

    flip_sign = -1 if flip else +1

    def lorentzian(f, f0, a, gamma, offset, flip_sign):
        penalization = abs(min(0, gamma)) * 1000
        return offset + flip_sign * a / (1 + (f - f0) ** 2 / gamma**2) + penalization

    (f_0, a, gamma, offset, flip_sign), _ = curve_fit(
        lorentzian, qspec_freq, y, (f0, a, gamma, offset, flip_sign)
    )

    y_fit = lorentzian(qspec_freq, f_0, a, gamma, offset, flip_sign)

    plt.figure()
    plt.plot(qspec_freq, y)
    plt.plot(qspec_freq, abs(qspec_res), ".")
    plt.plot(qspec_freq, y_fit)
    plt.show()

    return f_0


def create_x90(qubit_parameters, qubit):
    return pulse_library.drag(
        uid=f"gaussian_x90_q{qubit}",
        length=qubit_parameters[qubit]["qb_len"],
        beta=qubit_parameters[qubit]["q_scale"],
        amplitude=qubit_parameters[qubit]["pi_amp"] / 2,
    )


def create_x180(qubit_parameters, qubit):
    return pulse_library.drag(
        uid=f"gaussian_x180_q{qubit}",
        length=qubit_parameters[qubit]["qb_len"],
        beta=qubit_parameters[qubit]["q_scale"],
        amplitude=qubit_parameters[qubit]["pi_amp"],
    )


def create_x180_ef(qubit_parameters, qubit):
    return pulse_library.gaussian(
        uid=f"gaussian_x180_ef_q{qubit}",
        length=qubit_parameters[qubit]["ef_qb_len"],
        beta=qubit_parameters[qubit]["q_scale_ef"],
        amplitude=qubit_parameters[qubit]["ef_pi_amp"],
    )


def create_x90_ef(qubit_parameters, qubit):
    return pulse_library.gaussian(
        uid=f"gaussian_x90_ef_q{qubit}",
        length=qubit_parameters[qubit]["ef_qb_len"],
        beta=qubit_parameters[qubit]["q_scale_ef"],
        amplitude=qubit_parameters[qubit]["ef_pi2_amp"],
    )


def evaluate_rabi(res, handle, plot=True, rotate=False, flip=False, real=False):
    def rabi_curve(x, offset, phase_shift, amplitude, period):
        return amplitude * np.sin(np.pi / period * x - phase_shift) + offset

    #  return amplitude*np.sin(2*np.pi/period*x+np.pi/2)+offset

    x = res.get_axis(handle)[0]
    if rotate:
        y = np.real(rotate_to_real_axis(res.get_data(handle)))
    elif real:
        y = np.real(res.get_data(handle))
    else:
        y = np.abs(res.get_data(handle))

    if flip:
        y = -y

    plt.scatter(x, y)
    plt.show()

    offset_guess = np.mean(y)
    phase_shift_guess = np.pi / 2
    amplitude_guess = (max(y) - min(y)) / 2
    period_guess = abs(x[np.argmax(y)] - x[np.argmin(y)])
    p0 = [offset_guess, phase_shift_guess, amplitude_guess, period_guess]
    print(p0)
    popt = scipy.optimize.curve_fit(rabi_curve, x, y, p0=p0)[0]

    pi_amp = scipy.optimize.fmin(
        lambda x: -rabi_curve(x, *popt), x[np.argmax(y)], disp=False
    )[0]

    pi2_amp = scipy.optimize.fmin(
        lambda x: abs(rabi_curve(x, *popt) - popt[0]), pi_amp / 2, disp=False
    )[0]

    if plot:
        plt.figure()
        plt.plot(x, rabi_curve(x, *popt))
        plt.plot(x, y, ".")
        plt.plot([pi_amp, pi_amp], [min(y), rabi_curve(pi_amp, *popt)])
        plt.plot([pi2_amp, pi2_amp], [min(y), rabi_curve(pi2_amp, *popt)])
    print(popt)
    print(f"Pi amp: {pi_amp}, pi/2 amp: {pi2_amp}")
    return [pi_amp, pi2_amp]


def evaluate_ramsey(res, handle, plot=True, rotate=False, flip=False, use_phase=False):
    def ramsey_curve(x, offset, phase_shift, amplitude, period, t2):
        #   return amplitude*np.sin(2*np.pi*period*x-phase_shift)+offset
        return (
            amplitude * np.exp(-x / t2) * np.sin(2 * np.pi * x / period + phase_shift)
            + offset
        )

    x = res.get_axis(handle)[0]
    if use_phase:
        y = np.angle(res.get_data(handle))
    else:
        y = (
            np.real(rotate_to_real_axis(res.get_data(handle)))
            if rotate
            else np.abs(res.get_data(handle))
        )
    if flip:
        y = -y

    offset_guess = np.mean(y)
    phase_shift_guess = np.pi / 2 if y[0] > y[-1] else -np.pi / 2
    amplitude_guess = (max(y) - min(y)) / 2
    period_guess = 2 * abs(x[np.argmax(y)] - x[np.argmin(y)])
    t2_guess = 10e-6
    # TODO: Refine t2 guess algorithm, potentially by finding peaks and fitting only them

    p0 = [offset_guess, phase_shift_guess, amplitude_guess, period_guess, t2_guess]
    print(p0)

    popt = scipy.optimize.curve_fit(ramsey_curve, x, y, p0=p0)[0]

    t2 = popt[4]
    detuning_freq = 1 / popt[3]

    envelope_param = np.copy(popt)
    envelope_param[3] = 1e9
    envelope_param[1] = np.pi / 2

    if plot:
        plt.figure()
        plt.plot(x, ramsey_curve(x, *popt))
        plt.plot(x, y, ".")
        plt.plot(x, ramsey_curve(x, *envelope_param))
    #    plt.plot([pi_amp, pi_amp], [min(y), ramsey_curve(pi_amp, *popt)])
    #    plt.plot([pi2_amp, pi2_amp], [min(y), ramsey_curve(pi2_amp, *popt)])

    # print(f"Pi amp: {pi_amp}, pi/2 amp: {pi2_amp}")
    print(f"Detuned by {detuning_freq/1e6} MHz; T2 found to be {t2*1e6} us.")
    return [t2, detuning_freq]


# def evaluate_T1(res, handle, plot=True, rotate=False):
#     def T1_curve(x, offset, amplitude, t1):
#         return amplitude * np.exp(-x / t1) + offset

#     x = res.get_axis(handle)[0]

#     y = (
#         np.abs(res.get_data(handle))
#         if not rotate
#         else -np.real(rotate_to_real_axis(res.get_data(handle)))
#     )

#     offset_guess = min(y)
#     amplitude_guess = max(y)
#     t1_guess = 20e-6

#     p0 = [offset_guess, amplitude_guess, t1_guess]

#     popt = scipy.optimize.curve_fit(T1_curve, x, y, p0=p0)[0]

#     t1 = popt[2]

#     # pi_amp = scipy.optimize.fmin(lambda x: -rabi_curve(x, *popt), x[np.argmax(y)], disp=False)[0]

#     #  pi2_amp = scipy.optimize.fmin(lambda x: abs(rabi_curve(x, *popt)-popt[0]), pi_amp/2, disp=False )[0]

#     if plot:
#         plt.figure()
#         plt.plot(x, T1_curve(x, *popt))
#         plt.plot(x, y, ".")
#     #  plt.plot([pi_amp, pi_amp], [min(y), rabi_curve(pi_amp, *popt)])
#     #  plt.plot([pi2_amp, pi2_amp], [min(y), rabi_curve(pi2_amp, *popt)])

#     print(f"T1 found to be {t1*1e6} us.")
#     return t1


def evaluate_T1(res, handle, plot=True, rotate=False):
    def T1_curve(x, offset, amplitude, t1):
        return amplitude * np.exp(-x / t1) + offset

    x = res.get_axis(handle)[0]

    y = (
        np.abs(res.get_data(handle))
        if not rotate
        else -np.real(rotate_to_real_axis(res.get_data(handle)))
    )

    offset_guess = min(y)
    amplitude_guess = max(y)
    t1_guess = 20e-6

    p0 = [offset_guess, amplitude_guess, t1_guess]

    popt = scipy.optimize.curve_fit(T1_curve, x, y, p0=p0)[0]

    t1 = popt[2]

    # pi_amp = scipy.optimize.fmin(lambda x: -rabi_curve(x, *popt), x[np.argmax(y)], disp=False)[0]

    #  pi2_amp = scipy.optimize.fmin(lambda x: abs(rabi_curve(x, *popt)-popt[0]), pi_amp/2, disp=False )[0]

    if plot:
        plt.figure()
        plt.plot(x * 1e6, T1_curve(x, *popt))
        plt.plot(x * 1e6, y, ".")
        plt.xlabel("delay (us)")
        plt.ylabel("Amplitude (a.u.)")
        plt.axvline(x=t1 * 1e6, color="gray", linestyle="--", linewidth=2)
        plt.text(x=2 * t1 * 1e6, y=max(y) / 2, s=f"T1= {t1*1e6:.3f}us.")
    #  plt.plot([pi_amp, pi_amp], [min(y), rabi_curve(pi_amp, *popt)])
    #  plt.plot([pi2_amp, pi2_amp], [min(y), rabi_curve(pi2_amp, *popt)])

    print(f"T1 found to be {t1*1e6:.3f} us.")
    return t1


def calc_readout_weight(res_0, res_1, handle, plot=True):
    raw_0 = res_0.get_data(handle)
    raw_1 = res_1.get_data(handle)

    readout_weight = np.conj(raw_1 - raw_0)
    readout_weight = readout_weight / max(np.abs(readout_weight))

    if plot:
        plt.scatter(np.real(raw_0), np.imag(raw_0))
        plt.scatter(np.real(raw_1), np.imag(raw_1))
        plt.show()

    return readout_weight


def analyze_ACStark(qubit_parameters, results, handle, qubit, plot=True):
    spec_freq = qubit_parameters[qubit]["f0g1_lo"] + results.get_axis("f0g1")[0]
    amp = results.get_axis(handle)[1][0]
    data = np.transpose(results.get_data(handle))

    res_freq = amp.copy()
    for i, line in enumerate(data):
        res_freq[i] = spec_freq[
            np.argmin(line)
        ]  # for now, take argmin instead of Gauss fit
        if plot:
            plt.plot(spec_freq, line)
    if plot:
        plt.show()

    def parabola(x, a, b, c):
        return a * x * x + b * x + c

    # offset - a*max_amp**2 == res_freq[-1]

    a_guess = (res_freq[-1] - max(res_freq)) / amp[-1] ** 2
    b_guess = 0  # np.pi/2
    c_guess = max(res_freq)

    p0 = [a_guess, b_guess, c_guess]
    print(p0)
    popt = scipy.optimize.curve_fit(parabola, amp, res_freq, p0=p0)[0]
    print(popt)

    if plot:
        plt.plot(amp, res_freq / 1e9, ".")
        plt.plot(amp, parabola(amp, *popt) / 1e9)
        plt.xlabel("Amplitude [a.u.]")
        plt.ylabel("Frequency [GHz]")
        plt.grid()
        plt.show()

    return popt


# plotting helpers
def plot_with_trace_rabi(res):
    handles = list(res.acquired_results.keys())
    res1 = np.asarray(res.get_data(handles[0]))
    res_cal_trace = np.asarray(res.get_data(handles[1]))
    axis1 = res.get_axis(handles[0])[0]
    delta_x = axis1[-1] - axis1[-2]
    axis2 = np.linspace(axis1[-1] + delta_x, axis1[-1] + 2 * delta_x, 2)

    delta_vec = res_cal_trace[1] - res_cal_trace[0]
    angle = np.angle(delta_vec)
    rd = []
    for r in [res1, res_cal_trace]:
        r = r - res_cal_trace[0]
        r = r * np.exp(-1j * angle)
        r = r / np.abs(delta_vec)
        rd.append(r)

    pi_amp = axis1[np.argmax(np.real(rd[0]))]

    plt.xlabel(handles[0])
    plt.ylabel("|e> population")
    plt.plot(axis1, np.real(rd[0]), "o")
    plt.plot(axis2[0], np.real(rd[1][0]), "o", color="gray")
    plt.plot(axis2[0], np.real(rd[1][1]), "o", color="black")
    plt.axhline(y=np.real(rd[1][0]), color="gray", linestyle="--", linewidth=2)
    plt.axhline(y=np.real(rd[1][1]), color="gray", linestyle="--", linewidth=2)
    plt.axvline(x=pi_amp, color="gray", linestyle="--", linewidth=2)
    plt.text(x=pi_amp, y=max(np.real(rd[0])) / 2, s=f"piamp:{pi_amp}")
    plt.plot()


# havenot tested yet
def plot_with_trace_ramsey(res):
    handles = list(res.acquired_results.keys())
    res1 = np.asarray(res.get_data(handles[0]))
    res_cal_trace = np.asarray(res.get_data(handles[1]))
    axis1 = res.get_axis(handles[0])[0]
    delta_x = axis1[-1] - axis1[-2]
    axis2 = np.linspace(axis1[-1] + delta_x, axis1[-1] + 2 * delta_x, 2)

    delta_vec = res_cal_trace[1] - res_cal_trace[0]
    angle = np.angle(delta_vec)
    rd = []
    for r in [res1, res_cal_trace]:
        r = r - res_cal_trace[0]
        r = r * np.exp(-1j * angle)
        r = r / np.abs(delta_vec)
        rd.append(r)

    plt.xlabel(handles[0])
    plt.ylabel("|e> population")
    plt.plot(axis1, np.real(rd[0]), "o")
    plt.plot(axis2[0], np.real(rd[1][0]), "o", color="gray")
    plt.plot(axis2[0], np.real(rd[1][1]), "o", color="black")
    plt.axhline(y=np.real(rd[1][0]), color="gray", linestyle="--", linewidth=2)
    plt.axhline(y=np.real(rd[1][1]), color="gray", linestyle="--", linewidth=2)
    # plt.axvline(x=pi_amp, color='gray', linestyle='--', linewidth=2)
    # plt.text(x=pi_amp,y = max(np.real(rd[0]))/2, s= f"piamp:{pi_amp}")
    plt.plot()


def calculate_fidelity(res0_rot, res1_rot, threshold):
    prepared_g_measured_g = np.count_nonzero(res0_rot.real < threshold) / len(
        res0_rot.real
    )
    prepared_g_measured_e = np.count_nonzero(res0_rot.real > threshold) / len(
        res0_rot.real
    )
    prepared_e_measured_g = np.count_nonzero(res1_rot.real < threshold) / len(
        res1_rot.real
    )
    prepared_e_measured_e = np.count_nonzero(res1_rot.real > threshold) / len(
        res1_rot.real
    )

    fidelity = 1 - prepared_g_measured_e - prepared_e_measured_g
    print(f"ee:{prepared_e_measured_e}")
    print(f"gg:{prepared_g_measured_g}")
    print(f"eg:{prepared_e_measured_g}")
    print(f"ge:{prepared_g_measured_e}")
    print(f"Fidelity {fidelity}")

    return fidelity
