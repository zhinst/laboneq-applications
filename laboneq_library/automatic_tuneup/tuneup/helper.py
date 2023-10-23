###
# Common helper functions for fitting data - Needed to extract qubit and readout parameters from measurement data
###

import json
import pathlib
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import scipy
import scipy.optimize as opt
from laboneq.simple import *  # noqa: F403

from .configs import get_config

config = get_config()


def get_device_setup_sherloq():
    sherloq_descriptor = """\
      instruments:
        SHFQC:
        - address: dev12250
          uid: shfqc_psi
      connections:
        shfqc_psi:
          - iq_signal: q0/drive_line
            ports: SGCHANNELS/0/OUTPUT
          - iq_signal: q0/measure_line
            ports: [QACHANNELS/0/OUTPUT]
          - acquire_signal: q0/acquire_line
            ports: [QACHANNELS/0/INPUT]
          - iq_signal: q1/drive_line
            ports: SGCHANNELS/1/OUTPUT
          - iq_signal: q1/measure_line
            ports: [QACHANNELS/0/OUTPUT]
          - acquire_signal: q1/acquire_line
            ports: [QACHANNELS/0/INPUT]
    """
    device_setup = DeviceSetup.from_descriptor(
        sherloq_descriptor,
        server_host="localhost",
        server_port=8004,
        setup_name="sherloq",
    )
    return device_setup


def get_device_setup_qzilla_shfqc():
    sherloq_descriptor = """\
      instruments:
        SHFQC:
        - address: dev12144
          uid: shfqc_psi
      connections:
        shfqc_psi:
          - iq_signal: q0/drive_line
            ports: SGCHANNELS/0/OUTPUT
          - iq_signal: q0/measure_line
            ports: [QACHANNELS/0/OUTPUT]
          - acquire_signal: q0/acquire_line
            ports: [QACHANNELS/0/INPUT]

          - iq_signal: q1/drive_line
            ports: SGCHANNELS/1/OUTPUT
          - iq_signal: q1/measure_line
            ports: [QACHANNELS/0/OUTPUT]
          - acquire_signal: q1/acquire_line
            ports: [QACHANNELS/0/INPUT]
    """
    device_setup = DeviceSetup.from_descriptor(
        sherloq_descriptor,
        server_host="zireg-srv-lin05.zhinst.com",
        server_port=8004,
        setup_name="sherloq",
    )
    return device_setup


def set_dc_bias(session, qubit_uid, voltage):
    slot = qubit_uid + 2
    print(f"Setting slot {slot} voltage to {voltage}")
    try:
        requests.get(f"http://127.0.0.1:5000/setvoltage/{slot}/{voltage}/")
    except Exception as e:
        print(f"Error: {e}")
        return False


def save_qubits(qubits):
    if not isinstance(qubits, list):
        qubits = [qubits]
    save_folder = Path(config.get("Settings", "qubit_save_dir"))
    save_folder.mkdir(exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    for q in qubits:
        filename = f"{q.uid}_{current_time}"
        file_path = save_folder / filename
        q.to_json(file_path)


def find_latest_file_with_uid(folder_path, uid):
    file_pattern = f"{uid}_*"

    folder = Path(folder_path)
    files = list(folder.glob(f"{file_pattern}"))
    if not files:
        return None

    latest_file = max(files, key=lambda f: f.stat().st_ctime)
    return latest_file


def load_latest_qubit(uid):
    res = find_latest_file_with_uid("tuneup_data", uid)
    return QuantumElement.load(res)


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


def fit_rabi(x, y, freq, phase, amp=None, off=None, plot=False, bounds=None):
    """
    Function to fit Rabi oscillations, adapted from L1Q contrib.
    """
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


def qubit_parameters(filename="calib.json"):
    calib_file = open(f"{filename}").read()
    qubit_parameters = json.loads(calib_file)
    return qubit_parameters


def save_results(res, result_path="Results"):
    exp_id = res.compiled_experiment.experiment.uid
    timestamp = str(time.strftime("%Y-%m-%d_%H%M%S"))

    filename = result_path + "/" + timestamp + "_" + exp_id + ".json"

    pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)

    res.save(filename)


def save_qubit_parameters(qubit_parameters, history_path="calib_history"):
    calib_json = json.dumps(qubit_parameters, indent=4)
    with open("calib.json", "w") as f:
        f.write(
            calib_json,
        )

    pathlib.Path(history_path).mkdir(parents=True, exist_ok=True)

    timestamp = str(time.strftime("%Y-%m-%d_%H%M%S"))
    history_filename = history_path + "/" + timestamp + "calib.json"
    with open(history_filename, "w") as f:
        f.write(calib_json)


def rotate_to_real_axis(complex_values):
    # find angle
    slope = np.polyfit(np.real(complex_values), np.imag(complex_values), 1)[0]
    angle = np.arctan(slope)

    res_values = complex_values * np.exp(2 * np.pi * 1j * angle)

    # rotate
    return res_values


def analyze_qspec(res, handle, window_len=41, rotate=False, flip=False):
    qspec_res = res.get_data(handle)
    qspec_freq = res.get_axis(handle)[0]

    y = np.abs(qspec_res) if not rotate else np.real(rotate_to_real_axis(qspec_res))
    y = -y if flip else y

    window = np.hanning(window_len)
    y = np.convolve(window / window.sum(), y)
    y = y[int((window_len - 1) / 2) : len(y) - int((window_len - 1) / 2)]

    res_freq = qspec_freq[np.argmax(y)]

    plt.figure()
    plt.plot(qspec_freq, y)
    plt.plot(qspec_freq, abs(qspec_res), ".")
    plt.plot([res_freq, res_freq], [min(y), max(y)])
    plt.show()

    return res_freq


def create_x90(qubit):
    return pulse_library.drag(
        uid=f"gaussian_x90_q{qubit}",
        length=qubit_parameters[qubit]["qb_len"],
        beta=qubit_parameters[qubit]["q_scale"],
        amplitude=qubit_parameters[qubit]["pi_amp"] / 2,
    )


def create_x180(qubit):
    return pulse_library.drag(
        uid=f"gaussian_x180_q{qubit}",
        length=qubit_parameters[qubit]["qb_len"],
        beta=qubit_parameters[qubit]["q_scale"],
        amplitude=qubit_parameters[qubit]["pi_amp"],
    )


def create_x180_ef(qubit):
    return pulse_library.gaussian(
        uid=f"gaussian_x180_ef_q{qubit}",
        length=qubit_parameters[qubit]["ef_qb_len"],
        beta=qubit_parameters[qubit]["q_scale_ef"],
        amplitude=qubit_parameters[qubit]["ef_pi_amp"],
    )


def create_x90_ef(qubit):
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
        plt.plot(x, T1_curve(x, *popt))
        plt.plot(x, y, ".")
    #  plt.plot([pi_amp, pi_amp], [min(y), rabi_curve(pi_amp, *popt)])
    #  plt.plot([pi2_amp, pi2_amp], [min(y), rabi_curve(pi2_amp, *popt)])

    print(f"T1 found to be {t1*1e6} us.")
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


def analyze_ACStark(res, handle, qubit, plot=True):
    spec_freq = qubit_parameters[qubit]["f0g1_lo"] + res.get_axis("f0g1")[0]
    amp = res.get_axis(handle)[1][0]
    data = np.transpose(res.get_data(handle))

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
