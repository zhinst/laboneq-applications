import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from laboneq.dsl.experiment import pulse_library as pl
from laboneq.analysis import calculate_integration_kernels_thresholds

from laboneq_applications import loading_helpers as load_hlp


def extract_raw_data(folders, preparation_states, file_extension='p'):
    """
    Extract the raw data from the acquired results pickle files in folders.

    Args:
        folders: list of folders that contain the acquired results for the prepared
            states in preparation_states, saved by an OptimalIntegrationKernels
            experiment
        preparation_states: list of strings specifying which states have been prepared
            in the experiment. Example: ["g", "e", "f"]
        file_extension: the extension of the files from which the acquired results
            should be loaded. Defaults to "p". THe other accepted value is "json".

    Returns:
        raw_traces: dict where keys are preparation states, and values are the arrays
            with the corresponding raw traces
        timestamps: list of the timestamps of each folder
    """
    raw_data = {state: [] for state in preparation_states}
    timestamps = []
    for i, folder in enumerate(folders):
        day = folder.split("\\")[-2]
        time = folder.split("\\")[-1].split("_")[0]
        timestamp = f"{day}_{time}"
        timestamps += [timestamp]
        for state in preparation_states:
            acq_results = load_hlp.load_acquired_results_from_experiment_directory(
                folder, filename_to_match=f"acquired_results_{state}",
                file_extension=file_extension)
            handle = list(acq_results)[0]
            raw_data[state] += [acq_results[handle].data]

    raw_traces = {}
    for i, state in enumerate(preparation_states):
        trace = np.mean(raw_data[state], axis=0)
        raw_traces[state] = trace[: (len(trace) // 16) * 16]

    return raw_traces, timestamps


def filter_kernels(kernels, cutoff_frequency, sampling_rate=2e9):
    """
    Applies a low-pass filter to the kernels.

    Args:
        kernels: list with pulse functionals for the integration kernels
        cutoff_frequency: cutoff frequency for the low-pass filter
        sampling_rate: sampling rate of the detection instrument

    Returns:
        a list of the filtered kernels as pulse functionals
    """
    kernels_array_filtered = []
    for w in kernels:
        poles = 5
        sos = sp.signal.butter(poles, cutoff_frequency, 'lowpass',
                               fs=sampling_rate, output='sos')
        kernels_array_filtered += [sp.signal.sosfiltfilt(sos, w.samples)]

    kernels_filtered = [pl.sampled_pulse_complex(w, uid=f'w{i + 1}')
                        for i, w in enumerate(kernels_array_filtered)]

    return kernels_filtered


def plot_traces_kernels(raw_traces, kernels, kernels_filtered=None,
                        plot_title=None, save=False, filename="Traces_Kernels",
                        show=False):
    """
    Plots the raw traces and the integration kernels.

    Args:
        raw_traces: dict where keys are preparation states, and values are the arrays
            with the corresponding raw traces
        kernels: list with pulse functionals for the integration kernels
        kernels_filtered: list with pulse functionals for the filtered integration kernels
        plot_title: the title of the plot
        save: whether to save the plot
        filename: name under which to save the plot; can be a full path
        show: whether to show the figure

    """
    fig_size = plt.rcParams["figure.figsize"]
    fig, axs = plt.subplots(
        nrows=3 + int(kernels_filtered is not None),
        sharex=True,
        figsize=(fig_size[0], fig_size[1] * 2),
    )

    zorders = {'g': 3, 'e': 2, 'f': 1}
    for state, trace in raw_traces.items():
        axs[0].plot(np.real(trace), label=f"{state}: I", zorder=zorders[state])
        axs[1].plot(np.imag(trace), label=f"{state}: Q", zorder=zorders[state])

    axs[0].set_ylabel("Voltage, $V$ (a.u.)")
    axs[0].legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
    axs[1].set_ylabel("Voltage, $V$ (a.u.)")
    axs[1].legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)

    kernels_to_plot = [kernels]
    if kernels_filtered is not None:
        kernels_to_plot += [kernels_filtered]
    for ii, krns in enumerate(kernels_to_plot):
        ax = axs[2 + ii]
        for i, krn in enumerate(krns):
            krn_vals = krn.samples
            ax.plot(np.real(krn_vals),
                    label=f"w{i + 1} {'filtered' if ii == 1 else ''}: I",
                    zorder=len(krns) + 1 - i)
            ax.plot(np.imag(krn_vals),
                    label=f"w{i + 1} {'filtered' if ii == 1 else ''}: Q",
                    zorder=len(krns) + 1 - i)
            ax.set_ylabel("Voltage, $V$ (a.u.)")
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
    axs[-1].set_xlabel("Samples, $N$")
    axs[0].set_title(plot_title)
    fig.subplots_adjust(hspace=0.1)
    fig.align_ylabels()
    if save:
        fig.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()


def plot_kernels_fft(kernels, kernels_filtered=None, cutoff_frequency=None,
                     plot_title=None, save=False, filename="Kernels_FFT", show=False):
    """
    Plots the FFT of the integration kernels.

    Args:
        kernels: list with pulse functionals for the integration kernels
        kernels_filtered: list with pulse functionals for the filtered integration kernels
        cutoff_frequency: cutoff frequency for the low-pass filter; in order to show it
            on the plot
        plot_title: the title of the plot
        save: whether to save the plot
        filename: name under which to save the plot; can be a full path
        show: whether to show the figure

    """

    fig, axs = plt.subplots(nrows=len(kernels), sharex=True)

    for i, ax in enumerate(axs):
        y = kernels[i].samples
        # y = kernels_manual.samples
        # y = kernels_legacy_w_twpa.samples
        N = len(y)
        T = 0.5e-9
        yf = sp.fft.fft(y)
        xf = sp.fft.fftfreq(N, T)
        xf = sp.fft.fftshift(xf)
        yplot = sp.fft.fftshift(yf)
        ax.semilogy(xf / 1e6, 1.0 / N * np.abs(yplot), label=f"w{i + 1}: unfiltered")

        if kernels_filtered is not None:
            assert cutoff_frequency is not None
            y = kernels_filtered[i].samples
            N = len(y)
            T = 0.5e-9
            yf = sp.fft.fft(y)
            xf = sp.fft.fftfreq(N, T)
            xf = sp.fft.fftshift(xf)
            yplot = sp.fft.fftshift(yf)
            ax.semilogy(xf / 1e6, 1.0 / N * np.abs(yplot),
                        label=f"LPF: {cutoff_frequency / 1e6} MHz")

    axs[-1].set_xlabel("Readout IF Frequency, $f_{IF}$ (MHz)")
    axs[0].set_ylabel("FFT")
    axs[-1].set_ylabel("FFT")
    axs[0].legend(frameon=False)
    axs[1].legend(frameon=False)
    axs[0].set_title(plot_title)
    fig.subplots_adjust(hspace=0.1)
    if save:
        plt.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()


def optimal_kernels_analysis(qubit_name, folders, states, cutoff_frequency=None,
                             save=False, save_directory=None, show=False):
    """

    Args:
        qubit_name:
        folders:
        states:
        cutoff_frequency:
        save:
        save_directory:
        show:

    Returns: parameters to be set to the qubit
        - a list of the integration kernels as pulse functionals: to be set to
            qubit.parameters.readout_integration_kernels
        - a list of threshold values: to be set to
            qubit.parameters.readout_discrimination_thresholds

    """

    raw_traces_per_state, timestamps = extract_raw_data(folders, states)

    plot_title = f"Optimal Integration Kernels {qubit_name}\n{timestamps[0]} - {timestamps[-1]}"
    if save_directory is None:
        save_directory = folders[-1]

    raw_traces = list(raw_traces_per_state.values())
    kernels, thresholds = calculate_integration_kernels_thresholds(raw_traces)
    for i, kern in enumerate(kernels):
        kern.uid = f'{qubit_name}_w{i + 1}'

    kernels_filtered = None
    if cutoff_frequency is not None:
        kernels_filtered = filter_kernels(kernels, cutoff_frequency=cutoff_frequency)
        for i, kern in enumerate(kernels_filtered):
            kern.uid = f'{qubit_name}_w{i + 1}'

    filename = save_directory + f"\\Averaged_Kernels_{timestamps[0]}_{timestamps[-1]}.png"
    plot_traces_kernels(raw_traces_per_state, kernels, kernels_filtered, plot_title,
                        save=save, filename=filename, show=show)
    filename = folders[-1] + f"\\FFT_Weights_{timestamps[0]}_{timestamps[-1]}.png"
    plot_kernels_fft(kernels, kernels_filtered=kernels_filtered,
                     cutoff_frequency=cutoff_frequency,
                     plot_title=plot_title, save=save,
                     filename=filename, show=show)

    kernels_to_return = kernels_filtered if kernels_filtered is not None else kernels
    return kernels_to_return, thresholds
