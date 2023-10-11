import matplotlib.pyplot as plt
import numpy as np


def plot_resonator_spectroscopy_full_range(results):
    results_x = results.acquired_results.to_xarray()

    plt.figure()
    for i in range(results_x.sizes["sweep_0"]):
        plt.plot(
            results_x.data_vars["resonator_spectroscopy"][i]["sweep_0"]
            + results_x.data_vars["resonator_spectroscopy"][i]["sweep_1"],
            20 * np.log10(np.abs(results_x.data_vars["resonator_spectroscopy"][i])),
        )
    plt.grid()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Transmission [dB]")
    plt.show()
