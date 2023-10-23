import math
import random
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from laboneq.simple import Results
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from .helper import rotate_to_real_axis


class Analyzer(ABC):
    """Base class for all analyzers.
        Analyzers are used to analyze the results of a measurement and return a value.
        Analyzers can also be used to verify if a measurement is successful or not.
        Analyzers can be used to plot the results of a measurement.
    Args:
        truth (float, optional): The expected value of the measurement. Defaults to 0.
        tolerance (float, optional): The tolerance of the measurement. Defaults to 0.
    """

    def __init__(self, truth: float = 0, tolerance: float = 0) -> None:
        self.truth = truth
        self.tolerance = tolerance

    @abstractmethod
    def analyze(self, result: Results, **kwargs) -> float:
        pass

    @abstractmethod
    def verify(self, result: float) -> bool:
        pass

    def plot(self) -> None:
        raise NotImplementedError("Plotting of fitting values not implemented")


class DefaultAnalyzer(Analyzer):
    def analyze(self, result: Results, **kwargs) -> float:
        return 1234

    def verify(self, result: float) -> bool:
        return True


class RandomAnalyzer(DefaultAnalyzer):
    def __init__(self, truth=None, tolerance=0) -> None:
        super().__init__(truth=truth, tolerance=tolerance)

    def analyze(self, result: Results, **kwargs):
        return 6.5e9

    def verify(self, result: float) -> bool:
        # return a random bool
        return random.choice([True, False])


class AlwaysFailedAnalyzer(DefaultAnalyzer):
    def __init__(self, truth=None, tolerance=0) -> None:
        super().__init__(truth=truth, tolerance=tolerance)

    def analyze(self, result: Results, **kwargs):
        return 10e9

    def verify(self, result: float) -> bool:
        return False


class ResonatorSpectAnalyzerTranx(Analyzer):
    """Analyzer for resonator spectroscopy in transmission mode.
        Fit a lorentzian to the data and return the resonance frequency.
    Args:
        truth (float, optional): The expected value of the measurement. Defaults to 0.
        tolerance (float, optional): The tolerance of the measurement. Defaults to 0.
    """

    def analyze(
        self,
        result: Results,
        handle: Optional[str] = None,
        f0: float = 0.0e6,
        a: float = 1e-3,
        gamma: float = 1e6,
        offset: float = 0,
        flip_sign: bool = False,
        frequency_offset: float = 0,
    ) -> float:
        """Fit a lorentzian to the data and return the resonance frequency.
        Args:
            result (Results): The result of the measurement.
            handle (str, optional): The handle of the result to analyze. Defaults to None.
            f0 (float, optional): Initial guess for the resonance frequency. Defaults to 0.0e6.
            a (float, optional): Initial guess for the amplitude. Defaults to 1e-3.
            gamma (float, optional): Initial guess for the linewidth. Defaults to 1e6.
            offset (float, optional): Initial guess for the offset. Defaults to 0.
            flip_sign (bool, optional): Flip the sign of the amplitude. Defaults to False.
            frequency_offset (float, optional): Offset the resonance frequency. Defaults to 0.

        Returns:
            float: The resonance frequency.

        Note: frequency_offset: We don't park feedline drive at exactly the resonator resonance. Instead, a frequency_offset is introduced to have a better signal to noise.
        """
        if handle is None:
            handle = list(result.acquired_results.keys())[0]

        freqs = result.acquired_results[handle].axis[0]

        data = result.get_data(handle)

        flip_sign = -1 if flip_sign else 1

        def lorentzian(f, f0, a, gamma, offset, flip_sign):
            penalization = abs(min(0, gamma)) * 1000
            return (
                offset + flip_sign * a / (1 + (f - f0) ** 2 / gamma**2) + penalization
            )

        # f_offset = np.linspace(sweep_start, sweep_stop, sweep_count)
        amplitude = np.abs(data)

        (f_0, a, gamma, offset, flip_sign), _ = curve_fit(
            lorentzian, freqs, amplitude, (f0, a, gamma, offset, flip_sign)
        )
        return f_0 + frequency_offset

    def verify(self, result: float) -> bool:
        assert math.isclose(result, self.truth, abs_tol=self.tolerance)


class QubitSpecAnalyzer(Analyzer):
    """Not implemented yet. any different from the ResonatorSpectAnalyzerTranx ?"""

    def __init__(self, truth=None, tolerance=0) -> None:
        super().__init__(truth=truth, tolerance=tolerance)

    def analyze(self, result: Results, **kwargs):
        return 6.5e9

    def verify(self, result: float) -> bool:
        return True


class RabiAnalyzer(Analyzer):
    def __init__(self, truth=None, tolerance=0) -> None:
        super().__init__(truth=truth, tolerance=tolerance)

    def analyze(self, result: Results, amp_pi=None):
        def evaluate_rabi(
            res,
            amp_pi=None,
            handle=None,
            plot=True,
            rotate=False,
            flip=False,
            real=False,
        ):
            """
            Adapt from tuneup notebook example.
            Need to rework
            """

            def rabi_curve(x, offset, phase_shift, amplitude, period):
                return amplitude * np.sin(np.pi / period * x + phase_shift) + offset

            #  return amplitude*np.sin(2*np.pi/period*x+np.pi/2)+offset

            if handle is None:
                handle = list(res.acquired_results.keys())[0]

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
            if amp_pi is None:
                period_guess = abs(x[np.argmax(y)] - x[np.argmin(y)])
            else:
                period_guess = amp_pi
            p0 = [offset_guess, phase_shift_guess, amplitude_guess, period_guess]
            print(f"offset_guess: {offset_guess}")
            print(f"phase_shift_guess: {phase_shift_guess}")
            print(f"amplitude_guess: {amplitude_guess}")
            print(f"period_guess: {period_guess}")
            popt = curve_fit(rabi_curve, x, y, p0=p0)[0]

            pi_amp = popt[3]
            pi2_amp = popt[3] / 2

            if plot:
                plt.figure()
                plt.plot(x, rabi_curve(x, *popt))
                plt.plot(x, y, ".")
                plt.plot([pi_amp, pi_amp], [min(y), rabi_curve(pi_amp, *popt)])
                plt.plot([pi2_amp, pi2_amp], [min(y), rabi_curve(pi2_amp, *popt)])
            print("fitted results")
            print(f"offset_guess: {popt[0]}")
            print(f"phase_shift_guess: {popt[1]}")
            print(f"amplitude_guess: {popt[2]}")
            print(f"period_guess: {popt[3]}")
            print(f"Pi amp: {pi_amp}, pi/2 amp: {pi2_amp}")
            return [pi_amp, pi2_amp]

        pi_amp, pi2_amp = evaluate_rabi(result, amp_pi=amp_pi)
        return pi_amp

    def verify(self, result: float) -> bool:
        return True


class RamseyAnalyzer(DefaultAnalyzer):
    def __init__(self, truth=None, tolerance=0) -> None:
        super().__init__(truth=truth, tolerance=tolerance)

    def analyze(self, result: Results, **kwargs):
        return 10e-3

    def verify(self, result: float) -> bool:
        return True
