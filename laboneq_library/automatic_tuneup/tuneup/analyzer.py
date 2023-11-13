import math
import random
from abc import ABC, abstractmethod
from collections import UserDict
from dataclasses import dataclass
from functools import wraps
from typing import List, Optional

import numpy as np
from laboneq.simple import Results
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit

from .helper import rotate_to_real_axis


@dataclass
class AnalyzeDatum:
    name: str
    x: np.ndarray[(float,), 1]
    y: np.ndarray[(float,), 1]


class AnalyzeData(UserDict[str, AnalyzeDatum]):
    def __setitem__(self, key, item):
        if isinstance(item, AnalyzeDatum):
            super().__setitem__(key, item)
        else:
            raise TypeError("Value must be AnalyzeDatum")


def _create_hook(func, hook):
    @wraps(func)
    def wrapper(*args, **kwargs):
        hook(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


class Analyzer(ABC):
    """Base class for all analyzers.
    Analyzers are used to analyze the results of a measurement and return a value.
    Analyzers can also be used to verify if a measurement is successful or not.
    Analyzers can be used to plot the results of a measurement.
    """

    def __new__(cls, *args, **kwargs):
        cls.analyze = _create_hook(cls.analyze, cls._preprocess_result)
        return super().__new__(cls)

    def __init__(
        self,
        truth: float = 0,
        tolerance: float = 0,
        handles: Optional[List[str]] = None,
    ) -> None:
        """Initialize the analyzer.
        Args:
            truth (float, optional): The expected value of the measurement. Defaults to 0.
            tolerance (float, optional): The tolerance of the measurement. Defaults to 0.
            handles (Optional[List[str]], optional): The handles of the results to analyze. Defaults to None. If None, the first handle is used.

        """
        self.truth = truth
        self.tolerance = tolerance
        self.handles = handles
        self._result = AnalyzeData()

    @abstractmethod
    def analyze(self, result: Results) -> float:
        pass

    @abstractmethod
    def verify(self, result: float) -> bool:
        pass

    def _preprocess_result(self, result: Results | AnalyzeData) -> AnalyzeData:
        if isinstance(result, AnalyzeData):
            self._result = result
            return self._result
        if self.handles is None:
            handles = list(result.acquired_results.keys())[:1]
        else:
            handles = self.handles
        for h in handles:
            x = result.get_axis(h)
            y = result.get_data(h)
            temp = AnalyzeDatum(name=h, x=x, y=y)
            self._result.update({h: temp})
        return self._result

    def plot(self) -> None:
        raise NotImplementedError("Plotting of fitting values not implemented")

    def get_data_y(self, handle) -> ArrayLike:
        return self._result[handle].y

    def get_data_x(self, handle) -> List[ArrayLike]:
        return np.asarray(self._result[handle].x)


class MockAnalyzer(Analyzer):
    def analyze(self, result: Results) -> float:
        return 1234

    def verify(self, result: float) -> bool:
        return True


class RandomAnalyzer(Analyzer):
    def analyze(self, result: Results):
        return 6.5e9

    def verify(self, result: float) -> bool:
        # return a random bool
        return random.choice([True, False])


class AlwaysFailedAnalyzer(Analyzer):
    def analyze(self, result: Results):
        return 10e9

    def verify(self, result: float) -> bool:
        return False


class Lorentzian(Analyzer):
    """Analyzer for resonator spectroscopy in transmission mode.
    Fit a Lorentzian to the data and return the resonance frequency.
    """

    def __init__(
        self,
        truth=None,
        tolerance=0,
        handles=None,
        f0: float = 0.06,
        a: float = 1e-3,
        gamma: float = 1e6,
        offset: float = 0,
        frequency_offset: float = 0,
        flip: bool = True,
    ) -> None:
        """Initialize the analyzer.
        Args:
            f0 (float, optional): Initial guess for the resonance frequency. Defaults to 0.0e6.
            a (float, optional): Initial guess for the amplitude. Defaults to 1e-3.
            gamma (float, optional): Initial guess for the line-width. Defaults to 1e6.
            offset (float, optional): Initial guess for the offset. Defaults to 0.
            flip: (bool, optional): Flip the sign of the amplitude. Defaults to False.
            frequency_offset (float, optional): Offset the resonance frequency. Defaults to 0.

        Note on the usage of frequency_offset.
        When the Purcell filter does not resonate with the readout resonator, the transmission profile of the latter is just Lorentzian.
        In that case, we could use this simple Lorentzian fit to extract the resonance frequency.
        However, for better SNR of the readout, we park the readout at a frequency slightly different than the resonance.
        """

        super().__init__(truth=truth, tolerance=tolerance, handles=handles)
        self.f0 = f0
        self.a = a
        self.gamma = gamma
        self.offset = offset
        self.frequency_offset = frequency_offset
        self.flip = 1 if flip else -1

    def analyze(
        self,
        result: Results,
    ) -> float:
        """Fit a lorentzian to the data and return the resonance frequency.
        Args:
            result (Results): The result of the measurement.

        Returns:
            float: The resonance frequency.

        """

        frequency = self.get_data_x(self.handles[0])[0]
        amplitude = self.get_data_y(self.handles[0])

        flip = self.flip

        def lorentzian(f, f0, a, gamma, offset):
            penalization = abs(min(0, gamma)) * 1000
            return (
                offset + flip * a / (1 + (f - self.f0) ** 2 / gamma**2) + penalization
            )

        (f_0, a, gamma, offset), _ = curve_fit(
            lorentzian, frequency, amplitude, (self.f0, self.a, self.gamma, self.offset)
        )
        return f_0 + self.frequency_offset

    def verify(self, result: float) -> bool:
        return math.isclose(result, self.truth, abs_tol=self.tolerance)


class QubitSpecAnalyzer(Analyzer):
    """Not implemented yet. any different from the ResonatorSpectAnalyzerTranx ?"""

    def __init__(self, truth=None, tolerance=0) -> None:
        super().__init__(truth=truth, tolerance=tolerance)

    def analyze(self, result: Results, **kwargs):
        return 6.5e9

    def verify(self, result: float) -> bool:
        return True


class RabiAnalyzer(Analyzer):
    def __init__(
        self,
        truth=None,
        tolerance=0,
        handles=None,
        amp_pi=0.5,
        phase=0,
        offset=0,
        rotate=False,
        real=False,
    ) -> None:
        super().__init__(truth=truth, tolerance=tolerance, handles=handles)
        self.amp_pi = amp_pi
        self.rotate = rotate
        self.real = real
        self.phase = phase
        self.offset = offset

    @classmethod
    def rabi_curve(cls, x, offset, phase_shift, amplitude, period):
        return amplitude * np.sin(np.pi / period * x + phase_shift) + offset

    def analyze(self, result: Results):

        x = self.get_data_x(self.handles[0])[0]
        y = self.get_data_y(self.handles[0])

        if self.rotate:
            y = np.real(rotate_to_real_axis(y))
        elif self.real:
            y = np.real(y)
        else:
            y = np.abs(y)

        amplitude_guess = max(y) - min(y)
        if self.amp_pi is None:
            period_guess = abs(x[np.argmax(y)] - x[np.argmin(y)])
        else:
            period_guess = self.amp_pi
        p0 = [self.offset, self.phase, amplitude_guess, period_guess]
        self.popt = curve_fit(self.__class__.rabi_curve, x, y, p0=p0)[0]

        pi_amp = self.popt[3]
        return pi_amp

    def verify(self, result: float) -> bool:
        return math.isclose(result, self.truth, abs_tol=self.tolerance)

    def plot(self):
        x = self.get_data_x(self.handles[0])[0]
        y = self.get_data_y(self.handles[0])
        pi_amp = self.popt[3]
        pi2_amp = pi_amp / 2

        fig = plt.figure()
        plt.scatter(x, y)
        plt.plot(x, self.__class__.rabi_curve(x, *self.popt))
        plt.plot(
            [pi_amp, pi_amp], [min(y), self.__class__.rabi_curve(pi_amp, *self.popt)]
        )
        plt.plot(
            [pi2_amp, pi2_amp], [min(y), self.__class__.rabi_curve(pi2_amp, *self.popt)]
        )
        plt.show()

        return fig


class RamseyAnalyzer(Analyzer):
    """Analyzer for Ramsey measurement. Not implemented yet."""

    def __init__(self, truth=None, tolerance=0) -> None:
        super().__init__(truth=truth, tolerance=tolerance)

    def analyze(self, result: Results, **kwargs):
        return 10e-3

    def verify(self, result: float) -> bool:
        return True
