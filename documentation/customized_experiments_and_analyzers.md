```python
from laboneq import *
import requests
from tuneup.experiment import *
from tuneup.scan import *
import tuneup.analyzer as ta
from scipy.optimize import curve_fit
import math
import numpy as np
```

# How to write your customized tuneup experiments

`Automatic Tuneup` framework expects tuneup experiment classes to have certain interfaces. Users could write their own customized tuneup experiment classes by inheriting the base class `TuneUpExperimentFactory` and providing the concrete implementation for the required abstract methods.

We will illustrate both the concepts and how-to by writing an experiment that determines the resonance of a readout resonator as a function of dc biased flux. 


In general, the tuneup experiment class should have the following signatures for the constructor

- `parameters: List[LinearSweepParameter]`

    A list of sweep parameters, that will be included in the experiment.

- `qubit: QuantumElement`

    The official L1Q QuantumElement object, containing information about mapping between logical signal lines and experimental lines.
        
- `exp_settings: Optional[dict]`

    A dictionary of additional experiment settings such as averaging numbers.

- `ext_calls: Optional[Callable]`

    A user-defined function that may be used in the experiment.

- `pulse_storage: Optional[dict]`

    A dictionary of pulses that may be used in the experiment.

An example for the constructor is as following. Here, because we use an user-defined function to sweep the dc bias `ext_calls` is compulsory and we would rise an `Exception` is it is not provided.

On the last line, we also need to generate an experiment by calling `self._gen_experiment(self.parameter)` and assign it to the attribute `self.exp`


```python
class ReadoutSpectroscopyCWBiasSweep(TuneUpExperimentFactory):
    def __init__(
        self,
        parameters,
        qubit,
        exp_settings={"integration_time": 10e-6, "num_averages": 2**5, "slot": 0},
        ext_calls=None,
        pulse_storage=None,
    ):
        if ext_calls is None:
            raise ValueError("ext_calls must be defined for this experiment")
        self.parameters = parameters
        self.exp_settings = exp_settings
        self.qubit = qubit
        self.ext_call = ext_calls
        self.exp = self._gen_experiment(self.parameters)
```

If there is no special logics in the constructor, we could also call the constructor of the base class to save us a few code lines


```python
class ReadoutSpectroscopyCWBiasSweep(TuneUpExperimentFactory):
    def __init__(
        self,
        parameters,
        qubit,
        exp_settings={"integration_time": 10e-6, "num_averages": 2**5, "slot": 0},
        ext_calls=None,
        pulse_storage=None,
    ):
        if ext_calls is None:
            raise ValueError("ext_calls must be defined for this experiment")
        super().__init__(parameters, qubit, exp_settings, ext_calls, pulse_storage)
        self.exp = self._gen_experiment(self.parameters)
```

Now, let's go through the first required method: `_gen_experiment`. This method plays the central role in the tuneup experiment class by providing the ready-to-go L1Q experiment object. 
Here, we can plug in the attributes `exp_settings`, `parameters` and `qubit`.
The experiment is written in a standard way of L1Q with signals generated from the `experimental_signals(with_calibration=True)` of `qubit`.
Please note that `with_calibration` must be set to `True` so that we could transfer the qubit parameters to the experimental calibration.




```python
def _gen_experiment(self, parameters):
        freq_sweep, dc_volt_sweep = parameters
        exp_settings = self.exp_settings
        exp_spec = Experiment(
            uid="Resonator Spectroscopy",
            signals=self.qubit.experiment_signals(with_calibration=True),
        )

        exp_spec.signals[self.qubit.signals["measure"]].oscillator = Oscillator(
            "measure_osc",
            frequency=freq_sweep,
            modulation_type=ModulationType.HARDWARE,  # HAS TO USE HARDWARE MODULATION FOR SPECTROSCOPY MODE
        )

        with exp_spec.sweep(uid="dc_volt_sweep", parameter=dc_volt_sweep):
            exp_spec.call(self.ext_call, qubit_uid=0, voltage=dc_volt_sweep)
            with exp_spec.acquire_loop_rt(
                uid="shots",
                count=exp_settings["num_averages"],
                acquisition_type=AcquisitionType.SPECTROSCOPY,
            ):
                with exp_spec.sweep(uid="res_freq", parameter=freq_sweep):
                    with exp_spec.section(uid="spectroscopy"):
                        exp_spec.acquire(
                            signal=self.qubit.signals["acquire"],
                            handle="res_spec",
                            length=exp_settings["integration_time"],
                        )
                    with exp_spec.section(uid="delay", length=1e-6):
                        exp_spec.reserve(signal=self.qubit.signals["measure"])

        return exp_spec
```

Sometimes the analyzed results do not match exactly what we want to assign to the qubit parameters. 

For instance, in the spectroscopy experiment we only sweep the frequency of the baseband oscillators and the analyzed results from the fitting do not contain information about the local oscillator frequency.

To overcome this, we could implement `get_updated_value` which is the next abstract method required for the tune up experiment classes.
In `get_updated_value`, we will modify the `analyzed_result` and returns the value that will update the qubit parameters.


```python
def get_updated_value(self, analyzed_result):
    return self.qubit.parameters.readout_lo_frequency + analyzed_result

```

The next two methods are optional. There won't be any exceptions if they are not implemented. 

- `set_extra_calibration`

    In this methods, you can override some of the experiment calibration after the experiment is generated by `_gen_experiment`. 

- `plot`

    used for plotting.

    


```python
def set_extra_calibration(
    self, drive_range=None, measure_range=None, acquire_range=None
):
    if drive_range is not None:
        self.exp.signals[self.qubit.signals["drive"]].range = drive_range
    if measure_range is not None:
        self.exp.signals[self.qubit.signals["measure"]].range = measure_range
    if acquire_range is not None:
        self.exp.signals[self.qubit.signals["acquire"]].range = acquire_range
```

# Analyzers

In a similar way, we can write concrete analyzers for the tuneup by inheriting the base `Analyzer` class.

Here, we will write an analyzer that determines the resonance frequency of a readout resonator.

The constructor of the analyzer class is simple and requires only two parameters:
- `truth`: the ground truth of the analyzed result. It is used for comparison with the analyzed result.
- `tolerance`: the tolerance for the comparison between the truth and the analyzed result.


```python
def __init__(self, truth: float = 0, tolerance: float = 0) -> None:
    self.truth = truth
    self.tolerance = tolerance
```

The two abstract methods required for the analyzer class are: 
- `analyze`

    This method takes the L1Q experiment result object and returns the analyzed result.

- `verify`

    This method takes the analyzed result and verify it against the truth.

Both will be called by the `scan` class.

Concrete implementation for analyzing the resonance frequency of a readout resonator is as following:



```python
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
        gamma (float, optional): Initial guess for the line-width. Defaults to 1e6.
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
```
