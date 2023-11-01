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

The constructor of the analyzer class is simple and requires the following parameters:
- `truth`: the ground truth of the analyzed result. It is used for comparison with the analyzed result.
- `tolerance`: the tolerance for the comparison between the truth and the analyzed result.
- `handles`: the handle of the result to analyze. If not provided, the first handle will be used.


```python
def __init__(self, truth: float = 0, tolerance: float = 0, handles= None) -> None:
    self.truth = truth
    self.tolerance = tolerance
    self.handles = handles
```
You could also rely on the constructor of the base class by calling `super().__init__(truth, tolerance, handles)`

```python
def __init__(self, truth: float = 0, tolerance: float = 0, handles= None) -> None:
    super().__init__(truth, tolerance, handles)
```
Frequently, fitting parameters are required for the analyzer. We could provide them in the constructor as well.
Of course, the details of the fitting parameters depend on the concrete implementation of the analyzer.
One example is the fit for a lorentzian function, which requires the initial guess for the resonance frequency, the amplitude, the line-width and the offset.

```python
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
        super().__init__(truth=truth, tolerance=tolerance, handles=handles)
        self.f0 = f0
        self.a = a
        self.gamma = gamma
        self.offset = offset
        self.frequency_offset = frequency_offset
        self.flip = 1 if flip else -1
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
    assert math.isclose(result, self.truth, abs_tol=self.tolerance)
```

It is advisable to use `get_data_x` and `get_data_y` to extract the data from the result object. Both methods are inherited from the base class `Analyzer` and they will return the data which belongs to the given handle.
It is responsibility of the writers of analyzers to make sure the handle is valid and document it well in the docstring of the customized analyzer class.

In the background, a hook `_preprocess_result` is automatically called to preprocess the result object to the nice format `AnalyzeData` for the analyzers.

Having `AnalyzeData` helps to decouple the analyzers from the L1Q experiment result object, making `Analyzer` could be used outside of L1Q. It also provides a convenient way to access the data.




