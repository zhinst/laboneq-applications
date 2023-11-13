```python
from laboneq import *
import requests
from tuneup.experiment import *
from tuneup.scan import *
import tuneup.analyzer as ta
from scipy.optimize import curve_fit
import math
import numpy as np
from tuneup import TuneUp
```

LabOneQ Automatic Tuneup provides a simple way to set up customized workflow that can be used to tune up qubits. The workflow is defined by a list of inter-dependent `scan`. Each `scan` represents a measurement of a single qubit parameter, and the result of the measurement is used to determine the next `scan` in the workflow. 

In this tutorial, we will learn how to set up a workflow for tuning up a single qubit. We will use tune up experiment already defined in the library.

# Preparation steps




```python
from textwrap import dedent

descriptor = dedent(
    """\
instruments:
  SHFQC:
    - address: dev12132
      uid: device_shfqc

  PQSC:
    - address: dev10062
      uid: device_pqsc
connections:

  device_shfqc:
    - iq_signal: q0/measure_line
      ports: QACHANNELS/0/OUTPUT
    - iq_signal: q1/measure_line
      ports: QACHANNELS/0/OUTPUT
    - iq_signal: q2/measure_line
      ports: QACHANNELS/0/OUTPUT
    - iq_signal: q3/measure_line
      ports: QACHANNELS/0/OUTPUT
    - acquire_signal: q0/acquire_line
      ports: QACHANNELS/0/INPUT
      
    - iq_signal: q0/drive_line
      ports: SGCHANNELS/0/OUTPUT
  device_pqsc:
    - to: device_shfqc
      port: ZSYNCS/1
"""
)
device_setup = DeviceSetup.from_descriptor(
    descriptor,
    server_host="localhost",
    server_port=8004,
    setup_name="mysetup",
)
```


```python
q0 = Transmon.from_logical_signal_group(
    "q0",
    lsg=device_setup.logical_signal_groups["q0"],
    parameters=TransmonParameters(
        resonance_frequency_ge=4.7e9,
        resonance_frequency_ef=7e9,
        drive_lo_frequency=4.7e9,
        readout_resonator_frequency=7e9,
        readout_lo_frequency=7e9,
        drive_range=5,
        readout_range_out=-30,
        readout_range_in=-5,
        user_defined={"pi_pulse_amplitude": 1.0},
    ),
)
```


```python
session = Session(device_setup=device_setup)
session.connect(do_emulation=True)
```

# Setting up scans

We set up two basic spectroscopy measurements: one to obtain the resonance of the resonator (`scan_pulsed_resonator`) and the other (`scan_pulsed_qubit`) to obtain the resonance of the qubit.
`scan_pulsed_qubit` depends on `scan_pulsed_resonator`.

The qubit resonant frequency obtained by `scan_pulsed_qubit` will be used in `scan_amp_rabi` to find the optimal amplitude for $\pi$ pulses.

To simulate the workflow, we will use `MockAnalyzer` which returns a fixed value for each scan and always returns `True` when asked for verification.

## Pulsed resonator spectroscopy


```python
freq_sweep = LinearSweepParameter(start=35e6, stop=45e6, count=210)
spec_analyzer = ta.MockAnalyzer()
exp_settings = {"integration_time": 10e-6, "num_averages": 2**10}
readout_pulse = pulse_library.const(
    uid="readout_pulse", length=2e-6, amplitude=0.05
)
kernel_pulse = pulse_library.const(uid="kernel_pulse", length=2e-6, amplitude=1.0)
pulse_storage = {"readout_pulse": readout_pulse, "kernel_pulse": kernel_pulse}
scan_pulsed_resonator = Scan(
    uid="pulsed_resonator_spec",
    session=session,
    qubit=q0,
    params=[freq_sweep],
    update_key="readout_resonator_frequency",
    exp_fac=ReadoutSpectroscopyPulsed,
    exp_settings=exp_settings,
    analyzer=spec_analyzer,
    pulse_storage=pulse_storage,
    analyzing_parameters={
        "f0": 0.4e8,
        "a": 1e-5,
        "gamma": 0.05e8,
        "frequency_offset": 0,
    },
)

scan_pulsed_resonator.set_extra_calibration(measure_range=-30)
```

## Pulsed qubit spectroscopy


```python

freq_sweep = LinearSweepParameter(start=16e6, stop=22e6, count=201)
spec_analyzer = ta.MockAnalyzer()
exp_settings = {"num_averages": 2**11}
readout_pulse = pulse_library.const(
    uid="readout_pulse", length=2e-6, amplitude=0.05
)
kernel_pulse = pulse_library.const(uid="kernel_pulse", length=2e-6, amplitude=1.0)
drive_pulse = pulse_library.const(
    length=2.5e-5,
    amplitude=0.05,
)
pulse_storage = {
    "readout_pulse": readout_pulse,
    "drive_pulse": drive_pulse,
    "kernel_pulse": kernel_pulse,
}
scan_pulsed_qubit = Scan(
    uid="pulsed_qspec",
    session=session,
    qubit=q0,
    params=[freq_sweep],
    update_key="resonance_frequency_ge",
    exp_fac=PulsedQubitSpectroscopy,
    exp_settings=exp_settings,
    analyzer=spec_analyzer,
    pulse_storage=pulse_storage,
    analyzing_parameters={
        "f0": 1.85e7,
        "a": 0.02,
        "gamma": 0.05e8,
        "offset": 0.04,
        "flip_sign": True,
    },
)
scan_pulsed_qubit.set_extra_calibration(drive_range=-25)
```


```python
amp_sweep = LinearSweepParameter(start=0.01, stop=1, count=110)
exp_settings = {"num_averages": 2**12}
readout_pulse = pulse_library.const(
    uid="readout_pulse", length=2e-6, amplitude=0.05
)
kernel_pulse = pulse_library.const(uid="kernel_pulse", length=2e-6, amplitude=1.0)
drive_pulse = pulse_library.gaussian(
    length=1e-7,
    amplitude=1,
)
pulse_storage = {
    "readout_pulse": readout_pulse,
    "drive_pulse": drive_pulse,
    "kernel_pulse": kernel_pulse,
}
scan_amp_rabi = Scan(
    uid="amplitude_rabi",
    session=session,
    qubit=q0,
    params=[amp_sweep],
    update_key="pi_pulse_amplitude",
    exp_fac=AmplitudeRabi,
    exp_settings=exp_settings,
    analyzer=ta.MockAnalyzer(),
    pulse_storage=pulse_storage,
    analyzing_parameters={"amp_pi": 0.2},
)

```

# Setting up workflow

## Add dependency between scans


```python
scan_pulsed_qubit.add_dependencies(scan_pulsed_resonator)
scan_amp_rabi.add_dependencies(scan_pulsed_qubit)
```

## Create tuneup object

A list of scans used in the tuneup must be specified. The order of the scans in the list is not important.


```python
tuneup = TuneUp(uid="tuneupPSI", scans=[scan_prs,scan_qspec,scan_amp_rabi])
```

Alternatively, the highest level scan can be specified. In this case, the tuneup object will automatically find all the scans that are used in the workflow.


```python
tuneup = TuneUp(uid="tuneupPSI", scans=[scan_amp_rabi])
```

# Run the tuneup workflow


```python
tuneup.run(
    scan_amp_rabi,
    plot_graph=True,
    stop_at_failed=True,
    analyze=True,
    verify=True,
    update=True,
)
```


