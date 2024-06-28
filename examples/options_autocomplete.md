# Note: This notebook is only for early demonstration purposes and testing the idea. Will be removed later.



```python
from laboneq_applications.qpu_types.tunable_transmon import TunableTransmonOperations

from laboneq_applications.core.options import (
    BaseOptions,
    TaskBookOptions,
)
from laboneq_applications.workflow import taskbook, task
from laboneq_applications.workflow.task import task
from laboneq_applications.experiments import rabi
from laboneq.simple import *
import sys
from __future__ import annotations

sys.path.insert(0, "..")
from tests.helpers.device_setups import (
    single_tunable_transmon_setup,
    single_tunable_transmon_qubits,
)
```


```python
setup = single_tunable_transmon_setup()
[q0] = single_tunable_transmon_qubits(setup)
qop = TunableTransmonOperations()
```

The option for rabi experiment is now visible and can be imported from rabi module


```python
from laboneq_applications.experiments import rabi
```


```python
# To use option, you create an instance of the option class and set the desired values
opt = rabi.ExperimentOptions()
opt.count = 10
opt.averaging_mode = AveragingMode.CYCLIC
```


```python
# The options attributes can be shown as
rabi.ExperimentOptions?
# TODO will be polished later for better UI/UX experience
# or
print(opt)
```

    count=10 acquisition_type=AcquisitionType.INTEGRATION averaging_mode=AveragingMode.CYCLIC repetition_mode=RepetitionMode.FASTEST repetition_time=None reset_oscillator_phase=False transition='ge' use_cal_traces=True cal_states='ge'


    [0;31mInit signature:[0m
    [0mrabi[0m[0;34m.[0m[0mExperimentOptions[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0;34m*[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mcount[0m[0;34m:[0m [0mtyping[0m[0;34m.[0m[0mAnnotated[0m[0;34m[[0m[0mint[0m[0;34m,[0m [0mGe[0m[0;34m([0m[0mge[0m[0;34m=[0m[0;36m0[0m[0;34m)[0m[0;34m][0m [0;34m=[0m [0;36m4096[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0macquisition_type[0m[0;34m:[0m [0mstr[0m [0;34m|[0m [0mlaboneq[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mtypes[0m[0;34m.[0m[0menums[0m[0;34m.[0m[0macquisition_type[0m[0;34m.[0m[0mAcquisitionType[0m [0;34m=[0m [0mAcquisitionType[0m[0;34m.[0m[0mINTEGRATION[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0maveraging_mode[0m[0;34m:[0m [0mstr[0m [0;34m|[0m [0mlaboneq[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mtypes[0m[0;34m.[0m[0menums[0m[0;34m.[0m[0maveraging_mode[0m[0;34m.[0m[0mAveragingMode[0m [0;34m=[0m [0mAveragingMode[0m[0;34m.[0m[0mCYCLIC[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mrepetition_mode[0m[0;34m:[0m [0mstr[0m [0;34m|[0m [0mlaboneq[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mtypes[0m[0;34m.[0m[0menums[0m[0;34m.[0m[0mrepetition_mode[0m[0;34m.[0m[0mRepetitionMode[0m [0;34m=[0m [0mRepetitionMode[0m[0;34m.[0m[0mFASTEST[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mrepetition_time[0m[0;34m:[0m [0mfloat[0m [0;34m|[0m [0;32mNone[0m [0;34m=[0m [0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mreset_oscillator_phase[0m[0;34m:[0m [0mbool[0m [0;34m=[0m [0;32mFalse[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mtransition[0m[0;34m:[0m [0mLiteral[0m[0;34m[[0m[0;34m'ge'[0m[0;34m,[0m [0;34m'ef'[0m[0;34m][0m [0;34m=[0m [0;34m'ge'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0muse_cal_traces[0m[0;34m:[0m [0mbool[0m [0;34m=[0m [0;32mTrue[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mcal_states[0m[0;34m:[0m [0mstr[0m [0;34m|[0m [0mtuple[0m [0;34m=[0m [0;34m'ge'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m     
    Base options for a tune-up experiment.
    
    Attributes:
        transition:
            Transition to perform the experiment on. May be any
            transition supported by the quantum operations.
            Default: `"ge"` (i.e. ground to first excited state).
        use_cal_traces:
            Whether to include calibration traces in the experiment.
            Default: `True`.
        cal_states:
            The states to prepare in the calibration traces. Can be any
            string or tuple made from combining the characters 'g', 'e', 'f'.
            Default: same as transition
    [0;31mInit docstring:[0m
    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
    validated to form a valid model.
    
    `self` is explicitly positional-only to allow `self` as a field name.
    [0;31mFile:[0m           ~/code/related_laboneq/laboneq-applications/src/laboneq_applications/core/options.py
    [0;31mType:[0m           ModelMetaclass
    [0;31mSubclasses:[0m     


```python
# create rabi experiment
# use option in rabi task
rabi.create_experiment(qop, q0, [0, 1.0], opt)
```




    Experiment(uid='create_experiment', name='unnamed', signals={'/logical_signal_groups/q0/drive': ExperimentSignal(uid='/logical_signal_groups/q0/drive', calibration=SignalCalibration(oscillator=Oscillator(uid='q0_drive_ge_osc', frequency=100000000.0, modulation_type=ModulationType.HARDWARE, carrier_type=CarrierType.RF), local_oscillator=Oscillator(uid='q0_drive_local_osc', frequency=1500000000.0, modulation_type=ModulationType.AUTO, carrier_type=CarrierType.RF), mixer_calibration=None, precompensation=None, port_delay=None, port_mode=None, delay_signal=None, voltage_offset=None, range=10, threshold=None, amplitude=None, amplifier_pump=None, added_outputs=None, automute=False), mapped_logical_signal_path='/logical_signal_groups/q0/drive'), '/logical_signal_groups/q0/drive_ef': ExperimentSignal(uid='/logical_signal_groups/q0/drive_ef', calibration=SignalCalibration(oscillator=Oscillator(uid='q0_drive_ef_osc', frequency=200000000.0, modulation_type=ModulationType.HARDWARE, carrier_type=CarrierType.RF), local_oscillator=Oscillator(uid='q0_drive_local_osc', frequency=1500000000.0, modulation_type=ModulationType.AUTO, carrier_type=CarrierType.RF), mixer_calibration=None, precompensation=None, port_delay=None, port_mode=None, delay_signal=None, voltage_offset=None, range=10, threshold=None, amplitude=None, amplifier_pump=None, added_outputs=None, automute=False), mapped_logical_signal_path='/logical_signal_groups/q0/drive_ef'), '/logical_signal_groups/q0/measure': ExperimentSignal(uid='/logical_signal_groups/q0/measure', calibration=SignalCalibration(oscillator=Oscillator(uid='q0_measure_osc', frequency=100000000.0, modulation_type=ModulationType.SOFTWARE, carrier_type=CarrierType.RF), local_oscillator=Oscillator(uid='q0_readout_local_osc', frequency=2000000000.0, modulation_type=ModulationType.AUTO, carrier_type=CarrierType.RF), mixer_calibration=None, precompensation=None, port_delay=None, port_mode=None, delay_signal=None, voltage_offset=None, range=5, threshold=None, amplitude=None, amplifier_pump=None, added_outputs=None, automute=False), mapped_logical_signal_path='/logical_signal_groups/q0/measure'), '/logical_signal_groups/q0/acquire': ExperimentSignal(uid='/logical_signal_groups/q0/acquire', calibration=SignalCalibration(oscillator=Oscillator(uid='q0_acquire_osc', frequency=100000000.0, modulation_type=ModulationType.SOFTWARE, carrier_type=CarrierType.RF), local_oscillator=Oscillator(uid='q0_readout_local_osc', frequency=2000000000.0, modulation_type=ModulationType.AUTO, carrier_type=CarrierType.RF), mixer_calibration=None, precompensation=None, port_delay=2e-08, port_mode=None, delay_signal=None, voltage_offset=None, range=10, threshold=None, amplitude=None, amplifier_pump=None, added_outputs=None, automute=False), mapped_logical_signal_path='/logical_signal_groups/q0/acquire'), '/logical_signal_groups/q0/flux': ExperimentSignal(uid='/logical_signal_groups/q0/flux', calibration=SignalCalibration(oscillator=None, local_oscillator=None, mixer_calibration=None, precompensation=None, port_delay=None, port_mode=None, delay_signal=None, voltage_offset=0, range=None, threshold=None, amplitude=None, amplifier_pump=None, added_outputs=None, automute=False), mapped_logical_signal_path='/logical_signal_groups/q0/flux')}, version=DSLVersion.V3_0_0, epsilon=0.0, sections=[AcquireLoopRt(uid='unnamed_0', name='unnamed', alignment=SectionAlignment.LEFT, execution_type=ExecutionType.REAL_TIME, length=None, play_after=None, children=[Sweep(uid='amps_q0_0', name='amps_q0', alignment=SectionAlignment.LEFT, execution_type=None, length=None, play_after=None, children=[Section(uid='prepare_state_q0_0', name='prepare_state_q0', alignment=SectionAlignment.LEFT, execution_type=None, length=None, play_after=None, children=[Reserve(signal='/logical_signal_groups/q0/drive'), Reserve(signal='/logical_signal_groups/q0/drive_ef'), Reserve(signal='/logical_signal_groups/q0/measure'), Reserve(signal='/logical_signal_groups/q0/acquire'), Reserve(signal='/logical_signal_groups/q0/flux')], trigger={}, on_system_grid=False), Section(uid='x180_q0_0', name='x180_q0', alignment=SectionAlignment.LEFT, execution_type=None, length=None, play_after=None, children=[Reserve(signal='/logical_signal_groups/q0/drive'), Reserve(signal='/logical_signal_groups/q0/drive_ef'), Reserve(signal='/logical_signal_groups/q0/measure'), Reserve(signal='/logical_signal_groups/q0/acquire'), Reserve(signal='/logical_signal_groups/q0/flux'), PlayPulse(signal='/logical_signal_groups/q0/drive', pulse=PulseFunctional(function='drag', uid='rx_pulse_0', amplitude=1.0, length=1e-07, can_compress=False, pulse_parameters={'beta': 0.01, 'sigma': 0.21}), amplitude=SweepParameter(uid='amplitude_q0', values=[0, 1.0], axis_name=None, driven_by=None), increment_oscillator_phase=None, phase=0.0, set_oscillator_phase=None, length=5.1e-08, pulse_parameters=None, precompensation_clear=None, marker=None)], trigger={}, on_system_grid=False), Section(uid='measure_q0_0', name='measure_q0', alignment=SectionAlignment.LEFT, execution_type=None, length=None, play_after=None, children=[Reserve(signal='/logical_signal_groups/q0/drive'), Reserve(signal='/logical_signal_groups/q0/drive_ef'), Reserve(signal='/logical_signal_groups/q0/measure'), Reserve(signal='/logical_signal_groups/q0/acquire'), Reserve(signal='/logical_signal_groups/q0/flux'), PlayPulse(signal='/logical_signal_groups/q0/measure', pulse=PulseFunctional(function='const', uid='readout_pulse_0', amplitude=1.0, length=1e-07, can_compress=False, pulse_parameters=None), amplitude=1.0, increment_oscillator_phase=None, phase=None, set_oscillator_phase=None, length=2e-06, pulse_parameters=None, precompensation_clear=None, marker=None), Acquire(signal='/logical_signal_groups/q0/acquire', handle='result/q0', kernel=[PulseFunctional(function='const', uid='integration_kernel_q0_0', amplitude=1.0, length=2e-06, can_compress=False, pulse_parameters=None)], length=2e-06, pulse_parameters=None)], trigger={}, on_system_grid=False), Section(uid='passive_reset_q0_0', name='passive_reset_q0', alignment=SectionAlignment.LEFT, execution_type=None, length=None, play_after=None, children=[Reserve(signal='/logical_signal_groups/q0/drive'), Reserve(signal='/logical_signal_groups/q0/drive_ef'), Reserve(signal='/logical_signal_groups/q0/measure'), Reserve(signal='/logical_signal_groups/q0/acquire'), Reserve(signal='/logical_signal_groups/q0/flux'), Delay(signal='/logical_signal_groups/q0/drive', time=1e-06, precompensation_clear=None)], trigger={}, on_system_grid=False)], trigger={}, on_system_grid=False, parameters=[SweepParameter(uid='amplitude_q0', values=[0, 1.0], axis_name=None, driven_by=None)], reset_oscillator_phase=False, chunk_count=1), Section(uid='cal_q0_0', name='cal_q0', alignment=SectionAlignment.LEFT, execution_type=None, length=None, play_after=None, children=[Section(uid='prepare_state_q0_1', name='prepare_state_q0', alignment=SectionAlignment.LEFT, execution_type=None, length=None, play_after=None, children=[Reserve(signal='/logical_signal_groups/q0/drive'), Reserve(signal='/logical_signal_groups/q0/drive_ef'), Reserve(signal='/logical_signal_groups/q0/measure'), Reserve(signal='/logical_signal_groups/q0/acquire'), Reserve(signal='/logical_signal_groups/q0/flux')], trigger={}, on_system_grid=False), Section(uid='measure_q0_1', name='measure_q0', alignment=SectionAlignment.LEFT, execution_type=None, length=None, play_after=None, children=[Reserve(signal='/logical_signal_groups/q0/drive'), Reserve(signal='/logical_signal_groups/q0/drive_ef'), Reserve(signal='/logical_signal_groups/q0/measure'), Reserve(signal='/logical_signal_groups/q0/acquire'), Reserve(signal='/logical_signal_groups/q0/flux'), PlayPulse(signal='/logical_signal_groups/q0/measure', pulse=PulseFunctional(function='const', uid='readout_pulse_0', amplitude=1.0, length=1e-07, can_compress=False, pulse_parameters=None), amplitude=1.0, increment_oscillator_phase=None, phase=None, set_oscillator_phase=None, length=2e-06, pulse_parameters=None, precompensation_clear=None, marker=None), Acquire(signal='/logical_signal_groups/q0/acquire', handle='cal_trace/q0/g', kernel=[PulseFunctional(function='const', uid='integration_kernel_q0_0', amplitude=1.0, length=2e-06, can_compress=False, pulse_parameters=None)], length=2e-06, pulse_parameters=None)], trigger={}, on_system_grid=False), Section(uid='passive_reset_q0_1', name='passive_reset_q0', alignment=SectionAlignment.LEFT, execution_type=None, length=None, play_after=None, children=[Reserve(signal='/logical_signal_groups/q0/drive'), Reserve(signal='/logical_signal_groups/q0/drive_ef'), Reserve(signal='/logical_signal_groups/q0/measure'), Reserve(signal='/logical_signal_groups/q0/acquire'), Reserve(signal='/logical_signal_groups/q0/flux'), Delay(signal='/logical_signal_groups/q0/drive', time=1e-06, precompensation_clear=None)], trigger={}, on_system_grid=False), Section(uid='prepare_state_q0_2', name='prepare_state_q0', alignment=SectionAlignment.LEFT, execution_type=None, length=None, play_after=None, children=[Reserve(signal='/logical_signal_groups/q0/drive'), Reserve(signal='/logical_signal_groups/q0/drive_ef'), Reserve(signal='/logical_signal_groups/q0/measure'), Reserve(signal='/logical_signal_groups/q0/acquire'), Reserve(signal='/logical_signal_groups/q0/flux'), Section(uid='x180_q0_1', name='x180_q0', alignment=SectionAlignment.LEFT, execution_type=None, length=None, play_after=None, children=[Reserve(signal='/logical_signal_groups/q0/drive'), Reserve(signal='/logical_signal_groups/q0/drive_ef'), Reserve(signal='/logical_signal_groups/q0/measure'), Reserve(signal='/logical_signal_groups/q0/acquire'), Reserve(signal='/logical_signal_groups/q0/flux'), PlayPulse(signal='/logical_signal_groups/q0/drive', pulse=PulseFunctional(function='drag', uid='rx_pulse_0', amplitude=1.0, length=1e-07, can_compress=False, pulse_parameters={'beta': 0.01, 'sigma': 0.21}), amplitude=0.8, increment_oscillator_phase=None, phase=0.0, set_oscillator_phase=None, length=5.1e-08, pulse_parameters=None, precompensation_clear=None, marker=None)], trigger={}, on_system_grid=False)], trigger={}, on_system_grid=False), Section(uid='measure_q0_2', name='measure_q0', alignment=SectionAlignment.LEFT, execution_type=None, length=None, play_after=None, children=[Reserve(signal='/logical_signal_groups/q0/drive'), Reserve(signal='/logical_signal_groups/q0/drive_ef'), Reserve(signal='/logical_signal_groups/q0/measure'), Reserve(signal='/logical_signal_groups/q0/acquire'), Reserve(signal='/logical_signal_groups/q0/flux'), PlayPulse(signal='/logical_signal_groups/q0/measure', pulse=PulseFunctional(function='const', uid='readout_pulse_0', amplitude=1.0, length=1e-07, can_compress=False, pulse_parameters=None), amplitude=1.0, increment_oscillator_phase=None, phase=None, set_oscillator_phase=None, length=2e-06, pulse_parameters=None, precompensation_clear=None, marker=None), Acquire(signal='/logical_signal_groups/q0/acquire', handle='cal_trace/q0/e', kernel=[PulseFunctional(function='const', uid='integration_kernel_q0_0', amplitude=1.0, length=2e-06, can_compress=False, pulse_parameters=None)], length=2e-06, pulse_parameters=None)], trigger={}, on_system_grid=False), Section(uid='passive_reset_q0_2', name='passive_reset_q0', alignment=SectionAlignment.LEFT, execution_type=None, length=None, play_after=None, children=[Reserve(signal='/logical_signal_groups/q0/drive'), Reserve(signal='/logical_signal_groups/q0/drive_ef'), Reserve(signal='/logical_signal_groups/q0/measure'), Reserve(signal='/logical_signal_groups/q0/acquire'), Reserve(signal='/logical_signal_groups/q0/flux'), Delay(signal='/logical_signal_groups/q0/drive', time=1e-06, precompensation_clear=None)], trigger={}, on_system_grid=False)], trigger={}, on_system_grid=False)], trigger={}, on_system_grid=False, acquisition_type=AcquisitionType.INTEGRATION, averaging_mode=AveragingMode.CYCLIC, count=10, repetition_mode=RepetitionMode.FASTEST, repetition_time=None, reset_oscillator_phase=False)])



# Run taskbook rabi with options


```python
session = Session(setup)
session.connect(do_emulation=True)
```




    <laboneq.dsl.session.ConnectionState at 0x1504401d0>



Options of rabi taskbook could be created as:


```python
opt = rabi.amplitude_rabi.options()
```


```python
# run rabi taskbook with options
rabi.amplitude_rabi(session, qop, q0, [0, 1.0], options=opt)
rabi.ExperimentOptions
```




    laboneq_applications.core.options.TuneupExperimentOptions



# Create taskbook from scratch with options


```python
@task
def create_exp(options: rabi.ExperimentOptions | None = None):
    print(options.count)
    return options.count


class AnalysisOptions(BaseOptions):
    plotting: bool = True


@task
def analysis(options: AnalysisOptions | None = None):
    print(options.plotting)
    return options.plotting
```

Here, we list all options that needs to be passed to task using task's names.
For example,
`"create_exp": rabi.ExperimentOptions` means that we would pass an instance of rabi.ExperimentOptions to the task named "create_exp"


```python
class RabiTaskBookOption(TaskBookOptions):
    create_exp: rabi.ExperimentOptions = rabi.ExperimentOptions()
    analysis: AnalysisOptions = AnalysisOptions()


@taskbook(options=RabiTaskBookOption)
def rabi_taskbook(options: RabiTaskBookOption | None = None):
    create_exp()
    analysis()

@taskbook
def rabi_taskbook_normal(options=None):
    create_exp(options=options)
```


```python
options = rabi_taskbook.options()

# or opts = rabi_taskbook.options(create_exp = rabi.ExperimentOptions(), analysis = AnalysisOptions())
```


```python
options.create_exp.count = 1234
options.analysis.plotting = False
```


```python
rabi_taskbook(options=options)
```

    1234
    False





    Taskbook
    Tasks: Task(create_exp), Task(analysis)



Option classes for rabi taskbook are now visible via


```python
options.create_exp?
```

    [0;31mType:[0m           TuneupExperimentOptions
    [0;31mString form:[0m    count=1234 acquisition_type=AcquisitionType.INTEGRATION averaging_mode=AveragingMode.CYCLIC repet <...> tition_time=None reset_oscillator_phase=False transition='ge' use_cal_traces=True cal_states='ge'
    [0;31mFile:[0m           ~/code/related_laboneq/laboneq-applications/src/laboneq_applications/core/options.py
    [0;31mDocstring:[0m     
    Base options for a tune-up experiment.
    
    Attributes:
        transition:
            Transition to perform the experiment on. May be any
            transition supported by the quantum operations.
            Default: `"ge"` (i.e. ground to first excited state).
        use_cal_traces:
            Whether to include calibration traces in the experiment.
            Default: `True`.
        cal_states:
            The states to prepare in the calibration traces. Can be any
            string or tuple made from combining the characters 'g', 'e', 'f'.
            Default: same as transition
    [0;31mInit docstring:[0m
    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
    validated to form a valid model.
    
    `self` is explicitly positional-only to allow `self` as a field name.

Options could be declared for taskbook via another way.


```python
class RabiTaskBookOptions(TaskBookOptions):
    create_exp: rabi.ExperimentOptions = rabi.ExperimentOptions()
    analysis: AnalysisOptions = AnalysisOptions()


@taskbook(options=RabiTaskBookOptions)
def rabi_taskbook(options=None):
    create_exp()
    analysis()
```


```python
opts = rabi_taskbook.options()
rabi_taskbook(options=opts)
```

    4096
    True





    Taskbook
    Tasks: Task(create_exp), Task(analysis)


