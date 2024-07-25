---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# An Introduction to the Building Blocks of LabOne Q Applications

+++

The LabOne Q Applications library provides building blocks for developing your own experiment and analysis workflows on top of LabOne Q. In this guide we introduce these blocks and illustrate how to use them and how to write your own.

Let's get started.

+++

The three main building blocks are:

* **QuantumOperations**:
  Each set of quantum operations defines operations for a particular type of qubit.
  At the moment the library only provides operations for tunable transmon qubits.
  We'll introduce you to these operations and show you how to add to or modify them.
  You can also create your own kind of qubit and quantum operations for them but that
  will not be covered in this guide.

* **Tasks**:
  Tasks are used to build up experiment and analysis workflows. The library provides
  generic tasks for building, compiling and running LabOne Q experiments. It also
  provides specific tasks for simple experiments and the associated analysis (e.g.
  Rabi).

* **Taskbooks**:
  A `taskbook` is an ordinary Python function that calls tasks. When a task is called
  inside a task book function, the inputs and outputs of the tasks are automatically
  saved. The task book also provides the tasks inside it with their options,
  allowing you to control the details of task behaviour at run time. 

This guide will introduce you to these three building blocks.

There are two more building blocks that are not part of the beta release:

* **LogbookStore**:
  The logbook store defines where a `taskbook` function will store the inputs and
  results of its tasks. For example, in a folder on disk. The store may also be
  used to retreive data and to store your own data.

* **Workflow**:
  A `Workflow` is similar to a `Taskbook` in that it contains a set of tasks to be run and
  supplies the tasks their options and saves their inputs and outputs. However, while a
  `TaskBook` is an ordinary Python function, a `Workflow` is not.

  When run, a `Workflow` function builds a graph of tasks that will be executed later.
  This graph may be inspected and extended.
  The graph of tasks is not executed directly by Python, but by a workflow engine
  provided by the library.

The `LogbookStore` and `Workflow` will not be covered here.

+++

## Setting up a device and session

Build your LabOne Q `DeviceSetup`, qubits and `Session` as normal. Here we import an example from the applications library's test suite (this will change in the near future):

```{code-cell} ipython3
from laboneq.simple import *
from laboneq_applications.qpu_types.tunable_transmon import demo_qpu
```

```{code-cell} ipython3
# Create a demonstration tunable transmon QPU:
single_tunable_transmon_qpu = demo_qpu(n_qubits=1)

# setup is an ordinary LabOne Q DeviceSetup:
setup = single_tunable_transmon_qpu.setup

# qubits is a list of one signle LabOne Q Application Library TunableTransmonQubit qubit:
qubits = single_tunable_transmon_qpu.qubits
```

```{code-cell} ipython3
session = Session(setup)
session.connect(do_emulation=True)
```

## Qubits and qubit parameters

+++

Inspect the qubit parameters

```{code-cell} ipython3
print(qubits[0].parameters)
```

The following qubit parameters are used by the Applications Library:

* `drive_parameters_ge`/`drive_parameters_ef`: contain the pulse parameters for implementing a pi-pulse on the ge and ef transitions
* `readout_parameters`: contains the parameters of the readout pulse
* `readout_integration_parameters`: contains the parameters of the integration kernels. The key `kernels=default` indicates that a constant square pulse with the length specified will be used for the integration (created in `qubit.default_integration_kernels()`). The `kernels` entry can also be set to a list of pulse dictionaries of the form `{"function": pulse_functional_name, "func_par1": value, "func_par2": value, ... }`. `pulse_functional_name` must be the name of a function registered with the `pulse_library.register_pulse_functional` (decorator)[https://docs.zhinst.com/labone_q_user_manual/tutorials/reference/04_pulse_library/].
* `reset_delay_length`: the waiting time for passive qubit reset
* `resonance_frequency_ge`, `resonance_frequency_ef` ' `drive_lo_frequency`, `readout_resonator_frequency`, `readout_lo_frequency`, `drive_range`, `readout_range_out`, `readout_range_in`: used to configure the qubit calibration which then ends up in the `Experiment` calibration.

The remaining qubit parameters are still there for legacy reasons and have no effect. These will be cleaned up soon.',

+++

## Quantum Operations

Quantum operations provide a means for writing DSL at a higher level of abstraction than in base LabOne Q. When writing LabOne Q DSL one works with operations on signal lines. When writing DSL with quantum operations, one works with operations on *qubits*.

**Note**:

The experiments built using quantum operations are just ordinary LabOne Q experiments. It's how the experiments are described that differs. One also uses LabOne Q DSL to *define* quantum operations and one can combine quantum operations with ordinary LabOne Q DSL, because they are producing the same DSL.

+++

### Building a first experiment pulse sequence

Let's build our first experiment pulse sequence using quantum operations. The experiment pulse sequence is described by the LabOne Q `Experiment` object.

We'll need to import some things are the start. We'll explain what each of them is as we go:

```{code-cell} ipython3
import numpy as np

from laboneq_applications import dsl
from laboneq_applications.core.build_experiment import qubit_experiment
from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonOperations,
)
```

Let's start with a tiny experiment sequence that rotates a qubit a given angle about the x-axis and performs a measurement:

```{code-cell} ipython3
@qubit_experiment
def rotate_and_measure(qop, q, angle, count=10):
    """Rotate q by the given angle and measure it."""
    with dsl.acquire_loop_rt(count=count):
        qop.rx(q, angle)
        qop.measure(q, "measure_q")
```

and break down the code line by line:

* `@qubit_experiment`: This decorator creates a new experiment object and makes it accessible inside the `rotate_and_measure` function. It also finds the qubits in the function arguments (i.e. `q`) and sets the experiment calibration using them.

* `def rotate_and_measure(qop, q, angle, count=10):`: These are ordinary function arguments, except for the detection of the qubit objects just mentioned. The `qop` argument supplies the set of quantum operations to use. The same function can be used to build an experiment for any qubit platform that provides the same operations.

* `with dsl.acquire_loop_rt(count=count)`: This is just the `acquire_loop_rt` function from `laboneq.dsl.experiments.builtins`. The `laboneq_applications.dsl` module is just a convenient way to access the LabOne Q DSL functionality.

* `qop.rx(q, angle)`: Here `qop` is a set of quantum operations. The `rx` operation creates a pulse that rotates the qubit by the given angle (in radians) by linearly scaling the pulse amplitude with respect to the qubit pi-pulse amplitude stored in `qubit.parameters.drive_parameters_ge.amplitdue_pi`. The pulse type is specified in `qubit.parameters.drive_parameters_ge.pulse.function` and it uses the length in `qubit.parameters.drive_parameters_ge.length`.
   * To implement a pi-pulse and a pi-half pulse, we provide the operations `qop.x180`, `qop.y180`, `qop.x90`, `qop.y90`, which use the pulse amplitdues values in `qubit.parameters.drive_parameters_ge.amplitdue_pi` and `qubit.parameters.drive_parameters_ge.amplitdue_pi2`, respectively,

* `qop.measure(q, "measure_q")`: Performs a measurement on the qubit, using the readout pulse and kernels specified by the qubit parameters `qubit.parameters.readout_parameters` and `qubit.parameters.readout_integration_parameters`. `"measure_q"` is the handle to store the results under.

+++

To build the experiment we need some qubits and a set of quantum operations. Let's use the `TunableTransmonOperations` provided by the applications library and the qubit we defiend earlier:

```{code-cell} ipython3
qop = TunableTransmonOperations()
q0 = qubits[0]

exp = rotate_and_measure(qop, q0, np.pi / 2, count=10)
```

Here `exp` is just an ordinary LabOne Q experiment:

```{code-cell} ipython3
print(exp)
```

Have a look through the generated experiment and check that:

* the experiment signals are those for the qubit.
* the qubit calibration has been set.
* the experiment sections are those you expect.

+++

### Examining the set of operations

So far we've treated the quantum operations as a black box. Now let's look inside. We can start by listing the quantum operations:

```{code-cell} ipython3
[op for op in dir(qop) if not op.startswith("_")]
```

The `QUBIT_TYPES` specify the type of qubits support by the quantum operations object we've created.
In our case, that's the `TunableTransmonQubit`:

```{code-cell} ipython3
qop.QUBIT_TYPES
```

The `BASE_OPS` is an implementation detail -- they contain the original defintions of the quantum operations. We will ignore it for now except to mention that individual quantum operations can be overridden with alternative implementations if required.

The remainder of the items in the list are the quantum operations themselves.

Let's take a look at one.

+++

### Working with a quantum operation

```{code-cell} ipython3
qop.rx?
```

```{code-cell} ipython3
print(qop.rx.src)
```

One can write:

* `qop.rx?` to view the documentation as usual, or
* `print(qop.rx.src)` to easily see how a quantum operation is implemented.

Take a moment to read the documentation of a few of the other operations and their source, for example:

```{code-cell} ipython3
print(qop.x180.src)
```

```{code-cell} ipython3
print(qop.x90.src)
```

Calling a quantum operation by itself produces a LabOne Q section:

```{code-cell} ipython3
section = qop.rx(qubits[0], np.pi)
print(section)
```

Some things to note about the section:

* The section name is the name of the quantum operation, followed by the UIDs of the qubits it is applied to.
* The section UID is automatically generated from the name.
* The section starts by reserving all the signal lines of the qubit it operates on so that operations acting on the same qubits never overlap.

+++

In addition to `.src` each quantum operation also has two special attributes:

* `.op`: This returns the function that implements the quantum operation.
* `.omit_section(...)`: This method builds the quantum operation but without a containing section. This is useful if one wants to define a quantum operation in terms of another, but not have deeply nested sections.

Let's look at `.op` now. We'll use `.omit_section` once we've seen how to write our own operations.

```{code-cell} ipython3
qop.rx.op
```

### Writing a quantum operation

Often you'll want to write your own quantum operation, either to create a new operation or to replace an existing one.

Let's write our own very simple implementation of an `rx` operation that varies the pulse length instead of the amplitude:

```{code-cell} ipython3
@qop.register
def simple_rx(self, q, angle):
    """A very simple implementation of an RX operation that varies pulse length."""
    # Determined via rigorously calibration ;) :
    amplitude = 0.6
    length_for_pi = 50e-9
    # Calculate the length of the pulse
    length = length_for_pi * (angle / np.pi)
    dsl.play(
        q.signals["drive"],
        amplitude=amplitude,
        phase=0.0,
        length=length,
        pulse=dsl.pulse_library.const(),
    )
```

Applying the decorator `qop.register` wraps our function `simple_rx` in a quantum operation and registers it with our current set of operations, `qop`.

We can confirm that it's registered by checking that its in our set of operations, or by looking it up as an attribute or element of our operations:

```{code-cell} ipython3
"simple_rx" in qop
```

```{code-cell} ipython3
qop.simple_rx
```

```{code-cell} ipython3
qop["simple_rx"]
```

If an operation with the same name already exists it will be replaced.

Let's run our new operations and examine the section it produces:

```{code-cell} ipython3
section = qop.simple_rx(qubits[0], np.pi)
print(section)
```

We can also create aliases for existing quantum operations that are already registered by assigning additional names for them:

```{code-cell} ipython3
qop["rx_length"] = qop.simple_rx
```

```{code-cell} ipython3
"rx_length" in qop
```

### Using omit_section

Let's imagine that we'd like to write an `x90_length` operation that calls our new `rx_length` but always specifies an angle of $\frac{\pi}{2}$. We can write this as:

```{code-cell} ipython3
@qop.register
def x90_length(self, q):
    return self.rx_length(q, np.pi / 2)
```

However, when we call this we will have deeply nested sections and many signal lines reserved. This obscures the structure of our experiment:

```{code-cell} ipython3
section = qop.x90_length(qubits[0])
print(section)
```

We can remove the extra section and signal reservations by call our inner operation using `.omit_section` instead:

```{code-cell} ipython3
@qop.register
def x90_length(self, q):
    return self.rx_length.omit_section(q, np.pi / 2)
```

Note how much simpler the section structure looks now:

```{code-cell} ipython3
section = qop.x90_length(qubits[0])
print(section)
```

### Replacing a quantum operation

To end off our look at quantum operations, let's replace the original `rx` gate with our own one and then use our existing experiment definition to produce a new experiment with the operation we've just written.

```{code-cell} ipython3
qop["rx"] = qop.simple_rx  # replace the rx gate
exp = rotate_and_measure(qop, qubits[0], np.pi / 2)
print(exp)
```

Confirm that the generated experiment contains the new implementation of the RX gate.

+++

### Setting section attributes

Sometimes an operation will need to set special section attributes such as `on_system_grid`.

This can be done by retrieving the current section and directly manipulating it.

To demonstrate, we'll create an operation whose section is required to be on the system grid:

```{code-cell} ipython3
@qop.register
def op_on_system_grid(self, q):
    section = dsl.active_section()
    section.on_system_grid = True
    # ... play pulses, etc.
```

And then call it to confirm that the section has indeed been set to be on the grid:

```{code-cell} ipython3
section = qop.op_on_system_grid(qubits[0])
section.on_system_grid
```

### Accessing experiment calibration

When a qubit experiment is created by the library its calibration is initialized from the qubits it operates on. Typically oscillator frequencies and other signal calibration are set.

Sometimes it may be useful for quantum operations to access or manipulate this configuration using `experiment_calibration` which returns the calibration set for the current experiment.

**Note**:

* The experiment calibration is only accessible if there is an experiment, so quantum operations that call `experiment_calibration` can only be called inside an experiment and will raise an exception otherwise.

* There is only a single experiment calibration per experiment, so if mutliple quantum operations modify the same calibration items, only the last modification will be retained.

Here is how we define a quantum operation that accesses the calibration:

```{code-cell} ipython3
@qop.register
def op_that_examines_signal_calibration(self, q):
    calibration = dsl.experiment_calibration()
    signal_calibration = calibration[q.signals["drive"]]
    # ... examine or set calibration, play pulses, etc, e.g.:
    signal_calibration.oscillator.frequency = 0.2121e9
```

To use it we will have to build an experiment. For now, just ignore the pieces we haven't covered. Writing a complete experiment will be covered shortly:

```{code-cell} ipython3
@qubit_experiment
def exp_for_checking_op(qop, q):
    """Simple experiment to test the operation we've just written."""
    with dsl.acquire_loop_rt(count=1):
        qop.op_that_examines_signal_calibration(q)

exp = exp_for_checking_op(qop, qubits[0])
exp.get_calibration().calibration_items["/logical_signal_groups/q0/drive"]
```

Note above that the oscillator frequency has been set to the value we specified, `0.2121e9` Hz.

+++

## Tasks

When running experiments with LabOne Q there are usually a few higher-level steps that one has to perform:

- Build a DSL experiment.
- Compile the experiment.
- Run the experiment.
- Analyze the experiment.

In the applications library, we call these steps *tasks*. The library provides some predefined tasks and you can also write your own.

Let's use some of the predefined tasks to run an amplitude Rabi experiment.

+++

### Using provided tasks to run an experiment

```{code-cell} ipython3
from laboneq_applications.core.options import TuneupExperimentOptions
from laboneq_applications.experiments.amplitude_rabi import create_experiment
from laboneq_applications.tasks import compile_experiment, run_experiment
from laboneq_applications.workflow import task
```

```{code-cell} ipython3
print(create_experiment.src)
```

Let's create, compile and run the rabi experiment with some simple input parameters.

```{code-cell} ipython3
qop = TunableTransmonOperations()
amplitudes = np.linspace(0.0, 1.0, 10)
options = TuneupExperimentOptions(count=10)

exp = create_experiment(qop, qubits[0], amplitudes, options)
compiled_exp = compile_experiment(session, exp)
result = run_experiment(session, compiled_exp)
```

And let's examine the rabi measurement results:

```{code-cell} ipython3
result.result.q0
```

And the measurements of the 0 and 1 states for calibration:

```{code-cell} ipython3
result.cal_trace.q0.g
```

```{code-cell} ipython3
result.cal_trace.q0.e
```

Each of `create_experiment`, `compile_experiment` and `run_experiment` is a task. They are ordinary Python functions, but they provide some special hooks so that they can be incorporate into taskbooks and workflows later.

Like quantum operations, they can be inspected. Let's insepct the source code of the `create_experiment` task of the rabi to see how the rabi experiment is created. You can inspect the source code of any task.

```{code-cell} ipython3
# docstring
create_experiment?
```

```{code-cell} ipython3
# source code
print(create_experiment.src)
```

### Writing your own tasks

As mentioned, tasks are mostly just ordinary Python functions. You can write your own task as follows:

```{code-cell} ipython3
@task
def add(a, b):
    return a + b
```

```{code-cell} ipython3
add(1, 2)
```

### Writing an experiment task

Built-in tasks like `compile_experiment` and `run_experiment` are quite standard and you shouldn't need to write your own versions very often, but tasks that build experiments, such as `create_experiment` for the rabi, will often be written by you.

Let's write our own version of the `create_experiment` for the rabi experiment that sweeps pulse lengths instead of amplitudes.

```{code-cell} ipython3
from laboneq.dsl.parameter import SweepParameter
```

```{code-cell} ipython3
@task
@qubit_experiment
def duration_rabi(
    qop,
    q,
    q_durations,
    count,
    transition="ge",
    cal_traces=True,
):
    """Pulse duration Rabi experiment."""
    with dsl.acquire_loop_rt(
        count=count,
    ):
        with dsl.sweep(
            name=f"durations_{q.uid}",
            parameter=SweepParameter(f"durations_{q.uid}", q_durations),
        ) as length:
            qop.prepare_state(q, transition[0])
            qop.x180(q, length=length, transition=transition)
            qop.measure(q, f"result/{q.uid}")
            qop.passive_reset(q)

        if cal_traces:
            with dsl.section(
                name=f"cal_states/{q.uid}",
            ):
                for state in transition:
                    qop.prepare_state(q, state)
                    qop.measure(q, f"cal_trace/{q.uid}/{state}")
                    qop.passive_reset(q)
```

Quite a few new constructions and quantum operations have been introduced here, so let's take a closer look:

* `transition="ge"`: Each kind of qubit supports different transitions. For the tunable transmon qubits implement in the applications library the two transitions are `"ge"` (i.e. ground to first excited state) and `"ef"` (i.e. first to second excited state). The tunable transmon operations accept the transition to work with as a parameter.

* `with dsl.sweep`: This is an ordinary DSL sweep.

* `qop.prepare_state`: Prepare the specified qubit state. The tunable transmon `prepare_state` accepts `"g"`, `"e"` and `"f"` as states to prepare.

* `qop.passive_reset`: This operation resets the qubit to the ground state by delaying for an amount of time configured in the calibration.

* `if cal_traces:`: An ordinary Python `if` statement. It allows the calibration traces to be omitted if requested by the parameters.

* `dsl.section`: This creates a new section in the experiment. Sections are important to create timing-consistent and reproducible behavior.

```{code-cell} ipython3
qop = TunableTransmonOperations()
durations = np.linspace(10.0e-9, 50e-9, 10)
count = 10

exp = duration_rabi(qop, qubits[0], durations, count=count)
compiled_exp = compile_experiment(session, exp)
result = run_experiment(session, compiled_exp)
```

```{code-cell} ipython3
result.result.q0
```

```{code-cell} ipython3
result.cal_trace["q0"]["g"]
```

```{code-cell} ipython3
result.cal_trace["q0"]["e"]
```

### Why have tasks at all?

At the moment tasks don't provide much beyond encouraging some structure. Encouraging structure is valuable, but the motivation behind tasks is what's to come:

* Being able to produce a well-organized experiment record when tasks are used in taskbooks and workflows.
* Being able to supply global options to tasks in a structure way.
* Being able to build complex dynamic workflows that can execute tasks conditionally and dynamically add tasks.

+++

## Experiments as Taskbooks

+++

A `Taskbook` is a collection of logically connected `Tasks` whose inputs and outputs depend on each other. We use `Taskbooks` to implement experiments. Experiment taskbooks have a few standard tasks:

- `create_experiment` for creating the experimental pulse sequence
- `compile_experiment` for compiling the `Experiment` returned by `create_experiment`
- `run_experiment` for running the `CompiledExperiment` returned by `compile_experiment`

Let's see what the experiment `Taskbook` looks like for the amplitude Rabi.

```{code-cell} ipython3
from laboneq_applications.experiments import amplitude_rabi
from laboneq_applications.qpu_types.tunable_transmon import TunableTransmonOperations
```

Inspect the source code of the `amplitude_rabi.run` `TaskBook` to see what tasks it has.

```{code-cell} ipython3
print(amplitude_rabi.run.src)
```

### Run the experiment

```{code-cell} ipython3
qop = TunableTransmonOperations()
amplitudes = np.linspace(0.0, 0.9, 10)
options = amplitude_rabi.options()
options.create_experiment.count = 10
options.create_experiment.averaging_mode = "cyclic"
rabi_tb = amplitude_rabi.run(
    session,
    qop,
    qubits[0],
    amplitudes,
    options=options,
)
```

### Inspect the experiment TaskBook

+++

Inspect the input parameters to the `amplitude_rabi.run` `Taskbook`

```{code-cell} ipython3
rabi_tb.input
```

Inspect the tasks inside the taskbook

```{code-cell} ipython3
rabi_tb.tasks
```

```{code-cell} ipython3
[t.name for t in rabi_tb.tasks]
```

Inspect the source code of `create_experiment` to see how the experiment pulse sequence was created.

```{code-cell} ipython3
print(rabi_tb.tasks["create_experiment"].src)
```

Inspect the LabOne Q Experiment object returned by `create_experiment`

```{code-cell} ipython3
print(rabi_tb.tasks["create_experiment"].output)
# alternatively, print(rabi_tb.tasks[0].output)
```

Inspect the LabOne Q CompiledExperiment object returend by `compile_experiment`

```{code-cell} ipython3
print(rabi_tb.tasks["compile_experiment"].output)
```

```{code-cell} ipython3
# inspect pulse sequence with plot_simulation
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation

plot_simulation(
    rabi_tb.tasks["compile_experiment"].output,
    signal_names_to_show=["drive", "measure"],
    start_time=0,
    length=50e-6,
)
```

Inspect the acquired results

```{code-cell} ipython3
acquired_data = rabi_tb.tasks["run_experiment"].output  # the acquired results
acquired_data
```

```{code-cell} ipython3
acquired_data.result.q0
```

```{code-cell} ipython3
# inspect pulse sequence with plot_simulation
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation

plot_simulation(
    rabi_tb.tasks["compile_experiment"].output,
    signal_names_to_show=["drive", "measure"],
    start_time=0,
    length=50e-6,
)
```

### Run until a task

We can run `amplitude_rabi` only up to a specific task. Let's exclude the `run_measurement` task in order to first check if the experiment compiles sucessfully before running the measurement.

```{code-cell} ipython3
qop = TunableTransmonOperations()
amplitudes = np.linspace(0.0, 0.9, 10)
options = amplitude_rabi.options()
options.create_experiment.count = 10
options.run_until = "compile_experiment"
rabi_tb = amplitude_rabi.run(
    session,
    qop,
    qubits[0],
    amplitudes,
    options=options,
)
```

```{code-cell} ipython3
rabi_tb
```

### Inspect taskbook after an error

If an error occurs during the execution of `amplitude_rabi`, we can inspect the tasks that have run up to the task that produced the error using `recover()`. This is particularly useful to inspect the experiment pulse sequence in case of a compilation or measurement error.

Let's introduce a compilation error by sweeping the ampltude to values larger than 1, which is not allowed.

```{code-cell} ipython3
qop = TunableTransmonOperations()
amplitudes = np.linspace(0.0, 1.5, 10)
options = amplitude_rabi.options()
options.create_experiment.count = 10

# here we catch the exception such that the notebook runs through
try:
    rabi_tb = amplitude_rabi.run(
        session,
        qop,
        qubits[0],
        amplitudes,
        options=options,
    )
except Exception as e:
    print("ERROR: ", e)
```

```{code-cell} ipython3
rabi_tb = amplitude_rabi.run.recover()
rabi_tb
```

```{code-cell} ipython3
# inspect the experiment section tree
print(rabi_tb.tasks["create_experiment"].output)
```

```{code-cell} ipython3

```
