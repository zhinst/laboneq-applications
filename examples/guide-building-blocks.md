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

Build your LabOne Q `DeviceSetup`, qubits and `Session` as normal. Here we import an example from the applications library's test suite:

```{code-cell} ipython3
import sys
sys.path.insert(0, "..")

from laboneq.simple import *

from tests.helpers.device_setups import single_tunable_transmon_setup, single_tunable_transmon_qubits
```

```{code-cell} ipython3
# setup is an ordinary LabOne Q DeviceSetup:
setup = single_tunable_transmon_setup()
# qubits is a list of LabOne Q Application Library TunableTransmonQubit qubits:
qubits = single_tunable_transmon_qubits(setup)
```

```{code-cell} ipython3
session = Session(setup)
session.connect(do_emulation=True)
```

## Quantum Operations

Quantum operations provide a means for writing DSL at a higher level of abstraction than in base LabOne Q. When writing LabOne Q DSL one works with operations on signal lines. When writing DSL with quantum operations, one works with operations on *qubits*.

**Note**:

The experiments built using quantum operations are just ordinary LabOne Q experiments. It's how the experiments are described that differs. One also uses LabOne Q DSL to *define* quantum operations and one can combine quantum operations with ordinary LabOne Q DSL, because they are producing the same DSL.

+++

### Building a first experiment

Let's build our first experiment using quantum operations.

We'll need to import some things are the start. We'll explain what each of them is as we go:

```{code-cell} ipython3
import numpy as np

from laboneq_applications import dsl
from laboneq_applications.core.build_experiment import qubit_experiment
from laboneq_applications.core.quantum_operations import QuantumOperations
from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonOperations,
    TunableTransmonQubit
)
```

Let's start with a tiny experiment that rotates a qubit a given angle about the x-axis and performs a measurement:

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

* `qop.rx(q, angle)`: Here `qop` is a set of quantum operations. The `rx` operation rotates the qubit by the given angle (in radians).

* `qop.measure(q, "measure_q")`: Performs a measurement on the qubit. `"measure_q"` is the handle to store the results under.

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

The `QUBIT_TYPE` is the type of qubits support by the quantum operations object we've created.
In our case, that's the `TunableTransmonQubit`:

```{code-cell} ipython3
qop.QUBIT_TYPE
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

Take a moment to read the documentation of a few of the other operations and their source.

+++

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

In addition to `.src` each quantum operation also has two special methods:

* `.partial`: This allows one to create a new quantum operation with some parameter values already specified.
* `.section`: This allows one to create a new quantum operation and override some of the section parameters (e.g. section alignment, section length).

Let's try them out:

```{code-cell} ipython3
x180 = qop.rx.partial(angle=np.pi/5)
section = x180(qubits[0])
print(section)
```

```{code-cell} ipython3
rx_on_grid = qop.rx.section(on_system_grid=True)
section = rx_on_grid(qubits[0], np.pi/2)
print(section)
```

### Writing a quantum operation

Often you'll want to write your own quantum operation, either to create a new operation or to replace an existing one.

Let's write our own very simple implementation of an `rx` operation that varies the pulse length instead of the amplitude:

```{code-cell} ipython3
from laboneq_applications.core.quantum_operations import quantum_operation

def simple_rx(qop, q, angle):
    """A very simple implementation of an RX operation that varies pulse length."""
    # Determined via rigorously calibration ;) :
    amplitude = 0.6
    length_for_pi = 10e-9
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

And register the operation with our existing set of quantum operations:

```{code-cell} ipython3
qop["simple_rx"] = simple_rx
```

```{code-cell} ipython3
section = qop.simple_rx(qubits[0], np.pi)
print(section)
```

### Replacing a quantum operation

To end off our look at quantum operations, let's replace the original `rx` gate with our own one and then use our existing experiment definition to produce a new experiment with the operation we've just written.

```{code-cell} ipython3
qop["rx"] = simple_rx   # replace the rx gate
exp = rotate_and_measure(qop, qubits[0], np.pi / 2)
print(exp)
```

Confirm that the generated experiment contains the new implementation of the RX gate.

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
from laboneq_applications.experiments.rabi import create_experiment
from laboneq_applications.tasks import compile_experiment, run_experiment
from laboneq_applications.workflow import task
```

```{code-cell} ipython3
qop = TunableTransmonOperations()
amplitudes = np.linspace(0.0, 1.0, 10)
options = {"count": 10}

exp = create_experiment(qop, qubits[0], amplitudes, options)
compiled_exp = compile_experiment(session, exp)
result = run_experiment(session, compiled_exp)
```

And let's examine the rabi measurement results:

```{code-cell} ipython3
result.results.result.q0
```

And the measurements of the 0 and 1 states for calibration:

```{code-cell} ipython3
result.results.cal_trace.q0.g
```

```{code-cell} ipython3
result.results.cal_trace.q0.e
```

Each of `amplitude_rabi`, `compile_experiment` and `run_experiment` is a task. They are ordinary Python functions, but they provide some special hooks so that they can be incorporate into taskbooks and workflows later.

Like quantum operations, they can be inspected:

```{code-cell} ipython3
compile_experiment?
```

```{code-cell} ipython3
print(compile_experiment.src)
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

Built-in tasks like `compile_experiment` and `run_experiment` are quite standard and you shouldn't need to write your own versions very often, but tasks that build experiments, such as `amplitude_rabi`, will often be written by you.

Let's write our own version of the `amplitude_rabi` experiment that sweeps pulse lengths instead of amplitudes.

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
result.results.result.q0
```

```{code-cell} ipython3
result.results.cal_trace["q0"]["g"]
```

```{code-cell} ipython3
result.results.cal_trace["q0"]["e"]
```

### Why have tasks at all?

At the moment tasks don't provide much beyond encouraging some structure. Encouraging structure is valuable, but the motivation behind tasks is what's to come:

* Being able to produce a well-organized experiment record when tasks are used in taskbooks and workflows.
* Being able to supply global options to tasks in a structure way.
* Being able to build complex dynamic workflows that can execute tasks conditionally and dynamically add tasks.

+++

## Experiments as Taskbooks

```{code-cell} ipython3
from laboneq_applications.experiments.rabi import create_experiment
from laboneq_applications.tasks import compile_experiment, run_experiment
from laboneq_applications.workflow import task, taskbook
```

```{code-cell} ipython3
@taskbook
def amplitude_rabi_taskbook(qop, qubits, amplitudes, options):
    exp = create_experiment(
        qop,
        qubits,
        amplitudes=amplitudes,
        options=options,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    return result
```

```{code-cell} ipython3
qop = TunableTransmonOperations()
amplitudes = np.linspace(0.0, 0.9, 10)
# pass experiment options as a flat dictionary 
options = {"count": 10, "averaging_mode": AveragingMode.CYCLIC}

logbook = amplitude_rabi_taskbook(
    qop,
    qubits[0],
    amplitudes,
    options=options,
)
```

```{code-cell} ipython3
logbook.output.acquired_results
```

```{code-cell} ipython3
logbook.output.results.result.q0
```

### Inspect tasks

+++

Task names

```{code-cell} ipython3
str(logbook.tasks)
```

Task input arguments

```{code-cell} ipython3
task = logbook.tasks[0]
for k, v in task.parameters.items():
    print(f"    {k}={type(v)}")
```

Task results

```{code-cell} ipython3
for task in logbook.tasks:
    print(f"{task.name}")
    print("Result:")
    print(f"        {type(task.output)}")
    print()
```

### Use temporary qubit parameters

```{code-cell} ipython3
from laboneq_applications.qpu_types.tunable_transmon import modify_qubits
amplitudes = np.linspace(0.0, 0.9, 10)
temporary_qubit_parameters = [
    (qubits[0], {
        "reset_delay_length": 10e-6,
    }),
]

logbook = amplitude_rabi_taskbook(
    qop,
    modify_qubits(temporary_qubit_parameters)[0],
    amplitudes,
    options={"count": 10},
)
```
