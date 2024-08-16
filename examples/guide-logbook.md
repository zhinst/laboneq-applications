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

To convert this `md` notebook into a standard Jupyter notebook (`ipynb`), use the following command in the command line (in the folder of this notebook):

`jupytext --to ipynb guide-building-blocks.md`

+++

# Recording Experiment Workflow Results

+++

While running an experiment workflow one would like to keep a record of what took place -- a kind of digital lab book. The LabOne Q Applications Library provides logbooks for just this task.

Each workflow run creates its own logbook. The logbook records the tasks being run and may also be used to store additional data such as device settings, LabOne Q experiments, qubits, and the results of experiments and analyses.

Logbooks need to be stored somewhere, and within the Applications Library, this place is called a logbook store.

Currently the Applications Library supports two kinds of stores:

* `FolderStore`
* `LoggingStore`

The `FolderStore` writes logbooks to a folder on disk. It is used to keep a permanent record of the experiment workflow.

The `LoggingStore` logs what is happening using Python's logging. It provides a quick overview of the steps performed by a workflow.

We'll look at each of these in more detail shortly, but first let us set up a quantum platform to run some experiments on so we have something to record.

+++

## Setting up a quantum platform

Build your LabOne Q `DeviceSetup`, qubits and `Session` as normal. Here we import a demonstration tunable transmon quantum platform from the library and the amplitude Rabi experiment:

```{code-cell} ipython3
import numpy as np
from laboneq.simple import *
from laboneq_applications.experiments import amplitude_rabi
from laboneq_applications.qpu_types.tunable_transmon import demo_platform
```

```{code-cell} ipython3
# Create a demonstration QuantumPlatform for a tunable-transmon QPU:
qt_platform = demo_platform(n_qubits=6)

# The platofrm contains a setup, which is an ordinary LabOne Q DeviceSetup:
setup = qt_platform.setup

# And a tunable-transmon QPU:
qpu = qt_platform.qpu

# Inside the QPU, we have qubits, which is a list of six LabOne Q Application Library TunableTransmonQubit qubits:
qubits = qpu.qubits
```

```{code-cell} ipython3
session = Session(setup)
session.connect(do_emulation=True)
```

## The LoggingStore

+++ {"slideshow": {"slide_type": "subslide"}}

When you import the `laboneq_applications` library it automatically creates a default `LoggingStore` for you. This logging store is used whenever a workflow is executed and logs information about:

* the start and end of workflows
* the start and end of tasks
* any errors that occur
* comments (adhoc messages from tasks, more on these later)
* any data files that would be saved if a folder store was in use (more on these later too) 

These logs don't save anything on disk, but they will allow us to see what events are recorded and what would be saved if we did a have a folder store active.

+++

### An example of logging

+++

Let's run the amplitude Rabi experiment and take a look:

```{code-cell} ipython3
amplitudes = np.linspace(0.0, 0.9, 10)
options = amplitude_rabi.options()
options.create_experiment.count = 10
options.create_experiment.averaging_mode = "cyclic"
rabi_tb = amplitude_rabi.experiment_workflow(
    session,
    qpu,
    qubits[0],
    amplitudes,
    options=options,
)
```

The workflow has not yet been executed, but when you run the next cell, you should see messages like:

```
──────────────────────────────────────────────────────────────────────────────
 Workflow 'experiment_workflow': execution started
────────────────────────────────────────────────────────────────────────────── 
```

appear in the logs beneath the cell.

```{code-cell} ipython3
result = rabi_tb.run()
```

And that's all there is to the basic logging functionality.

+++

### Advanced logging uses

+++

If you need to create a logging store of your own you can do so as follows:

```{code-cell} ipython3
from laboneq_applications.logbook import LoggingStore

logging_store = LoggingStore()
```

The logging store created above won't be active unless you run:

```{code-cell} ipython3
logging_store.activate()
```

And you deactivate it with:

```{code-cell} ipython3
logging_store.deactivate()
```

You can access the default logging store by importing it from `laboneq_applications.logbook`:

```{code-cell} ipython3
from laboneq_applications.logbook import DEFAULT_LOGGING_STORE
DEFAULT_LOGGING_STORE
```

## The FolderStore

+++

### Using the folder store

+++

The `FolderStore` saves workflow results on disk and is likely the most important logbook store you'll use.

You can import it as follows:

```{code-cell} ipython3
from laboneq_applications.logbook import FolderStore
```

To create a folder store you'll need to pick a folder on disk to store logbooks in. Here we select `./experiment_store` as the folder name but you should pick your own.

Each logbook created by a workflow will have its own sub-folder. The sub-folder name will start with a timestamp, followed by the name of the workflow, for example `20240728T175500-amplitude-rabi/`. If necessary, a unique count will be added at the end to make the sub-folder name unique.

The timestamps are in UTC, so they might be offset from your local time, but will be meaningful to users in other timezones and will remain correctly ordered when changing to or from daylight savings.

The folder store will need to be activated before workflows will use it automatically.

```{code-cell} ipython3
folder_store = FolderStore("./experiment_store")
folder_store.activate()
```

Now let's run the amplitude Rabi workflow. As before we'll see the task events being logged. Afterwards we'll explore the folder to see what has been written to disk.

```{code-cell} ipython3
result = rabi_tb.run()
```

If you no longer wish to automatically store workflow results in the folder store, you can deactivate it with:

```{code-cell} ipython3
folder_store.deactivate()
```

### Exploring what was written to disk

+++

Here we will use Python's `pathlib` functionality to explore what has been written to disk, but you can also use whatever ordinary tools you prefer (terminal, file navigator).

```{code-cell} ipython3
import json
from pathlib import Path
```

Remember that above we requested that the folder store use a folder named `experiment_store`. Let's list the logbooks that were created in that folder:

```{code-cell} ipython3
store_folder = Path("experiment_store")

amplitude_rabi_folders = sorted(store_folder.glob("*-experiment-workflow"))
amplitude_rabi_folders
```

Our amplitude Rabi experiment is the most recent one run, so let's look at the files within the most recent folder. Note that the logbook folder names start with a timestamp followed by the name of the workflow, which allows us to easily order them by time and to find the workflow we're looking for:

```{code-cell} ipython3
amplitude_rabi_folder = amplitude_rabi_folders[-1]

amplitude_rabi_files = sorted(amplitude_rabi_folder.iterdir())
amplitude_rabi_files
```

At the moment there is only a single file saved. This is the log of what took place. The log is stored in a format called "JSONL" which means each line of the log is a simple Python dictionary stored as JSON.

Let's open the file and list the logs:

```{code-cell} ipython3
experiment_log = amplitude_rabi_folder / "log.jsonl"
logs = [
    json.loads(line) for line in experiment_log.read_text().splitlines()
]
logs
```

In the remaining sections we'll look at how to write adhoc comments into the logs and how to save data files to disk.

+++

## Logging comments from within tasks

+++

Logbooks allow tasks to add their own messages to the logbook as comments.

This is done by calling the `comment(...)` function within a task.

We'll work through an example below:

```{code-cell} ipython3
from laboneq_applications.workflow import task, workflow
from laboneq_applications.logbook import comment
```

Let's write a small workflow and a tiny task that just writes a comment to the logbook:

```{code-cell} ipython3
@task
def log_a_comment(msg):
    comment(msg)

@workflow
def demo_comments():
    log_a_comment("Activating multi-state discrimination! <sirens blare>")
    log_a_comment("Analysis successful! <cheers>")
```

Now when we run the workflow we'll see the comments appear in the logs:

```{code-cell} ipython3
wf = demo_comments()
result = wf.run()
```

Above you should see the two comments. They look like this:
```
Comment: Activating multi-state discrimination! <sirens blare>
...
Comment: Analysis successful! <cheers>
```

+++

## Store data from within tasks

+++

Logbooks also allow files to be saved to disk using the function `save_artifact`.

Here we will create a figure with matplotlib and save it to disk. The folder store will automatically save it as a PNG.

The kinds of objects the folder store can currenly save are:

* Python strings (saved as a text file)
* Python bytes (saved as raw data)
* Pydantic models (saved as JSON)
* PIL images (saved as PNGs by default)
* Matplotlib figures (saved as PNGs by default)
* Numpy arrays (saved as Numpy data files)

Support for more kinds of objects coming soon (e.g. `DeviceSetup`, `Experiment`).

```{code-cell} ipython3
import PIL
from matplotlib import pyplot as plt

from laboneq_applications.logbook import save_artifact
```

Let's write a small workflow that plots the sine function and saves the plot using `save_artifact`:

```{code-cell} ipython3
@task
def sine_plot():
    fig = plt.figure()
    plt.title("A sine wave")
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    plt.plot(x, y)

    save_artifact("Sine Plot", fig)

@workflow
def demo_saving():
    sine_plot()
```

Since we deactivated the folder store, let's activate it again now:

```{code-cell} ipython3
folder_store.activate()
```

And run our workflow:

```{code-cell} ipython3
wf = demo_saving()
result = wf.run()
```

You can see in the logs that an artifact was created:
```
Artifact: 'Sine Plot' of type 'Figure' logged
```
Now let's load the image from disk.

First we need to find the logbook folder created for our workflow:

```{code-cell} ipython3
demo_saving_folders = sorted(store_folder.glob("*-demo-saving"))
demo_saving_folder = demo_saving_folders[-1]
demo_saving_folder
```

And let's list its contents:

```{code-cell} ipython3
sorted(demo_saving_folder.iterdir())
```

And finally let's load the saved image using PIL:

```{code-cell} ipython3
PIL.Image.open(demo_saving_folder / "Sine Plot.png")
```

Saving an object also generates an entry in the folder store log.

We can view it by opening the log:

```{code-cell} ipython3
experiment_log = demo_saving_folder / "log.jsonl"
logs = [
    json.loads(line) for line in experiment_log.read_text().splitlines()
]
logs
```

As you can see above the log records the name (`artifact_name`) and type (`artifact_type`) of the object saved, and the name of the file it was written to (`artifact_files`)

Saving an artifact might potentially write multiple files to disk.

The `artifact_metadata` contains additional user supplied information about the object saved, while `artifact_options` provide initial information on how to save the object. For example, we could have elected to save the figure in another file format. We'll see how to use both next.

+++

### Specifying metadata and options when saving

+++

Let's again make a small workflow that saves a plot, but this time we'll add some options and metadata.

```{code-cell} ipython3
@task
def sine_plot_with_options():
    fig = plt.figure()
    plt.title("A sine wave")
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    plt.plot(x, y)
    [ax] = fig.get_axes()

    save_artifact(
        "Sine Plot",
        fig,
        metadata={
            "title": ax.get_title(),
        },
        options={
            "format": "jpg",
        },
    )

@workflow
def demo_saving_with_options():
    sine_plot_with_options()
```

And run the workflow to save the plot:

```{code-cell} ipython3
wf = demo_saving_with_options()
result = wf.run()
```

Again we open the workflow folder and load the saved image:

```{code-cell} ipython3
demo_saving_with_options_folders = sorted(store_folder.glob("*-demo-saving-with-options"))
demo_saving_with_options_folder = demo_saving_with_options_folders[-1]
demo_saving_with_options_folder
```

```{code-cell} ipython3
sorted(demo_saving_with_options_folder.iterdir())
```

Now when we load the image it is very slightly blurry, because it was saved as a JPEG which uses lossy compression:

```{code-cell} ipython3
PIL.Image.open(demo_saving_with_options_folder / "Sine Plot.jpg")
```

And if we view the logs we can see that the title was recorded in the `artifact_metadata`:

```{code-cell} ipython3
experiment_log = demo_saving_with_options_folder / "log.jsonl"
logs = [
    json.loads(line) for line in experiment_log.read_text().splitlines()
]
logs
```

We're done!
