# LabOne Q Applications Library

Welcome to the LabOne Q Applications Library documentation.

`laboneq-applications` is an experiment-execution framework for quantum computing experiments and an extensible 
collection of pre-built experiment workflows for various applications using LabOne Q.

## Summary

The package provides various modules to either run pre-defined experiment workflows, define
custom workflows or any other experiment building blocks.

### Using pre-built experiment workflows

- `laboneq_applications.experiments`

Ready-made experiment workflows ready to run with LabOne Q integration.
This includes experiments such as Amplitude Rabi ([amplitude_rabi][laboneq_applications.experiments.amplitude_rabi]) or
Ramsey ([ramsey][laboneq_applications.experiments.ramsey]) and many more.

- `laboneq_applications.qpu_types`

Definitions for the units of quantum-computing experiment quantum platforms, 
such as [TunableTransmonQubit][laboneq_applications.qpu_types.tunable_transmon.TunableTransmonQubit],
[TunableTransmonOperations][laboneq_applications.qpu_types.tunable_transmon.TunableTransmonOperations],
[QuantumPlatform][laboneq_applications.qpu_types.QuantumPlatform]
and [QPU][laboneq_applications.qpu_types.QPU] to be used in experiment workflows.

- `laboneq.workflow.logbook`

Logbooks for recording the execution of experiment workflows.
Objects such as [LogbookStore][laboneq.workflow.logbook.LogbookStore] to create logbooks, 
[FolderStore][laboneq.workflow.logbook.FolderStore] for saving the 
experiment data and [LoggingStore][laboneq.workflow.logbook.LoggingStore] for logging the execution of an experiment.

### Creating experiment workflows

- `laboneq.workflow`

A module that exposes building blocks for creating experiment workflows,
such as [@task][laboneq.workflow.task] and [@workflow][laboneq.workflow.workflow] decorators, 
which turn Python functions into task and workflow objects respectively, or 
[WorkflowOptions][laboneq.workflow.WorkflowOptions] and [TaskOptions][laboneq.workflow.TaskOptions] 
used to parametrize the control of a workflow.

- `laboneq_applications.tasks`

A module containing common tasks that may be used within experiment workflows such
as [compile_experiment][laboneq_applications.tasks.compile_experiment], [run_experiment][laboneq_applications.tasks.run_experiment] 
and [update_qubits][laboneq_applications.tasks.parameter_updating.update_qubits].

- `laboneq_applications.dsl`

A namespace that exposes commonly used objects and extensions for building an LabOne Q `Experiment`, such
as the decorator [@qubit_experiment][laboneq_applications.core.build_experiment.qubit_experiment], 
alongside common quantum operations such as `play()`, `measure()`, etc.

- `laboneq_applications.analysis`

Analysis workflows, tasks and tools for the analysis routines of various experiments, such as 
[calibration_traces_rotation][laboneq_applications.analysis.calibration_traces_rotation], 
[fitting_helpers][laboneq_applications.analysis.fitting_helpers], [plotting_helpers][laboneq_applications.analysis.plotting_helpers].

- `laboneq_applications.contrib`

Experiment and analysis workflows as well as `QuantumElement` types and quantum operations added by us or contributed by our customers. This 
content is primarily here for reference, and its correct functionality is not guaranteed. 
