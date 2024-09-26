# Tutorials

## Introduction

This section of the documentation provides practical guides and examples on how to use the components of the LabOneQ Applications library.

The main components are:

* **QuantumOperations**:
  Each set of quantum operations defines operations for a particular type of qubit.
  At the moment the library only provides operations for tunable transmon qubits.
  We'll introduce you to these operations and show you how to add to or modify them.
  You can also create your own kind of qubit and quantum operations for them.

* **Tasks**:
  Tasks are used to build up experiment and analysis workflows. The library provides
  generic tasks for building, compiling and running LabOne Q experiments. It also
  provides specific tasks for simple experiments and the associated analysis (e.g.
  Rabi).

* **Workflows**:
  A `Workflow` contains a set of tasks to be run. It supplies the tasks their options
  saves their inputs and outputs (if requested).
  When run, a `Workflow` function builds a graph of tasks that will be executed later.
  This graph may be inspected and extended.
  The graph of tasks is not executed directly by Python, but by a workflow engine
  provided by the library.
  Workflows may contain sub-workflows.

* **Options**:
  Tasks and workflows may take many optional parameters. These are controlled via
  `Options` which allows the defaults to be overridden and ensures that the optional
  values are propagated to the tasks that require them.

* **Logbooks**:
  The logbook store defines where a `Workflow` function will store the inputs and
  results of its tasks. For example, in a folder on disk. The store may also be
  used to retrieve data and to store your own data.

> **_NOTE:_** Most of the examples are generated from Jupyter Notebook files and the source
can be downloaded on each page by pressing the download button at the top right corner
of the page.

## Contents

<!--nav-->

* [Quantum Operations](sources/quantum_operations.ipynb)
* [Tasks](sources/tasks.ipynb)
* [Experiment Workflows](sources/experiment_workflows.ipynb)
* [Workflows](sources/workflows.ipynb)
* [Options](sources/options.ipynb)
* [Logbooks](sources/logbooks.ipynb)
