# Tutorials

## Introduction

This section of the documentation provides practical guides and examples on how to use the components of the LabOneQ Applications library.

## Experiment Workflows

[This tutorial](sources/experiment_workflows.ipynb) shows how to run predefined experiment workflows.

## QuantumOperations

Each set of quantum operations defines operations for a particular type of qubit.
At the moment the library only provides operations for tunable transmon qubits.
[In this tutorial](sources/quantum_operations.ipynb) we'll introduce you to these operations and 
show you how to add to or modify them.
You can also create your own kind of qubit and quantum operations for them.

## Tasks

[This tutorial](sources/tasks.ipynb) shows how tasks are used to build up experiment and analysis workflows. 
The library provides generic tasks for building, compiling and running LabOne Q experiments. 
It also provides specific tasks for simple experiments and the associated analysis (e.g.
Rabi).


## Logbooks

[This tutorial](sources/logbooks.ipynb) is an introduction for logbooks.
The logbook store defines where a workflow will store the inputs and
results of its tasks. For example, in a folder on disk. The store may also be
used to retrieve data and to store your own data.

> **_NOTE:_** Most of the examples are generated from Jupyter Notebook files and the source
can be downloaded on each page by pressing the download button at the top right corner
of the page.
