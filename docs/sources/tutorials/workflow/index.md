# Tutorials - Workflow

A workflow is a collection of tasks and various other operations.
It supplies the tasks their options saves their inputs and outputs.
When run, a workflow function builds a graph of tasks that will be executed later.
This graph may be inspected.
The graph of tasks is not executed directly by Python, but by a workflow engine
provided by the library.

## Building a workflow

This [example](sources/build_workflow.ipynb) shows the basics of workflows

## Options

Tasks and workflows may take many optional parameters. These are controlled via
`Options` which allows the defaults to be overridden and ensures that the optional
values are propagated to the tasks that require them.

This [example](sources/options.ipynb) shows how workflow functionality can
be extended with options.
