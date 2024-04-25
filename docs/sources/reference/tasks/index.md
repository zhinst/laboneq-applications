# Workflow tasks

This section describes the tasks that are available in the workflow system. Tasks are the building blocks of workflows. They are the individual steps that are executed in a workflow. Each task has a specific purpose and can be used to perform a specific action. They can be ran standalone, but usually are used in combination with other tasks in a workflow, where they are evaluated once the workflow is executed.

##  [`run_experiment`](run_experiment.md)

The `run_experiment` task is used to run an experiment on a quantum processor. The task requires a connected session and a compiled experiment to be specified. The task will return the results of a LabOne Q Session.run() call.

<!--nav-->

* [run_experiment](run_experiment.md)
