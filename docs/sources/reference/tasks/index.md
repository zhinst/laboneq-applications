# Workflow tasks

This section describes the tasks that are available in the workflow system. Tasks are the building blocks of workflows. They are the individual steps that are executed in a workflow. Each task has a specific purpose and can be used to perform a specific action. They can be ran standalone, but usually are used in combination with other tasks in a workflow, where they are evaluated once the workflow is executed.

##  [`compile_experiment`](compile_experiment.md)

The `compile_experiment()` task is used to compile an experiment on a quantum processor. The task requires a connected session and a LabOne Q DSL experiment to be specified. Optionally, [compiler settings](https://docs.zhinst.com/labone_q_user_manual/tips_tricks/#setting-the-compilers-minimal-waveform-and-zero-lengths) can be passed. The task will return the results of LabOne Q `Session.compile()`.

##  [`run_experiment`](run_experiment.md)

The `run_experiment()` task is used to run an experiment on a quantum processor. The task requires a connected session and a compiled experiment to be specified. The task extracts the relevant data from the results of LabOne Q `Session.run()`.

<!--nav-->

* [compile_experiment](compile_experiment.md)
* [run_experiment](run_experiment.md)