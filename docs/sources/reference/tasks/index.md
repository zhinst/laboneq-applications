# Workflow tasks

This section describes the tasks that are available in the workflow system. Tasks are the building blocks of workflows. They are the individual steps that are executed in a workflow. Each task has a specific purpose and can be used to perform a specific action. They can be ran standalone, but usually are used in combination with other tasks in a workflow, where they are evaluated once the workflow is executed.

## `run_experiment`

The `run_experiment` task is used to run an experiment on a quantum processor. The task requires a connected session and a compiled experiment to be specified. The task will return the results of a LabOne Q Session.run() call.

### Parameters

- `session` (required): The connected session to use for running the experiment.
- `compiled_experiment` (required): The compiled experiment to run.

### Returns

- `result`: The result of the LabOne Q Session.run() call.

### Example

```python
from laboneq_library.tasks import run_experiment
from laboneq_library.workflow.workflow import Workflow

with Workflow() as wf:
    run_experiment(
        session=session,
        compiled_experiment=compiled_experiment,
    )
```

<!--nav-->
