"""Workflow engine for task execution.

# Summary

Workflow is a collection of tasks and other constructs.
To determite and control the execution of the workflow, the tasks
and other constructs do not behave normally within it and a specific
domain specific language (DSL) must be used.

To achieve this behaviour, workflow first runs through the code
to build a dependency graph of tasks and constructs used in it. The actual code
within the tasks is not yet executed.
For this reason, some of the Python expressions cannot be used within the workflow
and it is recommended to use tasks for any complex logic.

## Workflow constructs

* `if_`: Replaces Python's `if` clause
* `for_`: Replaces Python's `for` loop

## Example

### Building and running the workflow

```python
from laboneq.workflow import task, engine

@task
def my_task(x):
   return x + 1

@engine.workflow
def my_workflow(x):
    if engine.if_(x == 5):
        my_task(x)

result = my_workflow(x=5)
```

### Inspecting the results

The results of all tasks are recorded and may be inspected later
using the result object returned by the workflow.

The result object, `WorkflowResult`, has a tasklog which records each task's execution:

```python
>>> result.tasklog["my_task"]
[6]
```

"""

from laboneq_applications.workflow.engine.core import (
    Workflow,
    WorkflowResult,
    workflow,
)
from laboneq_applications.workflow.engine.expressions import (
    ForExpression as for_,  # noqa: N813
)
from laboneq_applications.workflow.engine.expressions import (
    IFExpression as if_,  # noqa: N813
)
from laboneq_applications.workflow.engine.options import WorkflowOptions

__all__ = ["Workflow", "workflow", "if_", "for_", "WorkflowResult", "WorkflowOptions"]
