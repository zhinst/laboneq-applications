"""A sub-package for creating workflows.

This package provides tools and building blocks to define workflows.

# Summary

Workflow is a collection of tasks and other constructs.
To determine and control the execution of an workflow, the tasks
and other constructs do not behave normally within it and a specific
domain specific language (DSL) must be used.

To achieve this behavior, workflow first runs through the code
to build a dependency graph of tasks and constructs used in it. The actual code
within the tasks is not yet executed.
For this reason, regular Python expressions cannot be used within the workflow
and it is recommended to use tasks for any complex logic.
"""

from laboneq_applications.workflow.engine import (
    Workflow,
    return_,
    workflow,
)
from laboneq_applications.workflow.options import (
    SpectroscopyWorkflowOptions,
    TuneUpWorkflowOptions,
    WorkflowOptions,
)
from laboneq_applications.workflow.recorder import comment, save_artifact
from laboneq_applications.workflow.result import WorkflowResult
from laboneq_applications.workflow.task import task

__all__ = [
    "task",
    "TuneUpWorkflowOptions",
    "Workflow",
    "WorkflowResult",
    "workflow",
    "WorkflowOptions",
    "SpectroscopyWorkflowOptions",
    "return_",
    "comment",
    "save_artifact",
]
