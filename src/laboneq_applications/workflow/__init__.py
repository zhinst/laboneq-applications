"""A package for creating workflows.

The package provides tools and building blocks to define workflows.
"""

# TODO: Remove and fix imports elsewhere

from laboneq.workflow.blocks import (
    break_,
    elif_,
    else_,
    for_,
    if_,
    return_,
)
from laboneq.workflow.core import (
    Workflow,
    workflow,
)
from laboneq.workflow.exceptions import WorkflowError
from laboneq.workflow.executor import execution_info
from laboneq.workflow.options import TaskOptions, WorkflowOptions
from laboneq.workflow.options_base import option_field, options
from laboneq.workflow.options_builder import show_fields
from laboneq.workflow.recorder import (
    comment,
    log,
    save_artifact,
)
from laboneq.workflow.result import TaskResult, WorkflowResult
from laboneq.workflow.task_wrapper import task

__all__ = [
    # Decorators
    "task",
    "workflow",
    # Core
    "Workflow",
    "WorkflowResult",
    "TaskResult",
    # Options
    "options",
    "option_field",
    "WorkflowOptions",
    "TaskOptions",
    "show_fields",
    # Workflow operations
    "return_",
    "if_",
    "elif_",
    "else_",
    "for_",
    "break_",
    # Task operations
    "comment",
    "log",
    "save_artifact",
    "execution_info",
    # Errors
    "WorkflowError",
]
