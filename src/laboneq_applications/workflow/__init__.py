"""A package for creating workflows.

The package provides tools and building blocks to define workflows.
"""

from laboneq_applications.workflow.blocks.for_block import for_
from laboneq_applications.workflow.blocks.if_block import elif_, else_, if_
from laboneq_applications.workflow.blocks.return_block import return_
from laboneq_applications.workflow.core import (
    Workflow,
    workflow,
)
from laboneq_applications.workflow.exceptions import WorkflowError
from laboneq_applications.workflow.executor import execution_info
from laboneq_applications.workflow.options import TaskOptions, WorkflowOptions
from laboneq_applications.workflow.options_base import option_field, options
from laboneq_applications.workflow.options_builder import show_fields
from laboneq_applications.workflow.recorder import (
    comment,
    log,
    save_artifact,
)
from laboneq_applications.workflow.result import TaskResult, WorkflowResult
from laboneq_applications.workflow.task_wrapper import task

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
    # Task operations
    "comment",
    "log",
    "save_artifact",
    "execution_info",
    # Errors
    "WorkflowError",
]
