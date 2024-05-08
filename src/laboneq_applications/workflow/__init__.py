"""A sub-package for creating workflows.

This package provides tools and building blocks to define workflows.
"""

from laboneq_applications.workflow.expressions import IFExpression as if_  # noqa: N813
from laboneq_applications.workflow.task import task
from laboneq_applications.workflow.workflow import Workflow, workflow

__all__ = ["Workflow", "task", "workflow", "if_"]
