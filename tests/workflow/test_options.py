from __future__ import annotations

import pytest
from pydantic import ValidationError

from laboneq_applications.workflow import (
    TaskOptions,
    WorkflowOptions,
)


class A(WorkflowOptions): ...


class TestWorkflowOptions:
    def test_create_opt(self):
        class A(WorkflowOptions):
            alice: int = 1

        a = A()
        assert a.alice == 1

        # option attributes can be updated
        a.alice = 2
        assert a.alice == 2

    def test_init_non_existing_fields(self):
        # Initialize option with non-existing attributes will raises error
        with pytest.raises(ValidationError):
            WorkflowOptions(non_existing=1)

    def test_task_options(self):
        a = A()
        a.task_options = {"task1": TaskOptions()}
        assert a.task_options["task1"] == TaskOptions()
