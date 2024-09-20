from __future__ import annotations

import pytest
from pydantic import ValidationError

from laboneq_applications.workflow import (
    TaskOptions,
    WorkflowOptions,
)


class A(WorkflowOptions):
    alice: int = 1


class TestWorkflowOptions:
    def test_create_opt(self):
        # default attributes are correct
        a = A()
        assert a.alice == 1

        # option attributes can be updated
        a.alice = 2
        assert a.alice == 2

    def test_init_non_existing_fields(self):
        # Creating option with non-existing attributes will raise errors
        with pytest.raises(ValidationError):
            WorkflowOptions(non_existing=1)

    def test_task_options(self):
        a = A()
        a.task_options = {"task1": TaskOptions()}
        assert a.task_options["task1"] == TaskOptions()

    def test_to_dict(self):
        class B(TaskOptions):
            bob: int = 2

        class C(WorkflowOptions):
            charlie: int = 3

        a = A()
        a.task_options = {"b": B(), "c": C()}
        saved_a = a.to_dict()
        assert saved_a == {
            "alice": 1,
            "task_options": {"b": {"bob": 2}, "c": {"charlie": 3, "task_options": {}}},
        }
