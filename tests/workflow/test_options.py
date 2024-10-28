from __future__ import annotations

import pytest

from laboneq_applications.logbook import LoggingStore
from laboneq_applications.workflow import (
    TaskOptions,
    WorkflowOptions,
    options,
)


@options
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
        with pytest.raises(TypeError):
            WorkflowOptions(non_existing=1)

    def test_task_options(self):
        a = A()
        a._task_options = {"task1": TaskOptions()}
        assert a._task_options["task1"] == TaskOptions()

    def test_to_dict(self):
        @options
        class B(TaskOptions):
            bob: int = 2

        @options
        class C(WorkflowOptions):
            charlie: int = 3

        a = A()
        a._task_options = {"b": B(), "c": C()}
        saved_a = a.to_dict()
        assert saved_a == {
            "alice": 1,
            "_task_options": {
                "b": {"bob": 2},
                "c": {"charlie": 3, "_task_options": {}},
            },
        }

    def test_logstore(self):
        obj = WorkflowOptions()
        assert obj.logstore is None

        obj = WorkflowOptions(logstore=[])
        assert obj.logstore == []

        store = LoggingStore()
        obj = WorkflowOptions(logstore=store)
        assert obj.logstore == [store]

        stores = [LoggingStore(), LoggingStore()]
        obj = WorkflowOptions(logstore=stores)
        assert obj.logstore == stores
