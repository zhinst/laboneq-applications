"""Tests for laboneq_applications.logbook.combined_store."""

from __future__ import annotations

import logging

import pytest

from laboneq_applications import logbook
from laboneq_applications.logbook.combined_store import CombinedStore
from laboneq_applications.logbook.logging_store import LoggingStore
from laboneq_applications.workflow import WorkflowOptions, task, workflow


@workflow
def empty_workflow(a, b, options: WorkflowOptions | None = None):
    pass


@task
def add_task(a, b):
    return a + b


@workflow
def simple_workflow(a, b, options: WorkflowOptions | None = None):
    return add_task(a, b)


@task
def error_task():
    raise ValueError("This is not a happy task.")


@workflow
def error_workflow(options: WorkflowOptions | None = None):
    error_task()


@workflow
def bad_ref_workflow(a, b, options: WorkflowOptions | None = None):
    return add_task(a, b.c)


@task
def comment_task(a):
    logbook.comment(a)


@workflow
def comment_workflow(a, options: WorkflowOptions | None = None):
    comment_task(a)


@task
def save_task(name, obj, metadata, opts):
    logbook.save_artifact(name, obj, metadata=metadata, options=opts)


@workflow
def save_workflow(name, obj, metadata, opts, options: WorkflowOptions | None = None):
    save_task(name, obj, metadata, opts)


class TestCombinedStore:
    @pytest.fixture()
    def logstore(self, caplog):
        caplog.set_level(logging.INFO)
        return self._combined_store(caplog)

    def _combined_store(self, caplog):
        stores = []
        for name in ["logger_a", "logger_b"]:
            logger = logging.Logger(name)  # noqa: LOG001
            logger.addHandler(caplog.handler)
            stores.append(LoggingStore(logger, rich=False))

        return CombinedStore(stores)

    def _messages(self, logger_name, caplog):
        return [r.getMessage() for r in caplog.records if r.name == logger_name]

    def test_on_start_and_end(self, caplog, logstore):
        wf = empty_workflow(3, 5, options={"logstore": logstore})
        wf.run()

        for logger_name in ["logger_a", "logger_b"]:
            assert self._messages(logger_name, caplog) == [
                "Workflow 'empty_workflow': execution started",
                "Workflow 'empty_workflow': execution ended",
            ]

    def test_on_error(self, caplog, logstore):
        wf = bad_ref_workflow(3, 5, options={"logstore": logstore})

        with pytest.raises(AttributeError) as err:
            wf.run()
        assert str(err.value) == "'int' object has no attribute 'c'"

        for logger_name in ["logger_a", "logger_b"]:
            assert self._messages(logger_name, caplog) == [
                "Workflow 'bad_ref_workflow': execution started",
                "Workflow 'bad_ref_workflow': execution failed with:"
                " AttributeError(\"'int' object has no attribute 'c'\")",
                "Workflow 'bad_ref_workflow': execution ended",
            ]

    def test_on_task_start_and_end(self, caplog, logstore):
        wf = simple_workflow(3, 5, options={"logstore": logstore})
        wf.run()

        for logger_name in ["logger_a", "logger_b"]:
            assert self._messages(logger_name, caplog) == [
                "Workflow 'simple_workflow': execution started",
                "Task 'add_task': started",
                "Task 'add_task': ended",
                "Workflow 'simple_workflow': execution ended",
            ]

    def test_on_task_error(self, caplog, logstore):
        wf = error_workflow(options={"logstore": logstore})
        with pytest.raises(ValueError) as err:
            wf.run()

        assert str(err.value) == "This is not a happy task."
        for logger_name in ["logger_a", "logger_b"]:
            assert self._messages(logger_name, caplog) == [
                "Workflow 'error_workflow': execution started",
                "Task 'error_task': started",
                "Task 'error_task': failed with:"
                " ValueError('This is not a happy task.')",
                "Task 'error_task': ended",
                "Workflow 'error_workflow': execution ended",
            ]

    def test_comment(self, caplog, logstore):
        wf = comment_workflow("A comment!", options={"logstore": logstore})
        wf.run()

        for logger_name in ["logger_a", "logger_b"]:
            assert self._messages(logger_name, caplog) == [
                "Workflow 'comment_workflow': execution started",
                "Task 'comment_task': started",
                "Comment: A comment!",
                "Task 'comment_task': ended",
                "Workflow 'comment_workflow': execution ended",
            ]

    def test_save(self, caplog, logstore):
        class DummyObj:
            pass

        obj = DummyObj()

        wf = save_workflow(
            "an_obj",
            obj,
            metadata={"created_at": "nowish"},
            opts=None,
            options={"logstore": logstore},
        )
        wf.run()

        for logger_name in ["logger_a", "logger_b"]:
            assert self._messages(logger_name, caplog) == [
                "Workflow 'save_workflow': execution started",
                "Task 'save_task': started",
                "Artifact: 'an_obj' of type 'DummyObj' logged",
                "Task 'save_task': ended",
                "Workflow 'save_workflow': execution ended",
            ]
