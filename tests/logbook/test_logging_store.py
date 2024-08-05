"""Tests for laboneq_applications.logbook.logging_store."""

import logging

import pytest

from laboneq_applications import logbook
from laboneq_applications.logbook.logging_store import LoggingStore
from laboneq_applications.workflow import task
from laboneq_applications.workflow.engine import workflow


@workflow
def empty_workflow(a, b):
    pass


@task
def add_task(a, b):
    return a + b


@workflow
def simple_workflow(a, b):
    return add_task(a, b)


@task
def error_task():
    raise ValueError("This is not a happy task.")


@workflow
def error_workflow():
    error_task()


@workflow
def bad_ref_workflow(a, b):
    return add_task(a, b.c)


@task
def comment_task(a):
    logbook.comment(a)


@workflow
def comment_workflow(a):
    comment_task(a)


@pytest.fixture()
def logstore(caplog):
    caplog.set_level(logging.INFO)
    return LoggingStore()


class TestLoggingStore:
    def test_on_start_and_end(self, caplog, logstore):
        wf = empty_workflow(3, 5)
        wf.run(logstore=logstore)

        assert caplog.messages == [
            "Workflow execution started",
            "Workflow execution ended",
        ]

    def test_on_error(self, caplog, logstore):
        wf = bad_ref_workflow(3, 5)

        with pytest.raises(AttributeError) as err:
            wf.run(logstore=logstore)
        assert str(err.value) == "'int' object has no attribute 'c'"

        assert caplog.messages == [
            "Workflow execution started",
            "Workflow execution failed with:"
            " AttributeError(\"'int' object has no attribute 'c'\")",
            "Workflow execution ended",
        ]

    def test_on_task_start_and_end(self, caplog, logstore):
        wf = simple_workflow(3, 5)
        wf.run(logstore=logstore)

        assert caplog.messages == [
            "Workflow execution started",
            "Task add_task started",
            "Task add_task ended",
            "Workflow execution ended",
        ]

    def test_on_task_error(self, caplog, logstore):
        wf = error_workflow()
        with pytest.raises(ValueError) as err:
            wf.run(logstore=logstore)

        assert str(err.value) == "This is not a happy task."
        assert caplog.messages == [
            "Workflow execution started",
            "Task error_task started",
            "Task error_task failed with: ValueError('This is not a happy task.')",
            "Task error_task ended",
            "Workflow execution ended",
        ]

    def test_comment(self, caplog, logstore):
        wf = comment_workflow("A comment!")
        wf.run(logstore=logstore)

        assert caplog.messages == [
            "Workflow execution started",
            "Task comment_task started",
            "A comment!",
            "Task comment_task ended",
            "Workflow execution ended",
        ]
