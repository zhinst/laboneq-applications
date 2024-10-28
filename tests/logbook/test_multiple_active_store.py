"""Tests for multiple active logbook stores in workflow."""

from __future__ import annotations

import datetime
import logging

import pytest
from freezegun import freeze_time

from laboneq_applications.logbook.logging_store import LoggingStore
from laboneq_applications.workflow import (
    WorkflowOptions,
    comment,
    log,
    save_artifact,
    task,
    workflow,
)


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
    comment(a)


@workflow
def comment_workflow(a, options: WorkflowOptions | None = None):
    comment_task(a)


@task
def log_task(a: object):
    log(logging.ERROR, "a %s, b %s, c %d", a, (1, 2), 5.7)


@workflow
def log_workflow(a, options: WorkflowOptions | None = None):
    log_task(a)


@task
def save_task(name, obj, metadata, opts):
    save_artifact(name, obj, metadata=metadata, options=opts)


@workflow
def save_workflow(name, obj, metadata, opts, options: WorkflowOptions | None = None):
    save_task(name, obj, metadata, opts)


TIMESTR = "2015-10-21 09:00:00.123456Z"


@freeze_time("2015-10-21 09:00:00.123456", tz_offset=0)
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
        return stores

    def _messages(self, logger_name, caplog):
        return [r.getMessage() for r in caplog.records if r.name == logger_name]

    def test_on_start_and_end(self, caplog, logstore):
        wf = empty_workflow(3, 5, options={"logstore": logstore})
        wf.run()

        for logger_name in ["logger_a", "logger_b"]:
            assert self._messages(logger_name, caplog) == [
                f"Workflow 'empty_workflow': execution started at {TIMESTR}",
                f"Workflow 'empty_workflow': execution ended at {TIMESTR}",
            ]

    def test_on_error(self, caplog, logstore):
        wf = bad_ref_workflow(3, 5, options={"logstore": logstore})

        with pytest.raises(AttributeError) as err:
            wf.run()
        assert str(err.value) == "'int' object has no attribute 'c'"

        for logger_name in ["logger_a", "logger_b"]:
            assert self._messages(logger_name, caplog) == [
                f"Workflow 'bad_ref_workflow': execution started at {TIMESTR}",
                f"Workflow 'bad_ref_workflow': execution failed at {TIMESTR} with:"
                f" AttributeError(\"'int' object has no attribute 'c'\")",
                f"Workflow 'bad_ref_workflow': execution ended at {TIMESTR}",
            ]

    def test_on_task_start_and_end(self, caplog, logstore):
        wf = simple_workflow(3, 5, options={"logstore": logstore})
        wf.run()

        for logger_name in ["logger_a", "logger_b"]:
            assert self._messages(logger_name, caplog) == [
                f"Workflow 'simple_workflow': execution started at {TIMESTR}",
                f"Task 'add_task': started at {TIMESTR}",
                f"Task 'add_task': ended at {TIMESTR}",
                f"Workflow 'simple_workflow': execution ended at {TIMESTR}",
            ]

    def test_on_task_error(self, caplog, logstore):
        wf = error_workflow(options={"logstore": logstore})
        with pytest.raises(ValueError) as err:
            wf.run()

        assert str(err.value) == "This is not a happy task."
        for logger_name in ["logger_a", "logger_b"]:
            assert self._messages(logger_name, caplog) == [
                f"Workflow 'error_workflow': execution started at {TIMESTR}",
                f"Task 'error_task': started at {TIMESTR}",
                f"Task 'error_task': failed at {TIMESTR} with:"
                f" ValueError('This is not a happy task.')",
                f"Task 'error_task': ended at {TIMESTR}",
                f"Workflow 'error_workflow': execution ended at {TIMESTR}",
            ]

    def test_comment(self, caplog, logstore):
        wf = comment_workflow("A comment!", options={"logstore": logstore})
        wf.run()

        for logger_name in ["logger_a", "logger_b"]:
            assert self._messages(logger_name, caplog) == [
                f"Workflow 'comment_workflow': execution started at {TIMESTR}",
                f"Task 'comment_task': started at {TIMESTR}",
                "Comment: A comment!",
                f"Task 'comment_task': ended at {TIMESTR}",
                f"Workflow 'comment_workflow': execution ended at {TIMESTR}",
            ]

    def test_log(self, caplog, logstore):
        thing = datetime.datetime.now()  # noqa: DTZ005
        thing_str = str(thing)
        wf = log_workflow(thing, options={"logstore": logstore})
        wf.run()

        for logger_name in ["logger_a", "logger_b"]:
            assert self._messages(logger_name, caplog) == [
                f"Workflow 'log_workflow': execution started at {TIMESTR}",
                f"Task 'log_task': started at {TIMESTR}",
                f"a {thing_str}, b (1, 2), c 5",
                f"Task 'log_task': ended at {TIMESTR}",
                f"Workflow 'log_workflow': execution ended at {TIMESTR}",
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
                f"Workflow 'save_workflow': execution started at {TIMESTR}",
                f"Task 'save_task': started at {TIMESTR}",
                f"Artifact: 'an_obj' of type 'DummyObj' logged at {TIMESTR}",
                f"Task 'save_task': ended at {TIMESTR}",
                f"Workflow 'save_workflow': execution ended at {TIMESTR}",
            ]
