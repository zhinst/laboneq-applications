"""Tests for laboneq_applications.logbook.logging_store."""

from __future__ import annotations

import logging

import pytest
from freezegun import freeze_time

from laboneq_applications.logbook.logging_store import LoggingStore
from laboneq_applications.workflow import (
    WorkflowOptions,
    comment,
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
def save_task(name, obj, metadata, opts):
    save_artifact(name, obj, metadata=metadata, options=opts)


@workflow
def save_workflow(name, obj, metadata, opts, options: WorkflowOptions | None = None):
    save_task(name, obj, metadata, opts)


TIMESTR = "2015-10-21 09:00:00.123456Z"


@freeze_time(TIMESTR, tz_offset=0)
class TestLoggingStore:
    @pytest.fixture()
    def logstore(self, caplog):
        caplog.set_level(logging.INFO)
        return LoggingStore(rich=False)

    def test_on_start_and_end(self, caplog, logstore):
        wf = empty_workflow(3, 5, options={"logstore": logstore})
        wf.run()

        assert caplog.messages == [
            f"Workflow 'empty_workflow': execution started at {TIMESTR}",
            f"Workflow 'empty_workflow': execution ended at {TIMESTR}",
        ]

    def test_on_error(self, caplog, logstore):
        wf = bad_ref_workflow(3, 5, options={"logstore": logstore})

        with pytest.raises(AttributeError) as err:
            wf.run()
        assert str(err.value) == "'int' object has no attribute 'c'"

        assert caplog.messages == [
            f"Workflow 'bad_ref_workflow': execution started at {TIMESTR}",
            f"Workflow 'bad_ref_workflow': execution failed at {TIMESTR} with:"
            " AttributeError(\"'int' object has no attribute 'c'\")",
            f"Workflow 'bad_ref_workflow': execution ended at {TIMESTR}",
        ]

    def test_on_task_start_and_end(self, caplog, logstore):
        wf = simple_workflow(3, 5, options={"logstore": logstore})
        wf.run()

        assert caplog.messages == [
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
        assert caplog.messages == [
            f"Workflow 'error_workflow': execution started at {TIMESTR}",
            f"Task 'error_task': started at {TIMESTR}",
            f"Task 'error_task': failed at {TIMESTR} with: "
            "ValueError('This is not a happy task.')",
            f"Task 'error_task': ended at {TIMESTR}",
            f"Workflow 'error_workflow': execution ended at {TIMESTR}",
        ]

    def test_comment(self, caplog, logstore):
        wf = comment_workflow("A comment!", options={"logstore": logstore})
        wf.run()

        assert caplog.messages == [
            f"Workflow 'comment_workflow': execution started at {TIMESTR}",
            f"Task 'comment_task': started at {TIMESTR}",
            "Comment: A comment!",
            f"Task 'comment_task': ended at {TIMESTR}",
            f"Workflow 'comment_workflow': execution ended at {TIMESTR}",
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

        assert caplog.messages == [
            f"Workflow 'save_workflow': execution started at {TIMESTR}",
            f"Task 'save_task': started at {TIMESTR}",
            f"Artifact: 'an_obj' of type 'DummyObj' logged at {TIMESTR}",
            f"Task 'save_task': ended at {TIMESTR}",
            f"Workflow 'save_workflow': execution ended at {TIMESTR}",
        ]


class pad:  # noqa: N801
    """Pad to the given width using the second last character."""

    def __init__(self, width):
        self.width = width

    def __rmatmul__(self, other):
        return other[:-1] + other[-2] * (self.width - len(other)) + other[-1]


@freeze_time(TIMESTR, tz_offset=0)
class TestRichLoggingStore:
    @pytest.fixture()
    def logstore(self, caplog):
        caplog.set_level(logging.INFO)
        return LoggingStore(rich=True)

    def test_on_start_and_end(self, caplog, logstore):
        wf = empty_workflow(3, 5, options={"logstore": logstore})
        wf.run()

        assert caplog.messages == [
            " ──────────────────────────────────────────────── " @ pad(80),
            f"  Workflow 'empty_workflow': execution started at {TIMESTR}  " @ pad(80),
            " ──────────────────────────────────────────────── " @ pad(80),
            " ──────────────────────────────────────────────── " @ pad(80),
            f"  Workflow 'empty_workflow': execution ended at {TIMESTR}  " @ pad(80),
            " ──────────────────────────────────────────────── " @ pad(80),
        ]

    def test_on_error(self, caplog, logstore):
        wf = bad_ref_workflow(3, 5, options={"logstore": logstore})

        with pytest.raises(AttributeError) as err:
            wf.run()

        assert str(err.value) == "'int' object has no attribute 'c'"
        assert caplog.messages == [
            " ──────────────────────────────────────────────── " @ pad(80),
            "  Workflow 'bad_ref_workflow': execution started at 2015-10-21  "
            @ pad(80),
            "  09:00:00.123456Z  " @ pad(80),
            " ──────────────────────────────────────────────── " @ pad(80),
            f"Workflow 'bad_ref_workflow': execution failed at {TIMESTR} ",
            "with: AttributeError(\"'int' object has no attribute 'c'\")",
            " ──────────────────────────────────────────────── " @ pad(80),
            f"  Workflow 'bad_ref_workflow': execution ended at {TIMESTR}  " @ pad(80),
            " ──────────────────────────────────────────────── " @ pad(80),
        ]

    def test_on_task_start_and_end(self, caplog, logstore):
        wf = simple_workflow(3, 5, options={"logstore": logstore})
        wf.run()

        assert caplog.messages == [
            " ──────────────────────────────────────────────── " @ pad(80),
            f"  Workflow 'simple_workflow': execution started at {TIMESTR}  " @ pad(80),
            " ──────────────────────────────────────────────── " @ pad(80),
            f"Task 'add_task': started at {TIMESTR}",
            f"Task 'add_task': ended at {TIMESTR}",
            " ──────────────────────────────────────────────── " @ pad(80),
            f"  Workflow 'simple_workflow': execution ended at {TIMESTR}  " @ pad(80),
            " ──────────────────────────────────────────────── " @ pad(80),
        ]

    def test_on_task_error(self, caplog, logstore):
        wf = error_workflow(options={"logstore": logstore})
        with pytest.raises(ValueError) as err:
            wf.run()

        assert str(err.value) == "This is not a happy task."
        assert caplog.messages == [
            " ──────────────────────────────────────────────── " @ pad(80),
            f"  Workflow 'error_workflow': execution started at {TIMESTR}  " @ pad(80),
            " ──────────────────────────────────────────────── " @ pad(80),
            f"Task 'error_task': started at {TIMESTR}",
            f"Task 'error_task': failed at {TIMESTR} with: ValueError('This ",
            "is not a happy task.')",
            f"Task 'error_task': ended at {TIMESTR}",
            " ──────────────────────────────────────────────── " @ pad(80),
            f"  Workflow 'error_workflow': execution ended at {TIMESTR}  " @ pad(80),
            " ──────────────────────────────────────────────── " @ pad(80),
        ]

    def test_comment(self, caplog, logstore):
        wf = comment_workflow("A comment!", options={"logstore": logstore})
        wf.run()

        assert caplog.messages == [
            " ──────────────────────────────────────────────── " @ pad(80),
            "  Workflow 'comment_workflow': execution started at 2015-10-21  "
            @ pad(80),
            "  09:00:00.123456Z  " @ pad(80),
            " ──────────────────────────────────────────────── " @ pad(80),
            f"Task 'comment_task': started at {TIMESTR}",
            "Comment: A comment!",
            f"Task 'comment_task': ended at {TIMESTR}",
            " ──────────────────────────────────────────────── " @ pad(80),
            f"  Workflow 'comment_workflow': execution ended at {TIMESTR}  " @ pad(80),
            " ──────────────────────────────────────────────── " @ pad(80),
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

        assert caplog.messages == [
            " ──────────────────────────────────────────────── " @ pad(80),
            f"  Workflow 'save_workflow': execution started at {TIMESTR}  " @ pad(80),
            " ──────────────────────────────────────────────── " @ pad(80),
            f"Task 'save_task': started at {TIMESTR}",
            f"Artifact: 'an_obj' of type 'DummyObj' logged at {TIMESTR}",
            f"Task 'save_task': ended at {TIMESTR}",
            " ──────────────────────────────────────────────── " @ pad(80),
            f"  Workflow 'save_workflow': execution ended at {TIMESTR}  " @ pad(80),
            " ──────────────────────────────────────────────── " @ pad(80),
        ]
