"""Tests for laboneq_applications.logbook.folder_store."""

from __future__ import annotations

import datetime
import json
import logging

import numpy as np
import PIL
import pytest
from freezegun import freeze_time

from laboneq_applications.logbook.folder_store import FolderStore
from laboneq_applications.workflow import (
    WorkflowOptions,
    comment,
    log,
    save_artifact,
    task,
    workflow,
)


@workflow(name="my empty workflow")
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


class FolderStoreFixture:
    def __init__(self, store_path):
        self._store_path = store_path
        self.logstore = FolderStore(self._store_path)

    def store_contents(self) -> list[str]:
        store_path = self._store_path
        return sorted(str(p.relative_to(store_path)) for p in store_path.iterdir())

    def contents(self, workflow_folder):
        workflow_path = self._store_path / workflow_folder
        return sorted(
            str(p.relative_to(workflow_path)) for p in workflow_path.iterdir()
        )

    def log(self, workflow_folder):
        log_path = self._store_path / workflow_folder / "log.jsonl"
        with log_path.open() as f:
            return [json.loads(line) for line in f.readlines()]


@pytest.fixture()
def logstore(tmp_path):
    return FolderStore(tmp_path / "store")


@pytest.fixture()
def folder(tmp_path):
    return FolderStoreFixture(tmp_path / "store")


TIMEDICTENTRY = {"time": "2024-07-28 17:55:00+00:00"}


@freeze_time("2024-07-28 17:55:00", tz_offset=0)
class TestFolderStore:
    def test_on_start_and_end(self, logstore, folder):
        wf = empty_workflow(3, 5, options={"logstore": logstore})
        wf.run()

        workflow_folder_name = folder.store_contents()[0]
        assert workflow_folder_name.startswith("20240728T")
        assert workflow_folder_name.endswith("my empty workflow")

        assert folder.contents(workflow_folder_name) == ["log.jsonl"]
        assert folder.log(workflow_folder_name) == [
            {"event": "start", "workflow": "my empty workflow"} | TIMEDICTENTRY,
            {"event": "end", "workflow": "my empty workflow"} | TIMEDICTENTRY,
        ]

    def test_on_error(self, logstore, folder):
        wf = bad_ref_workflow(3, 5, options={"logstore": logstore})

        with pytest.raises(AttributeError) as err:
            wf.run()
        assert str(err.value) == "'int' object has no attribute 'c'"

        workflow_folder_name = folder.store_contents()[0]
        assert workflow_folder_name.startswith("20240728T")
        assert workflow_folder_name.endswith("-bad-ref-workflow")

        assert folder.contents(workflow_folder_name) == ["log.jsonl"]
        assert folder.log(workflow_folder_name) == [
            {"event": "start", "workflow": "bad_ref_workflow"} | TIMEDICTENTRY,
            {
                "event": "error",
                "workflow": "bad_ref_workflow",
                "error": "AttributeError(\"'int' object has no attribute 'c'\")",
            }
            | TIMEDICTENTRY,
            {"event": "end", "workflow": "bad_ref_workflow"} | TIMEDICTENTRY,
        ]

    def test_on_task_start_and_end(self, logstore, folder):
        wf = simple_workflow(3, 5, options={"logstore": logstore})
        wf.run()

        workflow_folder_name = folder.store_contents()[0]
        assert workflow_folder_name.startswith("20240728T")
        assert workflow_folder_name.endswith("-simple-workflow")

        assert folder.contents(workflow_folder_name) == ["log.jsonl"]
        assert folder.log(workflow_folder_name) == [
            {"event": "start", "workflow": "simple_workflow"} | TIMEDICTENTRY,
            {
                "event": "task_start",
                "task": "add_task",
            }
            | TIMEDICTENTRY,
            {
                "event": "task_end",
                "task": "add_task",
            }
            | TIMEDICTENTRY,
            {"event": "end", "workflow": "simple_workflow"} | TIMEDICTENTRY,
        ]

    def test_on_task_error(self, logstore, folder):
        wf = error_workflow(options={"logstore": logstore})
        with pytest.raises(ValueError) as err:
            wf.run()

        assert str(err.value) == "This is not a happy task."
        workflow_folder_name = folder.store_contents()[0]
        assert workflow_folder_name.startswith("20240728T")
        assert workflow_folder_name.endswith("-error-workflow")

        assert folder.contents(workflow_folder_name) == ["log.jsonl"]
        assert folder.log(workflow_folder_name) == [
            {"event": "start", "workflow": "error_workflow"} | TIMEDICTENTRY,
            {
                "event": "task_start",
                "task": "error_task",
            }
            | TIMEDICTENTRY,
            {
                "event": "task_error",
                "task": "error_task",
                "error": "ValueError('This is not a happy task.')",
            }
            | TIMEDICTENTRY,
            {
                "event": "task_end",
                "task": "error_task",
            }
            | TIMEDICTENTRY,
            {"event": "end", "workflow": "error_workflow"} | TIMEDICTENTRY,
        ]

    def test_comment(self, logstore, folder):
        wf = comment_workflow("A comment!", options={"logstore": logstore})
        wf.run()

        workflow_folder_name = folder.store_contents()[0]
        assert workflow_folder_name.startswith("20240728T")
        assert workflow_folder_name.endswith("-comment-workflow")

        assert folder.contents(workflow_folder_name) == ["log.jsonl"]
        assert folder.log(workflow_folder_name) == [
            {"event": "start", "workflow": "comment_workflow"} | TIMEDICTENTRY,
            {
                "event": "task_start",
                "task": "comment_task",
            }
            | TIMEDICTENTRY,
            {
                "event": "comment",
                "message": "A comment!",
            }
            | TIMEDICTENTRY,
            {
                "event": "task_end",
                "task": "comment_task",
            }
            | TIMEDICTENTRY,
            {"event": "end", "workflow": "comment_workflow"} | TIMEDICTENTRY,
        ]

    def test_log(self, logstore, folder):
        thing = datetime.datetime.now()  # noqa: DTZ005
        thing_str = str(thing)
        wf = log_workflow(thing, options={"logstore": logstore})
        wf.run()

        workflow_folder_name = folder.store_contents()[0]
        assert workflow_folder_name.startswith("20240728T")
        assert workflow_folder_name.endswith("-log-workflow")

        assert folder.contents(workflow_folder_name) == ["log.jsonl"]
        assert folder.log(workflow_folder_name) == [
            {"event": "start", "workflow": "log_workflow"} | TIMEDICTENTRY,
            {
                "event": "task_start",
                "task": "log_task",
            }
            | TIMEDICTENTRY,
            {
                "event": "log",
                "message": f"a {thing_str}, b (1, 2), c 5",
                "level": logging.ERROR,
            }
            | TIMEDICTENTRY,
            {
                "event": "task_end",
                "task": "log_task",
            }
            | TIMEDICTENTRY,
            {"event": "end", "workflow": "log_workflow"} | TIMEDICTENTRY,
        ]

    def test_save(self, logstore, folder):
        obj = np.ndarray([1, 2, 3])

        wf = save_workflow(
            "an_obj",
            obj,
            metadata={"created_at": "nowish"},
            opts=None,
            options={"logstore": logstore},
        )
        wf.run()

        workflow_folder_name = folder.store_contents()[0]
        assert workflow_folder_name.startswith("20240728T")
        assert workflow_folder_name.endswith("-save-workflow")

        assert folder.contents(workflow_folder_name) == ["an-obj.npy", "log.jsonl"]
        assert folder.log(workflow_folder_name) == [
            {"event": "start", "workflow": "save_workflow"} | TIMEDICTENTRY,
            {"event": "task_start", "task": "save_task"} | TIMEDICTENTRY,
            {
                "event": "artifact",
                "artifact_name": "an_obj",
                "artifact_type": "ndarray",
                "artifact_metadata": {
                    "created_at": "nowish",
                },
                "artifact_options": {},
                "artifact_files": [
                    {"filename": "an-obj.npy"},
                ],
            }
            | TIMEDICTENTRY,
            {"event": "task_end", "task": "save_task"} | TIMEDICTENTRY,
            {"event": "end", "workflow": "save_workflow"} | TIMEDICTENTRY,
        ]

    def test_save_with_popped_options(self, logstore, folder):
        im = PIL.Image.effect_mandelbrot((10, 10), (-2.0, -1.5, 1.5, 1.5), 9)

        wf = save_workflow(
            "image",
            im,
            metadata={"created_at": "nowish"},
            opts={"format": "jpg"},
            options={"logstore": logstore},
        )
        wf.run()

        workflow_folder_name = folder.store_contents()[0]
        assert workflow_folder_name.startswith("20240728T")
        assert workflow_folder_name.endswith("-save-workflow")

        assert folder.contents(workflow_folder_name) == ["image.jpg", "log.jsonl"]
        assert folder.log(workflow_folder_name) == [
            {"event": "start", "workflow": "save_workflow"} | TIMEDICTENTRY,
            {"event": "task_start", "task": "save_task"} | TIMEDICTENTRY,
            {
                "event": "artifact",
                "artifact_name": "image",
                "artifact_type": "Image",
                "artifact_metadata": {
                    "created_at": "nowish",
                },
                "artifact_options": {
                    "format": "jpg",
                },
                "artifact_files": [
                    {"filename": "image.jpg"},
                ],
            }
            | TIMEDICTENTRY,
            {"event": "task_end", "task": "save_task"} | TIMEDICTENTRY,
            {"event": "end", "workflow": "save_workflow"} | TIMEDICTENTRY,
        ]
