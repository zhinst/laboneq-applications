"""Test the FolderStore class."""

import json

import pytest
from freezegun import freeze_time

from laboneq_applications import logbook
from laboneq_applications.logbook.folder_store import FolderStore
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


class FolderStoreFixture:
    def __init__(self, store_path):
        self._store_path = store_path
        self.logstore = FolderStore(self._store_path)

    def store_contents(self):
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


@freeze_time("2024-07-28 17:55:00")
class TestFolderStore:
    def test_on_start_and_end(self, logstore, folder):
        wf = empty_workflow(3, 5)
        wf.run(logstore=logstore)

        workflow_folder_name = "20240728T175500-empty-workflow"

        assert folder.store_contents() == [workflow_folder_name]
        assert folder.contents(workflow_folder_name) == ["log.jsonl"]
        assert folder.log(workflow_folder_name) == [
            {"event": "start"},
            {"event": "end"},
        ]

    def test_on_error(self, logstore, folder):
        wf = bad_ref_workflow(3, 5)

        with pytest.raises(AttributeError) as err:
            wf.run(logstore=logstore)
        assert str(err.value) == "'int' object has no attribute 'c'"

        workflow_folder_name = "20240728T175500-bad-ref-workflow"

        assert folder.store_contents() == [workflow_folder_name]
        assert folder.contents(workflow_folder_name) == ["log.jsonl"]
        assert folder.log(workflow_folder_name) == [
            {"event": "start"},
            {
                "event": "error",
                "error": "AttributeError(\"'int' object has no attribute 'c'\")",
            },
            {"event": "end"},
        ]

    def test_on_task_start_and_end(self, logstore, folder):
        wf = simple_workflow(3, 5)
        wf.run(logstore=logstore)

        workflow_folder_name = "20240728T175500-simple-workflow"

        assert folder.store_contents() == [workflow_folder_name]
        assert folder.contents(workflow_folder_name) == ["log.jsonl"]
        assert folder.log(workflow_folder_name) == [
            {"event": "start"},
            {"event": "task_start", "task": "add_task"},
            {"event": "task_end", "task": "add_task"},
            {"event": "end"},
        ]

    def test_on_task_error(self, logstore, folder):
        wf = error_workflow()
        with pytest.raises(ValueError) as err:
            wf.run(logstore=logstore)

        workflow_folder_name = "20240728T175500-error-workflow"

        assert str(err.value) == "This is not a happy task."
        assert folder.store_contents() == [workflow_folder_name]
        assert folder.contents(workflow_folder_name) == ["log.jsonl"]
        assert folder.log(workflow_folder_name) == [
            {"event": "start"},
            {"event": "task_start", "task": "error_task"},
            {
                "event": "task_error",
                "task": "error_task",
                "error": "ValueError('This is not a happy task.')",
            },
            {"event": "task_end", "task": "error_task"},
            {"event": "end"},
        ]

    def test_comment(self, logstore, folder):
        wf = comment_workflow("A comment!")
        wf.run(logstore=logstore)

        workflow_folder_name = "20240728T175500-comment-workflow"

        assert folder.store_contents() == [workflow_folder_name]
        assert folder.contents(workflow_folder_name) == ["log.jsonl"]
        assert folder.log(workflow_folder_name) == [
            {"event": "start"},
            {"event": "task_start", "task": "comment_task"},
            {
                "event": "comment",
                "message": "A comment!",
            },
            {"event": "task_end", "task": "comment_task"},
            {"event": "end"},
        ]
