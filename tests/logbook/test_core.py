from __future__ import annotations

from datetime import datetime, timezone

from freezegun import freeze_time

from laboneq_applications.logbook import format_time
from laboneq_applications.workflow import (
    WorkflowOptions,
    execution_info,
    task,
    workflow,
)


@task
def workflow_name_task():
    info = execution_info()
    if info is None:
        return None
    return info.workflows


@task
def workflow_start_time_task():
    info = execution_info()
    if info is None:
        return None
    return info.start_time


@workflow
def logdir_workflow_inner(options: WorkflowOptions | None = None):
    workflow_name_task()
    workflow_start_time_task()


@workflow
def logdir_workflow(options: WorkflowOptions | None = None):
    return logdir_workflow_inner()


def test_format_time():
    ts = datetime(2000, 1, 1, 15, tzinfo=timezone.utc)
    assert format_time(ts) == "2000-01-01 15:00:00.000000Z"


@freeze_time("2024-07-28 17:55:00", tz_offset=0)
def test_workflow_names_and_timestamp():
    task_result = workflow_name_task()
    assert task_result is None
    task_result = workflow_start_time_task()
    assert task_result is None

    wf = logdir_workflow()
    result = wf.run()
    assert result.tasks[0].output is None
    assert result.tasks[0].tasks[0].output == [
        "logdir_workflow",
        "logdir_workflow_inner",
    ]
    assert str(result.tasks[0].tasks[1].output) == "2024-07-28 17:55:00+00:00"
