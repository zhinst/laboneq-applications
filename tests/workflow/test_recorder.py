"""Tests for laboneq_applications.workflow.recorder."""

from __future__ import annotations

import pytest

from laboneq_applications.workflow import comment, log, save_artifact, workflow
from laboneq_applications.workflow.recorder import ExecutionRecorderManager


class TestComment:
    def test_comment_directly_in_workflow(self):
        @workflow
        def workflow_with_comment():
            comment("Hello!")

        with pytest.raises(RuntimeError) as err:
            workflow_with_comment()

        assert str(err.value) == (
            "Workflow comments are currently not supported outside of tasks."
        )

    def test_log_directly_in_workflow(self):
        @workflow
        def workflow_with_log():
            log(10, "Hello!")

        with pytest.raises(RuntimeError) as err:
            workflow_with_log()

        assert str(err.value) == (
            "Workflow log messages are currently not supported outside of tasks."
        )


class TestArtifactSaving:
    def test_comment_directly_in_workflow(self):
        @workflow
        def workflow_with_save():
            save_artifact("Object", object())

        with pytest.raises(RuntimeError) as err:
            workflow_with_save()

        assert str(err.value) == (
            "Workflow artifact saving is currently not supported outside of tasks."
        )


class MockRecorder:
    n_error_calls = 0

    def on_error(self, *args, **kwargs):  # noqa: ARG002
        self.n_error_calls += 1

    def on_task_error(self, *args, **kwargs):  # noqa: ARG002
        self.n_error_calls += 1


class TestExecutionRecorderManager:
    @pytest.mark.parametrize(
        ("func"),
        ["on_error", "on_task_error"],
    )
    def test_error_notification(self, func):
        rec = MockRecorder()
        recorder = ExecutionRecorderManager()
        recorder.add_recorder(rec)
        exc = KeyError()
        assert rec.n_error_calls == 0
        getattr(recorder, func)(None, error=exc)
        assert rec.n_error_calls == 1
        getattr(recorder, func)(None, error=exc)
        assert rec.n_error_calls == 1

        exc = IndexError()
        getattr(recorder, func)(None, error=exc)
        assert rec.n_error_calls == 2
