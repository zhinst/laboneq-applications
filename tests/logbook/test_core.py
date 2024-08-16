"""Tests for laboneq_applications.logbook.core."""

from __future__ import annotations

import pytest

from laboneq_applications.logbook import comment, save_artifact
from laboneq_applications.workflow import workflow


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
