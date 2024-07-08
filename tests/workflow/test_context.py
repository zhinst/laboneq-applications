import pytest

from laboneq_applications.workflow import task
from laboneq_applications.workflow._context import (
    LocalContext,
    TaskExecutor,
    TaskExecutorContext,
)


class TestLocalContext:
    def test_no_local_context(self):
        with pytest.raises(RuntimeError, match="No active context."):
            assert LocalContext().exit()
        assert LocalContext.get_active() is None

    def test_exit_in_scope(self):
        with pytest.raises(RuntimeError, match="No active context."):
            with LocalContext.scoped():
                LocalContext().exit()

        with pytest.raises(RuntimeError, match="No active context."):  # noqa: PT012
            with LocalContext.scoped():
                with LocalContext.scoped():
                    LocalContext().exit()

    def test_active_context(self):
        assert LocalContext.get_active() is None
        with LocalContext.scoped():
            assert LocalContext.get_active() is None
        assert LocalContext.get_active() is None

    def test_handler(self):
        handler = {"a": 5}
        with LocalContext.scoped(handler):
            # Nested scope
            with LocalContext.scoped():
                assert LocalContext.get_active() is None
            assert LocalContext.get_active() == handler


def test_get_active_context():
    assert LocalContext.get_active() is None
    with LocalContext.scoped(1):
        assert LocalContext.get_active() == 1
    assert LocalContext.get_active() is None


def test_executor_context():
    class TaskExecutorTest(TaskExecutor):
        results = []  # noqa: RUF012

        def execute_task(self, task, *args, **kwargs):
            self.results.append((task._run(*args, **kwargs), task.name))

    @task
    def mytask(x): ...

    with TaskExecutorContext.scoped(TaskExecutorTest()):
        assert LocalContext.get_active() is None
        mytask(1)
        ctx = TaskExecutorContext.get_active()
        assert len(ctx.results) == 1
