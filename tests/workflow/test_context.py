import pytest

from laboneq_applications.workflow import task
from laboneq_applications.workflow._context import (
    ExecutorContext,
    LocalContext,
    get_active_context,
)


class TestLocalContext:
    def test_no_local_context(self):
        with pytest.raises(RuntimeError, match="No active context."):
            assert LocalContext().exit()
        assert LocalContext().active_context() is None

    def test_exit_in_scope(self):
        with pytest.raises(RuntimeError, match="No active context."):
            with LocalContext.scoped():
                LocalContext().exit()

        with pytest.raises(RuntimeError, match="No active context."):  # noqa: PT012
            with LocalContext.scoped():
                with LocalContext.scoped():
                    LocalContext().exit()

    def test_enter_exit(self):
        with LocalContext.scoped():
            assert LocalContext.is_active() is True
        assert LocalContext.is_active() is False

    def test_active_context(self):
        assert LocalContext.active_context() is None
        with LocalContext.scoped():
            assert LocalContext.active_context() is None
        assert LocalContext.active_context() is None

    def test_handler(self):
        handler = {"a": 5}
        with LocalContext.scoped(handler):
            # Nested scope
            with LocalContext.scoped():
                assert LocalContext.active_context() is None
            assert LocalContext.active_context() == handler


def test_get_active_context():
    assert get_active_context() is None
    with LocalContext.scoped(1):
        assert get_active_context() == 1
    assert get_active_context() is None


def test_executor_context():
    class TaskExecutor(ExecutorContext):
        results = []  # noqa: RUF012

        def execute_task(self, task, *args, **kwargs):
            self.results.append((task._run(*args, **kwargs), task.name))

    @task
    def mytask(x):
        return x + 1

    with LocalContext.scoped(TaskExecutor()):
        mytask(1)
        ctx = get_active_context()
        assert ctx.results == [(2, "mytask")]
