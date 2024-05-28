from __future__ import annotations

import abc
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Generator

    from laboneq_applications.workflow.task import Task


class ExecutorContext(abc.ABC):
    """A base class for executor context."""

    @abc.abstractmethod
    def execute_task(  # noqa: ANN202
        self,
        task: Task,
        *args: object,
        **kwargs: object,
    ):
        """Run a task.

        Arguments:
            task: The task instance.
            *args: `task` arguments.
            **kwargs: `task` keyword arguments.

        Subclasses executing the `task` must use `task._run()`
        """


class _ContextStorage(threading.local):
    # NOTE: Subclassed for type hinting
    active: ClassVar[list[ExecutorContext]] = []


_contexts = _ContextStorage()


class LocalContext:
    """Local context."""

    @classmethod
    @contextmanager
    def scoped(cls, obj: ExecutorContext | None = None) -> Generator:
        cls.enter(obj)
        try:
            yield
        finally:
            cls.exit()

    @classmethod
    def enter(cls, obj: ExecutorContext | None = None) -> None:
        _contexts.active.append(obj)

    @classmethod
    def exit(cls) -> ExecutorContext:
        if _contexts.active:
            return _contexts.active.pop()
        raise RuntimeError("No active context.")

    @classmethod
    def is_active(cls) -> bool:
        return len(_contexts.active) != 0

    @classmethod
    def active_context(cls) -> ExecutorContext | None:
        if _contexts.active:
            return _contexts.active[-1]
        return None


def get_active_context() -> ExecutorContext | None:
    if _contexts.active:
        return _contexts.active[-1]
    return None
