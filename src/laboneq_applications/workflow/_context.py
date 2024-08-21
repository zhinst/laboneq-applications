from __future__ import annotations

import abc
import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

from laboneq_applications.workflow.engine.executor import ExecutorState

if TYPE_CHECKING:
    from collections.abc import Generator

    from laboneq_applications.workflow.task import task_


class _ContextStorage(threading.local):
    # NOTE: Subclassed for type hinting
    scopes: ClassVar[dict[str, list]] = defaultdict(list)


_contexts = _ContextStorage()


T = TypeVar("T")


class LocalContext(Generic[T]):
    """Local context."""

    _scope = "default"

    @classmethod
    @contextmanager
    def scoped(cls, obj: T | None = None) -> Generator:
        cls.enter(obj)
        try:
            yield
        finally:
            cls.exit()

    @classmethod
    def enter(cls, obj: T | None = None) -> None:
        _contexts.scopes[cls._scope].append(obj)

    @classmethod
    def exit(cls) -> T:
        try:
            return _contexts.scopes[cls._scope].pop()
        except (KeyError, IndexError) as error:
            raise RuntimeError("No active context.") from error

    @classmethod
    def get_active(cls) -> T | None:
        """Get an active context."""
        try:
            return _contexts.scopes[cls._scope][-1]
        except IndexError:
            return None


class TaskExecutor(abc.ABC):
    """A base class for task executor."""

    @abc.abstractmethod
    def execute_task(  # noqa: ANN202
        self,
        task: task_,
        *args: object,
        **kwargs: object,
    ):
        """Run a task.

        Arguments:
            task: The task instance.
            *args: `task` arguments.
            **kwargs: `task` keyword arguments.
        """


class TaskExecutorContext(LocalContext[TaskExecutor]):
    """Context for executing tasks."""

    _scope = "task_executor"


class ExecutorStateContext(LocalContext[ExecutorState]):
    """Context for workflow execution state."""

    _scope = "workflow_executor"
