"""Tasks used within Workflow."""
from __future__ import annotations

import abc
from functools import wraps
from typing import Any, Callable, TypeVar, overload

from laboneq_applications.workflow._context import LocalContext
from laboneq_applications.workflow.promise import ReferencePromise
from laboneq_applications.workflow.resolver import ArgumentResolver


def _wrapper(func: Callable) -> Callable:
    """Wrap a method.

    If called within context, produces an event instead of executing.
    """

    @wraps(func)
    def wrapped(self: Task, *args, **kwargs) -> Any:  # noqa: ANN401
        if not LocalContext.is_active():
            return func(self, *args, **kwargs)
        event = TaskEvent(self, *args, **kwargs)
        if LocalContext.is_active():
            LocalContext.active_context().register(event)
        return event._promise

    return wrapped


class Task(abc.ABC):
    """A base class for a Workflow task.

    Required methods that subclasses need to define:

        - `run()`
    """

    def __init__(self, name: str):
        self._name: str = name

    def __init_subclass__(cls, *args, **kwargs):
        cls.run = _wrapper(cls.run)
        super().__init_subclass__(*args, **kwargs)

    def __repr__(self):
        return f"Task(name={self.name})"

    @property
    def name(self) -> str:
        """The name of the task."""
        return self._name

    @abc.abstractmethod
    def run(self, *args, **kwargs) -> Any:  # noqa: ANN401
        """Run the task.

        If used within `Workflow` context, creates an event of this task
        instead of executing the task.
        """


T = TypeVar("T")


class FunctionTask(Task):
    """A task that wraps a Python function.

    Arguments:
        func: Function to be called as a task.
        name: Optional name of the task.
            If `None`, the name of the function is used.
    """

    def __init__(
        self,
        func: Callable[..., T],
        name: str | None = None,
    ) -> None:
        super().__init__(name if name is not None else func.__name__)
        self._func = func

    def run(self, *args, **kwargs) -> T:
        """Run the task."""
        return self._func(*args, **kwargs)


TaskFunction = TypeVar("TaskFunction", bound=Callable)


@overload
def task(func: TaskFunction, *, name: str | None = None) -> TaskFunction:
    ...


def task(func: TaskFunction | None = None, *, name: str | None = None):  # noqa: D417
    """Mark a function as a task.

    If the decorated function is used outside of Workflow context, or
    within another task, the underlying behaviour does not change.

    Arguments:
        name: Name of the task.
            Defaults to function name.
    """

    def wrapper(func):  # noqa: ANN001, ANN202
        @wraps(func)
        def inner(*args, **kwargs):  # noqa: ANN202
            if not LocalContext.is_active():
                return func(*args, **kwargs)
            return FunctionTask(func, name=name).run(*args, **kwargs)

        return inner

    if func:
        return wrapper(func)
    return wrapper




class TaskEvent:
    """Task run event."""

    def __init__(self, task: Task, *args, **kwargs):
        self.task = task
        self.args = args
        self.kwargs = kwargs
        self._id: int = LocalContext.active_context().create_id()
        self._promise = ReferencePromise(self)
        self._resolver = ArgumentResolver(*self.args, **self.kwargs)

    @property
    def event_id(self) -> int:
        """Task event id."""
        return self._id

    @property
    def requires(self) -> list[TaskEvent]:
        """Tasks this task depends on."""
        return [
            t.ref for t in self._resolver.requires if isinstance(t, ReferencePromise)
        ]

    def execute(self) -> Any:  # noqa: ANN401
        """Execute the task.

        Returns:
            Task results.
        """
        args, kwargs = self._resolver.resolve()
        result = self.task.run(*args, **kwargs)
        self._promise.set_result(result)
        return result
