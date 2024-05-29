"""Tasks used within Workflow."""

from __future__ import annotations

import abc
import inspect
import textwrap
from functools import wraps
from typing import Any, Callable, TypeVar, overload

from laboneq_applications.workflow._context import LocalContext


def _wrapper(func: Callable) -> Callable:
    """Wrap a method.

    If called within the workflow context, produces an event instead of executing
    the wrapped method.
    """

    @wraps(func)
    def wrapped(self: Task, *args, **kwargs) -> Any:  # noqa: ANN401
        # TODO: Active context should know how to run task. This decorator not.
        from laboneq_applications.workflow.engine.task_block import TaskBlock

        if not LocalContext.is_active():
            return func(self, *args, **kwargs)
        blk = TaskBlock(self, *args, **kwargs)
        LocalContext.active_context().register(blk)
        return blk._promise

    return wrapped


class Task(abc.ABC):
    """A base class for a Workflow task.

    Classes that subclass this class must implement:
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

    @property
    def src(self) -> str:
        """Source code of the task."""
        src = inspect.getsource(self.run)
        return textwrap.dedent(src)

    @abc.abstractmethod
    def run(self, *args, **kwargs) -> Any:  # noqa: ANN401
        """Run the task.

        If used within `Workflow` context, creates an event of this task
        instead of executing it.
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

    @property
    def src(self) -> str:
        """Source code of the task."""
        src = inspect.getsource(self._func)
        return textwrap.dedent(src)

    def __call__(self, *args, **kwargs):
        """Run the task."""
        return self.run(*args, **kwargs)

    def run(self, *args, **kwargs) -> T:
        """Run the task."""
        return self._func(*args, **kwargs)


TaskFunction = TypeVar("TaskFunction", bound=Callable)


@overload
def task(func: TaskFunction, *, name: str | None = None) -> TaskFunction: ...


def task(func: TaskFunction | None = None, *, name: str | None = None):  # noqa: D417
    """Mark a function as a task.

    If the decorated function is used outside of Workflow context, or
    within another task, the underlying behaviour does not change.

    Arguments:
        name: Name of the task.
            Defaults to function name.

    Example:
        ```python
        from laboneq_applications.workflow import task

        @task
        def my_task(x, y):
            return x + y

        my_task(1, 1)
        ```
    """

    def wrapper(func):  # noqa: ANN001, ANN202
        return FunctionTask(func, name=name)

    return wrapper(func) if func else wrapper
