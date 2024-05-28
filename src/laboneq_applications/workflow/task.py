"""Tasks used within workflows."""

from __future__ import annotations

import abc
import inspect
import textwrap
from typing import Any, Callable, TypeVar, overload

from laboneq_applications.workflow._context import get_active_context


class Task(abc.ABC):
    """A base class for a workflow task.

    Classes that subclass this class must implement:
        - `_run()`
    """

    def __init__(self, name: str):
        self._name: str = name

    def __repr__(self):
        return f"Task(name={self.name})"

    @property
    def name(self) -> str:
        """The name of the task."""
        return self._name

    @property
    def src(self) -> str:
        """Source code of the task."""
        src = inspect.getsource(self._run)
        return textwrap.dedent(src)

    @abc.abstractmethod
    def _run(self, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        """Run the task."""

    def run(self, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        """Run the task.

        The behaviour of the task depends on the context it is executed.
        The behaviour is unchanged when no context is active.
        """
        ctx = get_active_context()
        if ctx is None:
            return self._run(*args, **kwargs)
        return ctx.execute_task(self, *args, **kwargs)


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

    def __call__(self, *args: object, **kwargs: object) -> T:
        """Run the task."""
        return self.run(*args, **kwargs)

    def _run(self, *args: object, **kwargs: object) -> T:
        """Run the task."""
        return self._func(*args, **kwargs)


TaskFunction = TypeVar("TaskFunction", bound=Callable)


@overload
def task(func: TaskFunction, *, name: str | None = None) -> TaskFunction: ...


def task(func: TaskFunction | None = None, *, name: str | None = None):  # noqa: D417
    """Mark a function as a task.

    If the decorated function is used outside of an workflow context, or
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
