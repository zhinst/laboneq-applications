"""Tasks used within workflows."""

from __future__ import annotations

import abc
import inspect
import textwrap
from functools import partial, update_wrapper
from typing import Any, Callable, Generic, TypeVar, overload

from typing_extensions import ParamSpec

from laboneq_applications.workflow._context import get_active_context


class _BaseTask(abc.ABC):
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


T = ParamSpec("T")
B = TypeVar("B")


class task_(Generic[T, B], _BaseTask):  # noqa: N801
    """A task that wraps a Python function.

    Arguments:
        func: Function to be called as a task.
        name: Optional name of the task.
            If `None`, the name of the function is used.
    """

    def __init__(
        self,
        func: Callable[T, B],
        name: str | None = None,
    ) -> None:
        super().__init__(name if name is not None else func.__name__)
        self._func = func
        self.__doc__ = func.__doc__

    @property
    def src(self) -> str:
        """Source code of the task."""
        src = inspect.getsource(self._func)
        return textwrap.dedent(src)

    @property
    def func(self) -> Callable:
        """Underlying Python function."""
        return self._func

    def __call__(self, *args: T.args, **kwargs: T.kwargs) -> B:  # noqa: D102
        return self.run(*args, **kwargs)

    def _run(self, *args: T.args, **kwargs: T.kwargs) -> B:
        return self._func(*args, **kwargs)


@overload
def task(func: Callable[T, B], name: str) -> task_[T, B]: ...


@overload
def task(func: Callable[T, B]) -> task_[T, B]: ...


@overload
def task(
    name: str | None = None,
) -> Callable[[Callable[T, B]], task_[T, B]]: ...


def task(func: Callable[T, B] | None = None, name: str | None = None):
    """Mark a function as a task.

    If the decorated function is used outside of an workflow context, or
    within another task, the underlying behaviour does not change.

    Arguments:
        func: Function to be wrapped as a task.
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
    if func is None:
        return partial(task_, name=name)
    if isinstance(func, task_):
        return update_wrapper(
            task_(func=func.func, name=name),
            func.func,
            assigned=(
                "__module__",
                "__name__",
                "__qualname__",
                "__annotations__",
                "__type_params__",
                "__doc__",
            ),
        )
    return update_wrapper(
        task_(func=func, name=name),
        func,
        assigned=(
            "__module__",
            "__name__",
            "__qualname__",
            "__annotations__",
            "__type_params__",
            "__doc__",
        ),
    )
