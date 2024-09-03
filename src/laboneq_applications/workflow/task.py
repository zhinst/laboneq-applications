"""Tasks used within workflows."""

from __future__ import annotations

import inspect
import textwrap
from functools import partial, update_wrapper
from typing import TYPE_CHECKING, Callable, Generic, TypeVar, cast, overload

from typing_extensions import ParamSpec

from laboneq_applications.workflow import _context
from laboneq_applications.workflow.options import (
    BaseOptions,
    get_and_validate_param_type,
)

if TYPE_CHECKING:
    from datetime import datetime


class TaskResult:
    """Task result.

    The instance holds execution information of an task.
    """

    def __init__(
        self,
        task: task_,
        output: object,
        input: dict | None = None,  # noqa: A002
    ) -> None:
        self._task = task
        self._output = output
        self._input = input or {}
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None

    @property
    def name(self) -> str:
        """Task name."""
        return self._task.name

    @property
    def func(self) -> Callable:
        """Underlying function."""
        return self._task.func

    @property
    def src(self) -> str:
        """Source code of the task."""
        return self._task.src

    @property
    def output(self) -> object:
        """Output of the task."""
        return self._output

    @property
    def input(self) -> dict:
        """Input parameters of the task."""
        return self._input

    @property
    def start_time(self) -> datetime | None:
        """Time when the task has started."""
        return self._start_time

    @property
    def end_time(self) -> datetime | None:
        """Time when the task has ended regularly or failed."""
        return self._end_time

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, TaskResult):
            return NotImplemented
        return (
            self._task == value._task
            and self.output == value.output
            and self.input == value.input
        )

    def __repr__(self) -> str:
        attrs = ", ".join(
            [
                f"name={self.name}",
                f"output={self.output}",
                f"input={self.input}",
                f"func={self.func}",
            ],
        )
        return f"TaskResult({attrs})"

    def __str__(self) -> str:
        return f"TaskResult({self.name})"

    def _repr_pretty_(self, p, cycle):  # noqa: ANN001, ANN202, ARG002
        # For Notebooks
        p.text(str(self))


T = ParamSpec("T")
B = TypeVar("B")


class task_(Generic[T, B]):  # noqa: N801
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
        self._func = func
        self._name: str = name if name is not None else func.__name__
        self.__doc__ = func.__doc__
        self._options = get_and_validate_param_type(self._func, "options", BaseOptions)

    @property
    def src(self) -> str:
        """Source code of the task."""
        src = inspect.getsource(self._func)
        return textwrap.dedent(src)

    @property
    def func(self) -> Callable:
        """Underlying Python function."""
        return self._func

    @property
    def name(self) -> str:
        """The name of the task."""
        return self._name

    def __call__(self, *args: T.args, **kwargs: T.kwargs) -> B:  # noqa: D102
        ctx = _context.TaskExecutorContext.get_active()
        if ctx is None:
            return self._func(*args, **kwargs)
        return cast(B, ctx.execute_task(self, *args, **kwargs))

    def __repr__(self):
        return f"Task(name={self.name})"


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

    If the decorated function is used outside of an workflow related context, or
    within another task, the underlying behavior does not change.

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
