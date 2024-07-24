"""Tasks used within workflows."""

from __future__ import annotations

import inspect
import textwrap
from functools import partial, update_wrapper
from typing import Callable, Generic, Protocol, TypeVar, overload

from typing_extensions import ParamSpec

from laboneq_applications.workflow import _context, _utils
from laboneq_applications.workflow.exceptions import WorkflowError


class TaskStorageProtocol(Protocol):
    """A storage to save task entries."""

    def add_entry(self, task: Task) -> None:
        """Callback to attach a task into the given storage."""
        ...


def attach_storage_callback(task: Task, storage: TaskStorageProtocol) -> None:
    """Add a callback to attach a task into the given storage."""
    if task._task_store is not None:
        raise WorkflowError("Task is already attached to an storage object.")
    task._task_store = storage


class Task:
    """A task.

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
        self._task_store: TaskStorageProtocol | None = None
        self._figures = []

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

    def rerun(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Rerun the task.

        Rerunning the task appends the results into the attached storage,
        and returns the return value of the task.

        The given input parameters overwrite the original run parameters of the task,
        which can be inspected with `Task.input`. Therefore, it is possible
        to supply only specific parameters that one wishes to change and rerun
        the task.
        It is recommended that in the case of partial parameters, keyword
        arguments are used.

        Arguments:
            *args: Arguments forwarded into the task.
            **kwargs: Keyword arguments forwarded into the original task.

        Returns:
            The return value of the task.
        """
        kwargs = kwargs if kwargs is not None else {}
        args = args if args is not None else ()
        sig = inspect.signature(self.func)
        args_partial = sig.bind_partial(*args, **kwargs)
        params = self.input | args_partial.arguments
        r = self.func(**params)
        if self._task_store is not None:
            entry = Task(
                task=self._task,
                output=r,
                input=params,
            )
            self._task_store.add_entry(entry)
        return r

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Task):
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
        return f"Task({attrs})"

    def __str__(self) -> str:
        return f"Task({self.name})"

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

    @property
    def has_opts(self) -> bool:
        """Return `True` if the task has options in its arguments."""
        return "options" in inspect.signature(self._func).parameters

    def _run(self, *args: T.args, **kwargs: T.kwargs) -> Task:
        task = Task(
            task=self,
            output=None,
            input=_utils.create_argument_map(self._func, *args, **kwargs),
        )
        r = self._func(*args, **kwargs)
        task._output = r
        return task

    def __call__(self, *args: T.args, **kwargs: T.kwargs) -> B:  # noqa: D102
        ctx = _context.TaskExecutorContext.get_active()
        if ctx is None:
            return self._func(*args, **kwargs)
        return ctx.execute_task(self, *args, **kwargs)

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
