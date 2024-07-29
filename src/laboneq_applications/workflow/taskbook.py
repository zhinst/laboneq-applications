"""This module defines objects related to taskbook.

A taskbook is a collection of tasks and their records.

Whenever a task is executed within a taskbook, its'
arguments, result and other relevant information is
saved into the taskbook records.
"""

from __future__ import annotations

import inspect
import textwrap
import threading
from functools import update_wrapper
from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    Generic,
    TypeVar,
    cast,
)

from typing_extensions import ParamSpec

from laboneq_applications.workflow import _utils
from laboneq_applications.workflow._context import (
    TaskExecutor,
    TaskExecutorContext,
)
from laboneq_applications.workflow.exceptions import WorkflowError
from laboneq_applications.workflow.options import (
    TaskBookOptions,
    get_and_validate_param_type,
)
from laboneq_applications.workflow.task import (
    Task,
    attach_storage_callback,
    task_,
)
from laboneq_applications.workflow.taskview import TaskView

if TYPE_CHECKING:
    from typing import Callable


class _TaskBookStopExecution(BaseException):
    """Raised while a task book is running to end the execution."""


ReturnType = TypeVar("ReturnType")


class TaskBook(Generic[ReturnType]):
    """A taskbook.

    A taskbook is a collection of executed tasks and their results.
    """

    def __init__(
        self,
        input: dict | None = None,  # noqa: A002
    ) -> None:
        self._input = input or {}
        self._tasks: list[Task] = []
        self._output: ReturnType | None = None

    @property
    def tasks(self) -> TaskView:
        """Task entries of the taskbook.

        The ordering of the tasks is the order of the execution.

        Tasks is a `Sequence` of tasks, however item lookup
        is modified to support the following cases:

        Example:
            ```python
            taskbook = my_taskbook()
            taskbook.tasks["run_experiment"]  # First task of name 'run_experiment'
            taskbook.tasks["run_experiment", :]  # All tasks named 'run_experiment'
            taskbook.tasks["run_experiment", 1:5]  # Slice tasks named 'run_experiment'
            taskbook.tasks[0]  # First executed task
            taskbook.tasks[0:5]  # Slicing

            taskbook.tasks.unique() # Unique task names
            ```
        """
        return TaskView(self._tasks)

    @property
    def input(self) -> dict:
        """Input parameters of the taskbook."""
        return self._input

    @property
    def output(self) -> ReturnType:
        """Output of the taskbook."""
        return cast(ReturnType, self._output)

    def add_entry(self, task: Task) -> None:
        """Add an entry to the taskbook.

        Arguments:
            task: Task entry.
        """
        attach_storage_callback(task, self)
        self._tasks.append(task)

    def __repr__(self) -> str:
        attrs = ", ".join(
            [
                f"output={self.output}",
                f"input={self.input}",
                f"tasks={self.tasks!r}",
            ],
        )
        return f"TaskBook({attrs})"

    def __str__(self) -> str:
        return f"Taskbook\nTasks: {self.tasks}"

    def _repr_pretty_(self, p, cycle):  # noqa: ANN001, ANN202, ARG002
        # For Notebooks
        p.text(str(self))


class _TaskBookExecutor(TaskExecutor):
    """A taskbook executor."""

    def __init__(
        self,
        taskbook: TaskBook,
        options: TaskBookOptions,
    ) -> None:
        self.taskbook = taskbook
        if isinstance(TaskExecutorContext.get_active(), _TaskBookExecutor):
            # TODO: Should nested books append to the top level or?
            raise NotImplementedError("Taskbooks cannot be nested.")
        self.options = options
        self._run_until = self.options.run_until

    def __enter__(self):
        TaskExecutorContext.enter(self)

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        TaskExecutorContext.exit()
        return isinstance(exc_value, _TaskBookStopExecution)

    def execute_task(
        self,
        task: task_,
        *args: object,
        **kwargs: object,
    ) -> object:
        # TODO: Error handling and saving of the exception during execution
        if hasattr(self.options, task.name):
            if task.has_opts:
                kwargs["options"] = getattr(self.options, task.name)
            else:
                raise ValueError(f"Task {task.name} does not require options.")

        result = task._run(*args, **kwargs)
        self.taskbook.add_entry(result)
        if self._run_until is not None and task.name == self._run_until:
            raise _TaskBookStopExecution(task.name)
        return result.output


class _ContextStorage(threading.local):
    # NOTE: Subclassed for type hinting
    results: ClassVar[dict[str, TaskBook]] = {}


_results = _ContextStorage()

Parameters = ParamSpec("Parameters")


class taskbook_(Generic[Parameters, ReturnType]):  # noqa: N801
    """Task book wrapper for a function."""

    def __init__(
        self,
        func: Callable[Parameters, ReturnType],
    ) -> None:
        self._func = func
        self.__doc__ = self._func.__doc__
        self._options = get_and_validate_param_type(func)

    @property
    def func(self) -> Callable[Parameters, ReturnType]:
        """The underlying function."""
        return self._func

    @property
    def src(self) -> str:
        """Source code of the underlying function."""
        src = inspect.getsource(self._func)
        return textwrap.dedent(src)

    def _func_full_path(self) -> str:
        return ".".join([self.func.__module__, self.func.__qualname__])

    def __call__(  # noqa: D102
        self,
        *args: Parameters.args,
        **kwargs: Parameters.kwargs,
    ) -> TaskBook[ReturnType]:
        if self._options is not None:
            opts_input = kwargs.get("options", self._options())
            if isinstance(opts_input, dict):
                opt = self._options(**opts_input)
            elif isinstance(opts_input, self._options):
                opt = opts_input
            else:
                raise TypeError(
                    "Options must be a dictionary or an instance of TaskBookOptions.",
                )
        else:
            opt = TaskBookOptions()
        book = TaskBook(
            input=_utils.create_argument_map(self.func, *args, **kwargs),
        )

        with _TaskBookExecutor(
            book,
            options=opt,
        ):
            try:
                book._output = self._func(*args, **kwargs)
            except Exception:
                _results.results[self._func_full_path()] = book
                raise
        return book

    def recover(self) -> TaskBook[ReturnType]:
        """Recover the taskbook of the latest run that raised an exception.

        The value will be populated only if an exception is raised
        from the taskbook function.
        Getting the latest value will pop it from the memory and only one
        result is stored per taskbook.

        Returns:
            Latest taskbook that raised an exception.

        Raises:
            WorkflowError: Taskbook has no previous record.
        """
        try:
            return _results.results.pop(self._func_full_path())
        except KeyError as error:
            raise WorkflowError("Taskbook has no previous record.") from error


def taskbook(
    func: Callable[Parameters, ReturnType] | None = None,
) -> taskbook_[Parameters, ReturnType]:
    """A decorator to turn a function into a taskbook.

    When a function is marked as a taskbook, it will record
    each task's information.

    ### Storing the results in case of an error

    When a function wrapped as a taskbook raises an Exception,
    the partial result up until that point can be retrieved by using
    `.recover()` method.

    Arguments:
        func: Function to be marked as a taskbook.
    """
    out = update_wrapper(
        taskbook_(func),
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
    return cast(taskbook_[Parameters, ReturnType], out)
