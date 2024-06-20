"""This module defines objects related to taskbook.

A taskbook is a collection of tasks and their records.

Whenever a task is executed within a taskbook, its'
arguments, result and other relevant information is
saved into the taskbook records.

## Example

### Building and running the taskbook

```python
from laboneq_applications.workflow import task, taskbook

def addition(x, y):
    return x + y

@task
def my_task(x):
   return x + 1

@taskbook
def my_taskbook(x):
    x = addition(1, x)
    my_task(x)
    return "success"

result = my_taskbook(x=5)
```

### Inspecting the results

The results of all tasks are recorded and may be inspected later
using the result object returned by the taskbook.

The taskbook output and the information of each task execution
can be accessed from the result object:

```python
>>> result.output
"success"
>>> result.tasks[0].output
7
>>> result.tasks[0].args
(6,)
>>> result.tasks["addition"]
Task(name="addition")
>>> taskbook.tasks["addition", :]
[Task(name="addition")]
>>> taskbook.tasks.unique()
["addition", "my_task"]
```

As `addition()` was not marked as a task, it has no records in the taskbook.

### Running functions as tasks

If a function is a normal Python function and you'd wish to run it as a task within the
taskbook without adding the `task()` decorator, it can be done by wrapping the
function with it.

This also works if you wish to save only specific calls to the function.

```python
@taskbook
def my_taskbook(x):
    x = task(addition)(1, x)  # Record will be saved
    y = addition(2, x)  # No record saved
```

Now the normal Python function `addition()` wrapped in `task()` will have records
within the taskbook.
"""

from __future__ import annotations

import copy
import inspect
import textwrap
import threading
from collections.abc import Sequence
from functools import update_wrapper
from typing import TYPE_CHECKING, Callable, ClassVar, Generic, TypeVar, overload

from typing_extensions import ParamSpec

from laboneq_applications.workflow import _utils
from laboneq_applications.workflow._context import (
    ExecutorContext,
    LocalContext,
    get_active_context,
)
from laboneq_applications.workflow.exceptions import WorkflowError

if TYPE_CHECKING:
    from typing import Callable

    from laboneq_applications.workflow.task import task_


class Task:
    """A task.

    The instance holds execution information of an task when it
    was executed in a taskbook.

    Attributes:
        name: The name of the task.
        func: Underlying Python function.
        output: The output of the function.
        parameters: Input parameters of the task.
    """

    def __init__(
        self,
        task: task_,
        output: object,
        parameters: dict | None = None,
    ) -> None:
        self._task = task
        self._output = output
        self._parameters = parameters or {}
        self._taskbook: TaskBook | None = None

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
    def parameters(self) -> dict:
        """Input parameters of the task."""
        return self._parameters

    def rerun(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Rerun the task.

        Rerunning the task appends the results into the attached taskbook,
        and returns the return value of the task.

        The given parameters overwrite the original run parameters of the task,
        which can be inspected with `Task.parameters`. Therefore it is possible
        to supply only specific parameters that one wishes to change and rerun
        the task.
        It is recommended that in the case of an partial parameters, keyword
        arguments are used.

        Arguments:
            *args: Arguments forwarded into the task.
            **kwargs: Keyword arguments forwarded into the original task.

        Returns:
            The return value of the task.
        """
        # TODO: Fill args, kwargs with previous and overwrite with given arguments.
        kwargs = kwargs if kwargs is not None else {}
        args = args if args is not None else ()
        sig = inspect.signature(self.func)
        args_partial = sig.bind_partial(*args, **kwargs)
        params = self.parameters | args_partial.arguments
        r = self.func(**params)
        if self._taskbook is not None:
            entry = Task(
                task=self._task,
                output=r,
                parameters=params,
            )
            self._taskbook.add_entry(entry)
        return r

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Task):
            return NotImplemented
        return (
            self._task == value._task
            and self.output == value.output
            and self.parameters == value.parameters
        )

    def __repr__(self) -> str:
        attrs = ", ".join(
            [
                f"name={self.name}",
                f"output={self.output}",
                f"parameters={self.parameters}",
                f"func={self.func}",
            ],
        )
        return f"Task({attrs})"

    def __str__(self) -> str:
        return f"Task({self.name})"


class TasksView(Sequence):
    """A view of tasks.

    This class provides a view into tasks.

    Arguments:
        tasks: List of tasks.

    The class is a `Sequence` of tasks, however item lookup
    is modified to support the following cases:

        - Lookup by index and slicing
        - Lookup by name (string)
        - Lookup by name and slicing
    """

    def __init__(self, tasks: list[Task] | None = None) -> None:
        self._tasks = tasks or []

    def unique(self) -> set[str]:
        """Return unique names of the tasks."""
        return {t.name for t in self._tasks}

    def __repr__(self) -> str:
        return repr(self._tasks)

    def __str__(self) -> str:
        return ", ".join([str(t) for t in self._tasks])

    @overload
    def __getitem__(self, item: tuple[str, int | slice] | slice) -> list[Task]: ...

    @overload
    def __getitem__(self, item: str | int) -> Task: ...

    def __getitem__(
        self,
        item: str | int | tuple[str, int | slice] | slice,
    ) -> Task | list[Task]:
        """Get a single or multiple tasks.

        Arguments:
            item: Index, name of the task, slice or a tuple.

                If index or name is given, the return value will be a single `Task`,
                the first one found in the sequence.

                tuple: A tuple of format (<name>, <index/slice>) will return
                    list of tasks.

        Returns:
            Task or list of tasks, depending on the input filter.

        Raises:
            KeyError: Task by name was not found.
            IndexError: Task by index was not found.
        """
        if isinstance(item, str):
            try:
                return next(t for t in self._tasks if t.name == item)
            except StopIteration:
                raise KeyError(item) from None
        if isinstance(item, tuple):
            items = [t for t in self._tasks if t.name == item[0]]
            if not items:
                raise KeyError(item[0])
            return items[item[1]]
        return self._tasks[item]

    def __len__(self) -> int:
        return len(self._tasks)

    def __eq__(self, other: object) -> bool:
        return self._tasks == other


class TaskBook:
    """A taskbook.

    A taskbook is a collection of executed tasks and their results.
    """

    def __init__(self, parameters: dict | None = None) -> None:
        self._parameters = parameters or {}
        self._tasks: list[Task] = []
        self._output: object | None = None

    @property
    def tasks(self) -> TasksView:
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
        return TasksView(self._tasks)

    @property
    def parameters(self) -> dict:
        """Input parameters of the taskbook."""
        return self._parameters

    @property
    def output(self) -> object | None:
        """Output of the taskbook."""
        return self._output

    def add_entry(self, entry: Task) -> None:
        """Add an entry to the taskbook.

        Arguments:
            entry: Task entry.
        """
        if entry._taskbook is not None:
            raise WorkflowError("Task is already attached to an taskbook.")
        entry._taskbook = self
        self._tasks.append(entry)

    def __repr__(self) -> str:
        attrs = ", ".join(
            [
                f"output={self.output}",
                f"parameters={self.parameters}",
                f"tasks={self.tasks!r}",
            ],
        )
        return f"TaskBook({attrs})"

    def __str__(self) -> str:
        return f"Taskbook\nTasks: {self.tasks}"


class _ContextStorage(threading.local):
    # NOTE: Subclassed for type hinting
    results: ClassVar[dict[str, TaskBook]] = {}


_results = _ContextStorage()

Parameters = ParamSpec("Parameters")
ReturnType = TypeVar("ReturnType")


class taskbook_(Generic[Parameters, ReturnType]):  # noqa: N801
    """Task book wrapper for a function."""

    def __init__(self, func: Callable[Parameters, ReturnType]) -> None:
        self._func = func
        self.__doc__ = self._func.__doc__

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

    def __call__(self, *args: Parameters.args, **kwargs: Parameters.kwargs) -> TaskBook:  # noqa: D102
        ctx = get_active_context()
        if isinstance(ctx, _TaskBookExecutor):
            # TODO: Should nested books append to the top level or?
            raise NotImplementedError("Taskbooks cannot be nested.")
        book = TaskBook(
            parameters=_utils.create_argument_map(self.func, *args, **kwargs),
        )
        with LocalContext.scoped(
            _TaskBookExecutor(book, options=kwargs.get("options", None)),
        ):
            try:
                book._output = self._func(*args, **kwargs)
            except Exception:
                _results.results[self._func_full_path()] = book
                raise
        return book

    def recover(self) -> TaskBook:
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
    func: Callable[Parameters, ReturnType],
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
    return update_wrapper(
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


class _TaskBookExecutor(ExecutorContext):
    """A taskbook executor."""

    def __init__(self, taskbook: TaskBook, options: dict | None) -> None:
        self.taskbook = taskbook
        if options is None:
            self._options = TaskBookOptions()
        else:
            self._options = TaskBookOptions(**options)

    def execute_task(
        self,
        task: task_,
        *args: object,
        **kwargs: object,
    ) -> None:
        # TODO: Error handling and saving of the exception during execution
        if self._options.task_options(task.name) and task.has_opts:
            # if a task is called with options explicitly in the taskbook,
            # update the options with the task options.
            # Otherwise use the task options.
            # If no options are provided at the taskbook level,
            # don't update the options.
            kwargs.setdefault("options", {}).update(
                self._options.task_options(task.name),
            )

        r = task._run(*args, **kwargs)
        entry = Task(
            task=task,
            output=r,
            parameters=_utils.create_argument_map(task.func, *args, **kwargs),
        )
        self.taskbook.add_entry(entry)
        return r


class TaskBookOptions:
    """A class for organizing options for the taskbook."""

    _DELIMITER = "."
    _PREFIX = "task"

    def __init__(self, **kwargs) -> None:
        self._broadcast = {}  # contain broadcast options
        self._specific = {}  # contain task specific options
        for k, v in kwargs.items():
            key_prefix, key_sep, key_suffix = k.partition(self._DELIMITER)
            if key_prefix == self._PREFIX and key_sep == self._DELIMITER:
                self._specific[key_suffix] = v
            else:
                self._broadcast[k] = v

    def task_options(self, task_name: str) -> dict:
        """Return an option dict for a specific task following these rules.

        The original options include broadcast options and task-specific options.
        1. If the task_name is a not in the specific options, return a
        broadcast options dict.
        2. Otherwise, return a dict with broadcast options but overridden
        by the task-specific options.

        Args:
            task_name: The name of the task.

        Returns:
            dict: The options for the task.
        """
        res = copy.deepcopy(self._broadcast)
        if task_name not in self._specific:
            return res
        res.update(self._specific[task_name])
        return res
